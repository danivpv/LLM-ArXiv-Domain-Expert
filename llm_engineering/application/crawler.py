import re
import subprocess
from datetime import datetime
from io import BytesIO
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import requests
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (EasyOcrOptions,
                                                PdfPipelineOptions,
                                                TesseractCliOcrOptions)
from docling.document_converter import (ConversionError, DocumentConverter,
                                        PdfFormatOption)
from loguru import logger

from llm_engineering.domain.documents import ExpertDocument, PaperDocument


class ArxivClient:
    """
    Singleton client for managing arxiv API connections and OCR processing.
    Handles paper metadata fetching, PDF downloading and OCR processing.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize all long-lived objects here
            cls._instance._session = requests.Session()
            cls._instance._ocr = OCRClient()  # Single OCR client instance
        return cls._instance

    def _fetch_metadata(self, arxiv_url: str) -> Tuple[bool, Dict]:
        # Extract arXiv ID from URL
        arxiv_id = arxiv_url.split('/')[-1]
        
        # Fetch metadata
        metadata_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        metadata_response = self._session.get(metadata_url)
        
        if metadata_response.status_code != 200:
            logger.error(f"Failed to fetch metadata for {arxiv_url}: {metadata_response.status_code}")
            return False, {}
        
        logger.info(f"Fetched metadata for {arxiv_url} successfully")

        # Parse metadata
        metadata_root = ET.fromstring(metadata_response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = metadata_root.find('.//atom:entry', namespace)
        
        if entry is None:
            logger.error(f"No metadata entry found for {arxiv_url}")
            return False, {}
            
        title = entry.find('atom:title', namespace).text
        published = entry.find('atom:published', namespace).text

        return True, {
            "title": title,
            "published_at": published
        }
    
    def _fetch_pdf(self, arxiv_url: str) -> Tuple[bool, Dict]:
        # Fetch PDF
        pdf_response = self._session.get(arxiv_url)
        if pdf_response.status_code != 200:
            logger.error(f"Failed to download PDF from {arxiv_url}")
            return False, {}
        
        logger.info(f"Downloaded PDF from {arxiv_url} successfully")
        return True, pdf_response.content

    def _fetch_paper(self, arxiv_url: str) -> Tuple[bool, Dict]:
        """
        Fetch paper metadata and PDF content from arxiv.
        
        Args:
            arxiv_url: Full arxiv URL
            
        Returns:
            Tuple[bool, Dict]: Success status and paper info dictionary
        """
        success, metadata = self._fetch_metadata(arxiv_url)
        if not success:
            return False, {}
        
        success, pdf_response_content = self._fetch_pdf(arxiv_url)
        if not success:
            return False, {}
        
        return True, {
            "title": metadata["title"],
            "published_at": metadata["published_at"],
            "content": pdf_response_content, 
            "link": arxiv_url
        }
            
    def _process_pdf(self, bytes_pdf_stream: bytes, title: str) -> Tuple[bool, Dict]:
        # Process PDF with OCR
        pdf_stream = BytesIO(bytes_pdf_stream) # get file like object
        return self._ocr.ocr(title, pdf_stream)
        
    def process_paper(self, link: str, expert: ExpertDocument) -> bool:
        """Process a single arxiv paper."""
        # Check if paper already exists
        existing_paper = PaperDocument.objects(link=link).first()
        if existing_paper:
            logger.info(f"Paper already exists in database: {link}")
            return True

        try:
            success, paper_info = self._fetch_paper(link)
            if not success:
                return False
            
            markdown_content = self._process_pdf(
                paper_info["content"], 
                paper_info["title"]
                )
            if markdown_content is None:
                return False
                
            paper = PaperDocument(
                title=self._sanitize_title(paper_info["title"]),
                content=markdown_content,  # Contains OCR'd text in markdown format
                expert_id=expert,
                link=paper_info["link"],
                published_at=datetime.fromisoformat(paper_info["published_at"]),
            ).save()
            
            logger.info(f"Successfully processed paper: {paper.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper {link}: {str(e)}")
            return False

    @staticmethod
    def _sanitize_title(title: str, max_length: int = 100) -> str:
        """
        Sanitize title by removing special characters and limiting length.
        
        Args:
            title: The original title
            max_length: Maximum length for the title (easyocr 100)
        
        Returns:
            A sanitized title safe for MongoDB
        """
        # Remove newlines and tabs using regex
        cleaned = re.sub(r'[\n\t]', '', title)
        # Remove special characters, keeping only alphanumeric, spaces, and common punctuation
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
        # Remove consecutive spaces, trim to max length, and remove trailing spaces
        cleaned = re.sub(r' +', ' ', cleaned)[:max_length].strip(' ')
        return cleaned
    

class OCRClient:
    def __init__(self):
        self.ocr_backends = ["easyocr", "tesseract_cli"]
        if not self._check_tesseract_installed():
            self.ocr_backends.remove("tesseract_cli")
        logger.info(f"Initialize docling's OCRClient with backends: {self.ocr_backends}")
        self.pipeline_options = {}

    def _check_tesseract_installed(self):
        """Check if Tesseract is installed and the command is available."""
        try:
            subprocess.run(['tesseract', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Tesseract is installed and available.")
            return True
        except subprocess.CalledProcessError:
            logger.warning("Tesseract command is not available.")
            return False

    def _get_pipeline_options(self, ocr_backend: str) -> PdfPipelineOptions:
        """
        Returns PdfPipelineOptions based on the specified OCR backend.

        Args:
            ocr_backend (str): The OCR backend to use ('tesseract', 'tesseract_cli', or 'easyocr').

        Returns:
            PdfPipelineOptions: Configured pipeline options.
        """
        if ocr_backend not in self.pipeline_options:
            pipeline_options = PdfPipelineOptions()

            if ocr_backend == "easyocr":
                pipeline_options.ocr_options = EasyOcrOptions(
                    lang=["en"]
                )

            if ocr_backend == "tesseract_cli":
                pipeline_options.ocr_options = TesseractCliOcrOptions(
                    lang=["eng"]
                )

            self.pipeline_options[ocr_backend] = pipeline_options

        return self.pipeline_options[ocr_backend]

    def _ocr(self, title, pdf_stream, backends_to_try: list[str]) -> str | None:
        current_backend = backends_to_try.pop(0)

        logger.info(f"Starting conversion of {title} with OCR backend: {current_backend}")

        pipeline_options = self._get_pipeline_options(current_backend)
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        return doc_converter.convert(DocumentStream(name=title, stream=pdf_stream))

    def ocr(self, title, pdf_stream, backends_to_try: list[str] | None = None) -> str | None:
        if backends_to_try is None:
            backends_to_try = self.ocr_backends.copy()

        try: 
            conv_result = self._ocr(title, pdf_stream, backends_to_try=backends_to_try)
            return conv_result.document.export_to_markdown()
        
        except ConversionError as e:
            if backends_to_try:
                return self._ocr(title, pdf_stream, backends_to_try=backends_to_try)
            else:
                logger.error(f"Conversion failed for {title} with all backends")
                return None
        
  

