import re
import subprocess
import time
from datetime import datetime
from io import BytesIO
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

import requests
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import (AcceleratorDevice,
                                                AcceleratorOptions,
                                                PdfPipelineOptions,
                                                TesseractCliOcrOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption
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

    def get_paper_info(self, arxiv_url: str) -> Tuple[bool, Dict]:
        """
        Fetch paper metadata and PDF content from arxiv.
        
        Args:
            arxiv_url: Full arxiv URL
            
        Returns:
            Tuple[bool, Dict]: Success status and paper info dictionary
        """
        try:
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
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            metadata_root = ET.fromstring(metadata_response.content)
            entry = metadata_root.find('.//atom:entry', namespace)
            
            if entry is None:
                logger.error(f"No metadata entry found for {arxiv_url}")
                return False, {}
                
            title = entry.find('atom:title', namespace).text
            published = entry.find('atom:published', namespace).text
            
            # Fetch PDF
            pdf_response = self._session.get(arxiv_url)
            if pdf_response.status_code != 200:
                logger.error(f"Failed to download PDF from {arxiv_url}")
                return False, {}
            
            logger.info(f"Downloaded PDF from {arxiv_url} successfully")

            # Process PDF with OCR
            pdf_stream = BytesIO(pdf_response.content)
            markdown_content = self._ocr.ocr(title, pdf_stream)
            
            if markdown_content is None:
                logger.error(f"OCR processing failed for {title}")
                return False, {}
            
            logger.info(f"OCR processing completed for {title}")
                
            return True, {
                "title": title,
                "published_at": published,
                "content": markdown_content,  # Store OCR result as markdown
                "link": arxiv_url
            }
            
        except Exception as e:
            logger.error(f"Error processing {arxiv_url}: {str(e)}")
            return False, {}
        
    def process_paper(self, link: str, expert: ExpertDocument) -> bool:
        """Process a single arxiv paper."""
        try:
            success, paper_info = self.get_paper_info(link)
            if not success:
                return False
                
            paper = PaperDocument(
                title=self.sanitize_filename(paper_info["title"]),
                content=paper_info["content"],  # Contains OCR'd text in markdown format
                expert_id=expert,
                link=paper_info["link"],
                published_at=datetime.fromisoformat(paper_info["published_at"])
            ).save()
            
            logger.info(f"Successfully processed paper: {paper.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing paper {link}: {str(e)}")
            return False

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 100) -> str:
        """
        Sanitize filename by removing special characters and limiting length.
        
        Args:
            filename: The original filename
            max_length: Maximum length for the filename (default 100)
        
        Returns:
            A sanitized filename safe for Windows
        """
        # Remove newlines and tabs first
        cleaned = filename.replace('\n', '').replace('\t', '')
        # Remove special characters, keeping only alphanumeric, spaces, and common punctuation
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
        # Replace spaces with underscores
        cleaned = cleaned.replace(' ', '_')
        # Remove consecutive underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Trim to max length
        cleaned = cleaned[:max_length]
        # Remove trailing underscores
        cleaned = cleaned.rstrip('_')
        return cleaned
    

class OCRClient:
    ocr_backends: list[str] = ["default", "tesseract_cli"]

    def __init__(self):
        if not self.check_tesseract_installed():
            self.ocr_backends.remove("tesseract_cli")
        self.pipeline_options = {}

    def check_tesseract_installed(self):
        """Check if Tesseract is installed and the command is available."""
        try:
            subprocess.run(['tesseract', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info("Tesseract is installed and available.")
            return True
        except subprocess.CalledProcessError:
            logger.warning("Tesseract command is not available.")
            return False

    def get_pipeline_options(self, ocr_backend: str) -> PdfPipelineOptions:
        """
        Returns PdfPipelineOptions based on the specified OCR backend.

        Args:
            ocr_backend (str): The OCR backend to use ('tesseract', 'tesseract_cli', or 'default').

        Returns:
            PdfPipelineOptions: Configured pipeline options.
        """
        if ocr_backend not in self.pipeline_options:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.ocr_options.lang = ["en"]
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.AUTO
            )

            if ocr_backend == "tesseract_cli":
                pipeline_options.ocr_options = TesseractCliOcrOptions()

            self.pipeline_options[ocr_backend] = pipeline_options

        return self.pipeline_options[ocr_backend]

    def ocr(self, title, pdf_stream) -> str | None:
        backends_to_try = self.ocr_backends.copy()
        current_backend = backends_to_try.pop(0)

        logger.info(f"Starting conversion of {title} with OCR backend: {current_backend}")

        pipeline_options = self.get_pipeline_options(current_backend)
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        try: 
            start_time = time.time()
            conv_result = doc_converter.convert(DocumentStream(name=title, stream=pdf_stream))
            elapsed_time = time.time() - start_time
            logger.info(f"Document converted in {elapsed_time:.2f} seconds using {current_backend}.")

        except Exception as e:
            if backends_to_try:
                logger.warning(f"Conversion failed for {title} with {current_backend}, trying next backend")
                return self.ocr(title, pdf_stream, backends_to_try)
            else:
                logger.error(f"Conversion failed for {title} with all backends")
                return None
        
        return conv_result.document.export_to_markdown()

if __name__ == "__main__":
    ocr_client = OCRClient()
    

