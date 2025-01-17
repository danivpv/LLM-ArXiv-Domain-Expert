import re
import subprocess
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import requests
import yaml
from docling.datamodel.base_models import ConversionStatus, DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from loguru import logger


def get_pipeline_options(ocr_backend: str) -> PdfPipelineOptions:
    """
    Returns PdfPipelineOptions based on the specified OCR backend.

    Args:
        ocr_backend (str): The OCR backend to use ('tesseract', 'tesseract_cli', or 'default').

    Returns:
        PdfPipelineOptions: Configured pipeline options.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["en"]
    """ pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=1, device=AcceleratorDevice.AUTO
    ) """

    if ocr_backend == "tesseract_cli":
        pipeline_options.ocr_options = TesseractCliOcrOptions()

    return pipeline_options


def convert_document(input_path: Path, output_dir: Path, ocr_backend: str):
    """
    Converts the document using the specified OCR backend and saves the output.

    Args:
        input_path (Path): Path to the input PDF document.
        output_dir (Path): Directory to save the output files.
        ocr_backend (str): The OCR backend to use.
    """
    logger.info(f"Starting conversion with OCR backend: {ocr_backend}")

    pipeline_options = get_pipeline_options(ocr_backend)
    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()
    conv_result = doc_converter.convert(input_path)
    elapsed_time = time.time() - start_time
    logger.info(f"Document converted in {elapsed_time:.2f} seconds using {ocr_backend}.")

    doc_filename = conv_result.input.file.stem
    backend_output_dir = output_dir / ocr_backend
    backend_output_dir.mkdir(parents=True, exist_ok=True)

    # Export Text format:
    with (backend_output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_text())

    # Export Markdown format:
    with (backend_output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_markdown())


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
    cleaned = filename.replace("\n", "").replace("\t", "")
    # Remove special characters, keeping only alphanumeric, spaces, and common punctuation
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    # Replace spaces with underscores
    cleaned = cleaned.replace(" ", "_")
    # Remove consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Trim to max length
    cleaned = cleaned[:max_length]
    # Remove trailing underscores
    cleaned = cleaned.rstrip("_")
    return cleaned


def export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0
    files_to_process = []

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = sanitize_filename(conv_res.input.file.stem)

            # Export Docling document format to markdown:
            with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
                fp.write(conv_res.document.export_to_markdown())

            # Export Docling document format to text:
            with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
                fp.write(conv_res.document.export_to_markdown(strict_text=True))

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.info(f"Document {conv_res.input.file} was partially converted with the following errors:")
            for item in conv_res.errors:
                logger.info(f"\t{item.error_message}")
            partial_success_count += 1
            files_to_process.append(conv_res.input.file)
        else:
            logger.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1
            files_to_process.append(conv_res.input.file)

    logger.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count, files_to_process


def convert_with_backend(input_doc_paths: list[Path], ocr_backend: str, output_dir: Path | str):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    logger.info(f"Starting conversion with OCR backend: {ocr_backend}")

    pipeline_options = get_pipeline_options(ocr_backend)
    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    start_time = time.time()

    conv_results = doc_converter.convert_all(
        input_doc_paths,
        raises_on_error=False,  # to let conversion run through all and examine results at the end
    )
    success_count, partial_success_count, failure_count, files_to_process = export_documents(
        conv_results, output_dir=output_dir
    )

    end_time = time.time() - start_time

    logger.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        logger.error(f"Conversion failed with {failure_count} failures and {partial_success_count} partial successes.")

    logger.info(f"Files to process: {files_to_process}")
    return files_to_process


def get_pdf_streams(urls: list[str]) -> dict[str, BytesIO]:
    pdf_streams = {}

    with requests.Session() as session:
        for url in urls:
            # Extract arXiv ID from URL
            arxiv_id = url.split("/")[-1]

            # Fetch metadata from arXiv API
            metadata_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            metadata_response = session.get(metadata_url)

            if metadata_response.status_code == 200:
                # Define the namespace
                namespace = {"atom": "http://www.w3.org/2005/Atom"}

                metadata_root = ET.fromstring(metadata_response.content)
                # Use proper namespace to find the title within entry
                entry = metadata_root.find(".//atom:entry", namespace)
                if entry is not None:
                    title = entry.find("atom:title", namespace).text
                    logger.info(f"Found paper title: {title}")
                else:
                    logger.warning(f"No entry found for {url}")
                    title = f"Untitled_{arxiv_id}"  # Fallback title
            else:
                logger.warning(f"Failed to fetch metadata for {url}: {metadata_response.status_code}")
                title = f"Error_{arxiv_id}"  # Fallback title

            # Download the PDF
            pdf_response = session.get(url)

            if pdf_response.status_code == 200:
                pdf_streams[title] = BytesIO(pdf_response.content)
                logger.info(f"Downloaded PDF from {url} successfully!")
            else:
                logger.warning(f"Failed to download PDF from {url}.")

    return pdf_streams


def check_tesseract_installed():
    """Check if Tesseract is installed and the command is available."""
    try:
        subprocess.run(["tesseract", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Tesseract is installed and available.")
        return True
    except subprocess.CalledProcessError:
        logger.warning("Tesseract command is not available.")
        return False


def load_arxiv_yaml(yaml_path: Path) -> dict[str, BytesIO]:
    try:
        with Path(yaml_path).open(mode="r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        urls = config.get("parameters", {}).get("links", [])
        if not urls:
            logger.error("No URLs found in arxiv.yaml")
            return

        logger.info(f"Found {len(urls)} URLs in arxiv.yaml")

    except FileNotFoundError:
        logger.error(f"Could not find arxiv.yaml at {yaml_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error parsing arxiv.yaml: {e}")
        return

    return urls


def main():
    import sys

    logger.add(sys.stderr, level="INFO")

    # Get the directory where the script is located
    dir_path = Path(__file__).parent
    """
    # load pdfs from local directory
    pdf_dir = dir_path / 'pdf'
    input_doc_paths = list(pdf_dir.glob('*.pdf')) 
    input_doc_paths = [DocumentStream(name=i.name, stream=BytesIO(Path(i).open("rb").read())) for i in input_doc_paths[:2]]
    """
    """
    # load pdfs from arxiv.yaml
    yaml_path = dir_path / 'arxiv.yaml'
    urls = load_arxiv_yaml(yaml_path)
    """
    urls = [
        "https://arxiv.org/pdf/2208.07339",
        "https://arxiv.org/pdf/2306.00978",
    ]

    pdf_streams = get_pdf_streams(urls)

    input_doc_paths = [DocumentStream(name=title, stream=stream) for title, stream in pdf_streams.items()]

    output_dir = dir_path / "output"
    if check_tesseract_installed():
        input_doc_paths = convert_with_backend(
            input_doc_paths,
            "tesseract_cli",
            output_dir=output_dir,
        )

    if len(input_doc_paths) > 0:
        convert_with_backend(input_doc_paths, "default", output_dir=output_dir)


if __name__ == "__main__":
    main()
