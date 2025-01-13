from datetime import datetime
from typing import Dict

from loguru import logger
from tqdm import tqdm
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.crawler import ArxivClient
from llm_engineering.domain.documents import ExpertDocument, PaperDocument


@step
def crawl_links(expert: ExpertDocument, links: list[str]) -> Annotated[list[str], "crawled_links"]:
    """
    Crawl arxiv paper links and store them in MongoDB.
    
    Args:
        expert: The expert document these papers are associated with
        links: List of arxiv paper URLs
        
    Returns:
        list[str]: The processed links
    """
    client = ArxivClient()
    logger.info(f"Starting to crawl {len(links)} arxiv paper(s).")

    metadata = {}
    successful_crawls = 0
    
    for link in tqdm(links):
        success = client.process_paper(link, expert)
        successful_crawls += success
        metadata = _add_to_metadata(metadata, success)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)

    logger.info(f"Successfully crawled {successful_crawls} / {len(links)} papers.")
    return links


def _add_to_metadata(metadata: Dict, success: bool) -> Dict:
    """Update metadata with crawl results."""
    metadata["successful"] = metadata.get("successful", 0) + success
    metadata["total"] = metadata.get("total", 0) + 1
    return metadata
