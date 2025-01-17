from loguru import logger
from zenml import pipeline

from steps.etl import crawl_links, get_or_create_expert


@pipeline
def arxiv_data_etl(domain: str, links: list[str]) -> str:
    """
    Pipeline for crawling arxiv papers and storing them in MongoDB.

    Args:
        domain: The domain of expertise (e.g. "machine_learning")
        links: List of arxiv paper URLs to process

    Returns:
        str: The invocation ID of the last step
    """
    logger.info(f"Starting pipeline for domain: {domain}")
    expert = get_or_create_expert(domain)
    last_step = crawl_links(expert=expert, links=links)

    return last_step.invocation_id
