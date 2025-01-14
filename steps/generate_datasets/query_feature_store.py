from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from mongoengine import Document
from qdrant_client.http import exceptions
from typing_extensions import Annotated
from zenml import step

from llm_engineering.domain.cleaned_documents import CleanedPaperDocument


@step
def query_feature_store() -> Annotated[list, "queried_cleaned_documents"]:
    logger.info("Querying feature store.")

    results = fetch_all_data()

    cleaned_documents = [doc for query_result in results.values() for doc in query_result]

    return cleaned_documents


def fetch_all_data() -> dict[str, list[Document]]:
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(
                __fetch_papers,
            ): "papers",
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results


def __fetch_papers() -> list[CleanedPaperDocument]:
    return __fetch(CleanedPaperDocument)


def __fetch(cleaned_document_type: type[CleanedPaperDocument], limit: int = 1) -> list[CleanedPaperDocument]:
    try:
        cleaned_documents, next_offset = cleaned_document_type.bulk_find(limit=limit)
    except exceptions.UnexpectedResponse:
        return []

    while next_offset:
        documents, next_offset = cleaned_document_type.bulk_find(limit=limit, offset=next_offset)
        cleaned_documents.extend(documents)

    return cleaned_documents
