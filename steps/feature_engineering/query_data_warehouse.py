from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.domain.documents import ExpertDocument, PaperDocument


@step
def query_data_warehouse(
    domains: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    experts = []
    for domain in domains:
        logger.info(f"Querying data warehouse for domain: {domain}")

        expert = ExpertDocument.get_or_create(domain=domain)
        experts.append(expert)

        results = list(PaperDocument.bulk_find(query={"expert_id": expert.id}))
        documents.extend(results)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents


def _get_metadata(documents: list[PaperDocument]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    if documents:
        collection = PaperDocument.get_collection_name()
        metadata[collection] = {
            "num_documents": len(documents),
            "experts": list(set(str(doc.expert_id) for doc in documents))
        }

    return metadata
