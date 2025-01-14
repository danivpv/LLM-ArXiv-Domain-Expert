from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.preprocessing import DocumentCleaner
from llm_engineering.domain.cleaned_documents import CleanedPaperDocument


@step
def clean_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        cleaned_document = DocumentCleaner.clean(document)
        cleaned_documents.append(cleaned_document)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="cleaned_documents", metadata=_get_metadata(cleaned_documents))

    return cleaned_documents


def _get_metadata(cleaned_documents: list[CleanedPaperDocument]) -> dict:
    """Get metadata about the cleaned documents."""
    return {
        "num_documents": len(cleaned_documents),
        "expert": str(cleaned_documents[0].expert_id),  # Convert UUID to string
    }
