from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application import utils
from llm_engineering.application.preprocessing import DocumentChunker, DocumentEmbedder
from llm_engineering.domain.chunks import PaperChunk
from llm_engineering.domain.embedded_chunks import EmbeddedPaperChunk


@step
def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)}

    embedded_chunks = []
    for document in cleaned_documents:
        chunks = DocumentChunker.chunk(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = DocumentEmbedder.embed(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="embedded_documents", metadata=metadata)

    return embedded_chunks


def _add_chunks_metadata(chunks: list[PaperChunk], metadata: dict) -> dict:
    """Add metadata from paper chunks including expert and document information."""
    for chunk in chunks:
        category = chunk.get_category()
        if category not in metadata:
            metadata[category] = {"documents": {}, "experts": {}, "num_chunks": 0}

        # Track document-level metadata
        doc_id = str(chunk.document_id)
        if doc_id not in metadata[category]["documents"]:
            metadata[category]["documents"][doc_id] = {"num_chunks": 0, "metadata": chunk.metadata}
        metadata[category]["documents"][doc_id]["num_chunks"] += 1

        # Track expert-level metadata
        expert_id = str(chunk.expert_id)
        if expert_id not in metadata[category]["experts"]:
            metadata[category]["experts"][expert_id] = {"num_chunks": 0, "documents": set()}
        metadata[category]["experts"][expert_id]["num_chunks"] += 1

        # Check if documents is a list and convert back to set if needed
        if isinstance(metadata[category]["experts"][expert_id]["documents"], list):
            metadata[category]["experts"][expert_id]["documents"] = set(
                metadata[category]["experts"][expert_id]["documents"]
            )

        metadata[category]["experts"][expert_id]["documents"].add(doc_id)

        # Update total chunks count
        metadata[category]["num_chunks"] += 1

    # Move the conversion to list outside the main processing loop
    for category in metadata:
        for expert_data in metadata[category]["experts"].values():
            expert_data["documents"] = list(expert_data["documents"])

    return metadata


def _add_embeddings_metadata(embedded_chunks: list[EmbeddedPaperChunk], metadata: dict) -> dict:
    """Add metadata from embedded chunks including embedding information."""
    for chunk in embedded_chunks:
        category = chunk.get_category()
        if category not in metadata:
            metadata[category] = {
                "documents": {},
                "experts": {},
                "num_embedded_chunks": 0,
                "embedding_stats": {
                    "dimension": len(chunk.embedding) if chunk.embedding else 0,
                },
            }

        # Track document-level metadata
        doc_id = str(chunk.document_id)
        if doc_id not in metadata[category]["documents"]:
            metadata[category]["documents"][doc_id] = {"num_embedded_chunks": 0, "metadata": chunk.metadata}
        metadata[category]["documents"][doc_id]["num_embedded_chunks"] += 1

        # Track expert-level metadata
        expert_id = str(chunk.expert_id)
        if expert_id not in metadata[category]["experts"]:
            metadata[category]["experts"][expert_id] = {"num_embedded_chunks": 0, "documents": set()}
        metadata[category]["experts"][expert_id]["num_embedded_chunks"] += 1

        # Check if documents is a list and convert back to set if needed
        if isinstance(metadata[category]["experts"][expert_id]["documents"], list):
            metadata[category]["experts"][expert_id]["documents"] = set(
                metadata[category]["experts"][expert_id]["documents"]
            )

        metadata[category]["experts"][expert_id]["documents"].add(doc_id)

        # Update total embedded chunks count
        metadata[category]["num_embedded_chunks"] += 1

    # Convert sets to lists for JSON serialization
    for category in metadata:
        for expert_data in metadata[category]["experts"].values():
            if isinstance(expert_data["documents"], set):
                expert_data["documents"] = list(expert_data["documents"])

    return metadata
