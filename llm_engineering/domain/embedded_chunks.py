from pydantic import UUID4, Field

from llm_engineering.domain.types import DataCategory

from .base import VectorBaseDocument


class EmbeddedPaperChunk(VectorBaseDocument):
    content: str
    embedding: list[float] | None
    document_id: UUID4
    expert_id: UUID4
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedPaperChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Expert: {chunk.expert_id}
            Content: {chunk.content}\n
            """

        return context

    class Config:
        name = "embedded_papers"
        category = DataCategory.PAPERS
        use_vector_index = True
