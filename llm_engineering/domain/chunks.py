from pydantic import UUID4, Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class PaperChunk(VectorBaseDocument):
    content: str
    document_id: UUID4
    expert_id: UUID4
    metadata: dict = Field(default_factory=dict)

    class Config:
        category = DataCategory.PAPERS
