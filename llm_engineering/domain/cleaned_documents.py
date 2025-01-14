from pydantic import UUID4, Field

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedPaperDocument(VectorBaseDocument):
    content: str = Field(required=True)
    expert_id: UUID4 = Field(required=True)

    class Config:
        name = "cleaned_papers"
        category = DataCategory.PAPERS
        use_vector_index = False
        arbitrary_types_allowed = True
