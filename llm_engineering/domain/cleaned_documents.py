from abc import ABC
from typing import Optional

from pydantic import UUID4

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str


class CleanedPaperDocument(CleanedDocument):
    image: Optional[str] = None

    class Config:
        name = "cleaned_papers"
        category = DataCategory.PAPERS
        use_vector_index = False

