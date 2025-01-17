import datetime
from typing import ClassVar
from uuid import uuid4

from mongoengine import DateTimeField, Document, ReferenceField, StringField, UUIDField

from .types import DataCategory


class ExpertDocument(Document):
    """
    MongoDB document for ML domain experts from arxiv papers.
    Using mongoengine ODM which provides:
    1. Schema validation via field types
    2. Query builder interface similar to Django ORM
    3. Document lifecycle hooks (pre_save, post_save etc.)
    4. Automatic ID generation and handling
    """

    id = UUIDField(primary_key=True, default=uuid4)
    domain = StringField(required=True)  # e.g. "machine_learning", "deep_learning", "nlp"
    created_at = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updated_at = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

    meta: ClassVar[dict] = {
        "collection": "experts",
        "indexes": ["domain"],
    }

    @classmethod
    def get_or_create(cls, domain: str) -> "ExpertDocument":
        """
        Get an existing expert by domain or create a new one.
        Similar interface to Django's get_or_create for familiarity.
        """
        expert = cls.objects(domain=domain).first()
        if not expert:
            expert = cls(domain=domain).save()
        return expert

    @classmethod
    def get_collection_name(cls) -> str:
        """Get the MongoDB collection name for this document."""
        return cls._meta["collection"]


class PaperDocument(Document):
    id = UUIDField(primary_key=True, default=uuid4)
    content = StringField(required=True)  # Using DictField for content
    title = StringField(required=True)
    expert_id = ReferenceField("ExpertDocument", required=True)  # Changed from ReferenceField
    link = StringField(required=True)
    published_at = DateTimeField(required=True)
    created_at = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

    meta: ClassVar[dict] = {
        "collection": "papers",
        "indexes": [
            "expert_id",  # For quick lookups of papers by expert
            "link",
        ],
    }

    @classmethod
    def get_collection_name(cls) -> str:
        """Get the MongoDB collection name for this document."""
        return cls._meta["collection"]

    @classmethod
    def bulk_find(cls, query: dict) -> list["PaperDocument"]:
        """
        Fetch multiple documents based on query parameters.
        Returns a list of documents instead of a QuerySet for immediate evaluation.
        """
        return list(cls.objects(**query))

    class Settings:
        name = DataCategory.PAPERS
