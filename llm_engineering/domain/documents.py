import datetime

from mongoengine import (DateTimeField, DictField, Document, ReferenceField,
                         StringField)

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
    domain = StringField(required=True)  # e.g. "machine_learning", "deep_learning", "nlp"
    created_at = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))
    updated_at = DateTimeField(default=datetime.datetime.now(datetime.timezone.utc))

    meta = {
        'collection': 'experts',  # MongoDB collection name
        'indexes': ['domain'],    # Index for faster lookups
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


class PaperDocument(Document):
    content = StringField(required=True)  # Using DictField for content
    title = StringField(required=True)
    expert_id = ReferenceField('ExpertDocument', required=True)  # Better to use ReferenceField
    link = StringField(required=True)
    published_at = DateTimeField(required=True)

    meta = {
        'collection': 'papers',
        'indexes': [
            'expert_id',  # For quick lookups of papers by expert
        ]
    }

    class Settings:
        name = DataCategory.PAPERS

