import hashlib
from uuid import UUID

from bson import ObjectId
from loguru import logger

from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.chunks import PaperChunk
from llm_engineering.domain.cleaned_documents import CleanedPaperDocument
from llm_engineering.domain.documents import PaperDocument
from llm_engineering.domain.embedded_chunks import EmbeddedPaperChunk
from llm_engineering.domain.queries import EmbeddedQuery, Query
from llm_engineering.domain.types import DataCategory

from .operations import chunk_text, clean_text


def objectid_to_uuid4(obj_id: ObjectId) -> UUID:
    return UUID(str(obj_id).replace('-',''), version=4)

def uuid4_to_objectid(uuid: UUID) -> ObjectId:
    return ObjectId(str(uuid).replace('-',''))

class DocumentCleaner:
    @classmethod
    def clean(cls, doc: PaperDocument) -> CleanedPaperDocument:        
        logger.info(f"Cleaning document: {doc.id}")
        logger.info(f"Expert ID: {doc.expert_id}")
        logger.info(f"Expert ID type: {type(doc.expert_id)}")
        cleaned_doc = CleanedPaperDocument(
            id=doc.id,
            content=clean_text(" #### ".join(doc.content)),
            expert_id=doc.expert_id.id,
        )
        logger.info(
            "Document cleaned successfully.",
            cleaned_content_len=len(cleaned_doc.content),
        )
        return cleaned_doc


class DocumentChunker:
    CHUNK_SIZE = 250
    CHUNK_OVERLAP = 25

    @classmethod
    def chunk(cls, doc: CleanedPaperDocument) -> list[PaperChunk]:
        chunks = chunk_text(
            doc.content,
            chunk_size=cls.CHUNK_SIZE,
            chunk_overlap=cls.CHUNK_OVERLAP,
        )

        chunk_models = []
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            chunk_model = PaperChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                document_id=doc.id,
                expert_id=doc.expert_id,
                metadata={
                    "chunk_size": cls.CHUNK_SIZE,
                    "chunk_overlap": cls.CHUNK_OVERLAP,
                },
            )
            chunk_models.append(chunk_model)

        logger.info(
            "Document chunked successfully.",
            num=len(chunk_models),
        )
        return chunk_models


class DocumentEmbedder:
    _embedding_model = EmbeddingModelSingleton()

    @classmethod
    def embed(
        cls, doc: VectorBaseDocument | list[VectorBaseDocument]
    ) -> VectorBaseDocument | list[VectorBaseDocument]:
        is_list = isinstance(doc, list)
        if not is_list:
            doc = [doc]

        if len(doc) == 0:
            return []

        # Get embeddings for all documents in batch
        embedding_inputs = [d.content for d in doc]
        embeddings = cls._embedding_model(embedding_inputs, to_list=True)

        # Map embeddings to appropriate document types
        if doc[0].get_category() == DataCategory.QUERIES:
            embedded_models = [
                cls._create_embedded_query(doc, embedding)
                for doc, embedding in zip(doc, embeddings, strict=True)
            ]
        else:
            embedded_models = [
                cls._create_embedded_paper(doc, embedding)
                for doc, embedding in zip(doc, embeddings, strict=True)
            ]

        if not is_list:
            embedded_models = embedded_models[0]

        logger.info(
            "Documents embedded successfully.",
            num=len(doc) if is_list else 1,
        )

        return embedded_models

    @classmethod
    def _create_embedded_query(cls, doc: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=doc.id,
            expert_id=doc.expert_id,
            content=doc.content,
            embedding=embedding,
            metadata=cls._get_embedding_metadata(),
        )

    @classmethod
    def _create_embedded_paper(cls, doc: PaperChunk, embedding: list[float]) -> EmbeddedPaperChunk:
        return EmbeddedPaperChunk(
            id=doc.id,
            content=doc.content,
            embedding=embedding,
            document_id=doc.document_id,
            expert_id=doc.expert_id,
            metadata=cls._get_embedding_metadata(),
        )

    @classmethod
    def _get_embedding_metadata(cls) -> dict:
        return {
            "embedding_model_id": cls._embedding_model.model_id,
            "embedding_size": cls._embedding_model.embedding_size,
            "max_input_length": cls._embedding_model.max_input_length,
        }
