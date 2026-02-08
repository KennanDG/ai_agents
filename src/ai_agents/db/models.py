from sqlalchemy import String, Integer, DateTime, ForeignKey, UniqueConstraint, func, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class RagSource(Base):
    __tablename__ = "rag_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_uri: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)

    collection_name: Mapped[str] = mapped_column(String, nullable=False)
    namespace: Mapped[str] = mapped_column(String, nullable=False)

    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    chunks = relationship("RagChunk", back_populates="source", cascade="all, delete-orphan")


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        UniqueConstraint("source_id", "chunk_index", name="uq_source_chunk_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("rag_sources.id", ondelete="CASCADE"), nullable=False)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String, nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    source = relationship("RagSource", back_populates="chunks")




class RagIngestJob(Base):
    __tablename__ = "rag_ingest_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)  # Celery task id

    status: Mapped[str] = mapped_column(String, nullable=False)  # QUEUED|STARTED|SUCCEEDED|FAILED
    namespace: Mapped[str] = mapped_column(String, nullable=False)
    collection_name: Mapped[str] = mapped_column(String, nullable=False)

    # store inputs for debugging/auditing
    paths_json: Mapped[str] = mapped_column(Text, nullable=False)

    # outputs
    ingested_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

