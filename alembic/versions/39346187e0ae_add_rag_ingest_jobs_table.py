"""add rag_ingest_jobs table

Revision ID: 39346187e0ae
Revises: 481427cd9d63
Create Date: 2026-02-08 22:18:03.108438

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '39346187e0ae'
down_revision: Union[str, Sequence[str], None] = '481427cd9d63'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "rag_ingest_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),

        sa.Column("job_id", sa.String(), nullable=False, unique=True),

        sa.Column("status", sa.String(), nullable=False),

        sa.Column("namespace", sa.String(), nullable=False),
        
        sa.Column("collection_name", sa.String(), nullable=False),

        sa.Column("paths_json", sa.Text(), nullable=False),

        sa.Column(
            "ingested_chunks",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),

        sa.Column("error", sa.Text(), nullable=True),

        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    
    op.create_index(
        "idx_rag_ingest_jobs_job_id",
        "rag_ingest_jobs",
        ["job_id"],
        unique=True,
    )

    pass


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index("ix_rag_ingest_jobs_job_id", table_name="rag_ingest_jobs")
    op.drop_table("rag_ingest_jobs")

    pass
