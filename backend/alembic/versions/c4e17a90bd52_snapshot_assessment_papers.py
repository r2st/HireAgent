"""snapshot assessment papers

Adds the columns that let an issued assessment stand on its own: the questions
as they were sent, the pass mark that applied, and the time allowed. Reading
these back through ``template_id`` would mean a template edit silently rewrote
a test somebody had already sat.

Revision ID: c4e17a90bd52
Revises: 85ba1fb31caa
Create Date: 2026-08-09 12:40:11.104882

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'c4e17a90bd52'
down_revision: str | None = '85ba1fb31caa'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'assessments',
        sa.Column(
            'questions_json',
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'),
            nullable=False,
            # Existing rows predate the snapshot; an empty paper is the honest
            # value for them and the grader treats it as nothing to mark.
            server_default=sa.text("'[]'"),
        ),
    )
    op.add_column(
        'assessments',
        sa.Column(
            'passing_score',
            sa.Numeric(precision=5, scale=2),
            nullable=False,
            server_default=sa.text('60'),
        ),
    )
    op.add_column(
        'assessments',
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
    )

    # The defaults exist to backfill, not to become part of the contract: the
    # application always supplies all three.
    op.alter_column('assessments', 'questions_json', server_default=None)
    op.alter_column('assessments', 'passing_score', server_default=None)


def downgrade() -> None:
    op.drop_column('assessments', 'duration_minutes')
    op.drop_column('assessments', 'passing_score')
    op.drop_column('assessments', 'questions_json')
