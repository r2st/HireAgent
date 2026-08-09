"""offer signature name

Adds the column that records who a candidate said they were when they
accepted an offer with no e-signature provider configured. Self-serve
acceptance has no envelope from DocuSign or Digio to point back to, so the
typed name is the only record that a specific person confirmed — see
offer_service.accept.

Revision ID: f3a9c1d84e6b
Revises: c4e17a90bd52
Create Date: 2026-08-09 18:10:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'f3a9c1d84e6b'
down_revision: str | None = 'c4e17a90bd52'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'offer_letters',
        sa.Column('signed_by_name', sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('offer_letters', 'signed_by_name')
