"""SQLAlchemy models.

Every model must be imported here so ``Base.metadata`` is fully populated
before alembic autogenerate or ``create_all`` runs.
"""

from app.db.base import Base
from app.models.analytics import AnalyticsSnapshot, JobBoardPosting
from app.models.application import Application, Screening, StageEvent
from app.models.assessment import Assessment, AssessmentTemplate
from app.models.candidate import Candidate, CandidateConsent, Resume
from app.models.interview import CalendarAccount, Interview, InterviewParticipant
from app.models.job import Job
from app.models.offer import OfferLetter, OfferTemplate
from app.models.organization import Organization, User
from app.models.outreach import (
    EmailAccount,
    MessageTemplate,
    OutreachMessage,
    OutreachSequence,
    SequenceEnrollment,
    SequenceStep,
)

__all__ = [
    "AnalyticsSnapshot",
    "Application",
    "Assessment",
    "AssessmentTemplate",
    "Base",
    "CalendarAccount",
    "Candidate",
    "CandidateConsent",
    "EmailAccount",
    "Interview",
    "InterviewParticipant",
    "Job",
    "JobBoardPosting",
    "MessageTemplate",
    "OfferLetter",
    "OfferTemplate",
    "Organization",
    "OutreachMessage",
    "OutreachSequence",
    "Resume",
    "Screening",
    "SequenceEnrollment",
    "SequenceStep",
    "StageEvent",
    "User",
]
