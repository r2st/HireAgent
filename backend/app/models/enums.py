"""Domain enumerations shared by models, schemas, and services."""

from __future__ import annotations

from enum import StrEnum


class OrganizationType(StrEnum):
    COMPANY = "company"
    AGENCY = "agency"
    STAFFING = "staffing"


class PlanTier(StrEnum):
    """Design §7.1 pricing tiers."""

    STARTUP = "startup"
    GROWTH = "growth"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


PLAN_LIMITS: dict[PlanTier, dict[str, int | None]] = {
    # ``None`` means unlimited.
    PlanTier.STARTUP: {"active_jobs": 5, "screening_credits": 100},
    PlanTier.GROWTH: {"active_jobs": 20, "screening_credits": 500},
    PlanTier.BUSINESS: {"active_jobs": None, "screening_credits": 2000},
    PlanTier.ENTERPRISE: {"active_jobs": None, "screening_credits": None},
}


class UserRole(StrEnum):
    """Design §8.3 role-based access control."""

    ADMIN = "admin"
    HIRING_MANAGER = "hiring_manager"
    RECRUITER = "recruiter"
    INTERVIEWER = "interviewer"


class JobStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"
    PAUSED = "paused"
    CLOSED = "closed"
    ARCHIVED = "archived"


class EmploymentType(StrEnum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"


class WorkMode(StrEnum):
    ONSITE = "onsite"
    HYBRID = "hybrid"
    REMOTE = "remote"


class SeniorityLevel(StrEnum):
    INTERN = "intern"
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"
    EXECUTIVE = "executive"


class PipelineStage(StrEnum):
    """Design §4.5 Kanban stages, in pipeline order."""

    SOURCED = "sourced"
    APPLIED = "applied"
    SCREENED = "screened"
    INTERVIEWED = "interviewed"
    ASSESSED = "assessed"
    OFFERED = "offered"
    HIRED = "hired"


# Ordered list used for progression checks and conversion analytics.
STAGE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.SOURCED,
    PipelineStage.APPLIED,
    PipelineStage.SCREENED,
    PipelineStage.INTERVIEWED,
    PipelineStage.ASSESSED,
    PipelineStage.OFFERED,
    PipelineStage.HIRED,
)

STAGE_INDEX: dict[PipelineStage, int] = {s: i for i, s in enumerate(STAGE_ORDER)}


class ApplicationStatus(StrEnum):
    ACTIVE = "active"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    ON_HOLD = "on_hold"
    HIRED = "hired"


class CandidateSource(StrEnum):
    DIRECT = "direct"
    CAREER_SITE = "career_site"
    NAUKRI = "naukri"
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    REFERRAL = "referral"
    AGENCY = "agency"
    SOURCED = "sourced"
    IMPORT = "import"


class ResumeParseStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PARSED = "parsed"
    FAILED = "failed"


class InterviewType(StrEnum):
    PHONE = "phone"
    VIDEO = "video"
    ONSITE = "onsite"
    TECHNICAL = "technical"
    PANEL = "panel"
    ASYNC_VIDEO = "async_video"


class InterviewStatus(StrEnum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"


class SequenceStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class OutreachChannel(StrEnum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    LINKEDIN = "linkedin"
    SMS = "sms"


class EnrollmentStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    REPLIED = "replied"
    BOUNCED = "bounced"
    UNSUBSCRIBED = "unsubscribed"
    FAILED = "failed"


class MessageStatus(StrEnum):
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    REPLIED = "replied"
    BOUNCED = "bounced"
    FAILED = "failed"


class EmailProvider(StrEnum):
    GMAIL = "gmail"
    OUTLOOK = "outlook"
    SMTP = "smtp"
    SES = "ses"


class WarmupStatus(StrEnum):
    NOT_STARTED = "not_started"
    WARMING = "warming"
    READY = "ready"
    PAUSED = "paused"
    BLOCKED = "blocked"


class AssessmentType(StrEnum):
    CODING = "coding"
    MCQ = "mcq"
    TAKE_HOME = "take_home"
    PSYCHOMETRIC = "psychometric"
    VIDEO_SCREENING = "video_screening"


class AssessmentStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    SKIPPED = "skipped"


class OfferStatus(StrEnum):
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    SENT = "sent"
    VIEWED = "viewed"
    SIGNED = "signed"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    WITHDRAWN = "withdrawn"


class ConsentType(StrEnum):
    """Design §8.2 consent categories."""

    RESUME_PROCESSING = "resume_processing"
    EMAIL_COMMUNICATION = "email_communication"
    WHATSAPP_COMMUNICATION = "whatsapp_communication"
    SMS_COMMUNICATION = "sms_communication"
    DATA_SHARING = "data_sharing"
    ASSESSMENT = "assessment"
    BACKGROUND_CHECK = "background_check"


class ConsentStatus(StrEnum):
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class JobBoard(StrEnum):
    NAUKRI = "naukri"
    INDEED = "indeed"
    LINKEDIN = "linkedin"


class PostingStatus(StrEnum):
    PENDING = "pending"
    POSTED = "posted"
    FAILED = "failed"
    EXPIRED = "expired"
    REMOVED = "removed"


class BiasCategory(StrEnum):
    """Design §4.9 bias taxonomy for job-description analysis."""

    GENDERED = "gendered"
    AGE = "age"
    RACIAL_CULTURAL = "racial_cultural"
    DISABILITY = "disability"
    NATIONALITY = "nationality"
    MARITAL_FAMILY = "marital_family"
    APPEARANCE = "appearance"
    EXCLUSIONARY_REQUIREMENT = "exclusionary_requirement"
    ELITIST = "elitist"
