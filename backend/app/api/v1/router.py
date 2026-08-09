"""v1 API router aggregation."""

from fastapi import APIRouter

from app.api.v1 import (
    applications,
    assessments,
    auth,
    candidates,
    interviews,
    jobs,
    outreach,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(jobs.router)
api_router.include_router(candidates.router)
api_router.include_router(applications.router)
api_router.include_router(interviews.router)
api_router.include_router(interviews.calendar_router)
api_router.include_router(assessments.router)
# Candidate self-service booking (design §4.3). Unauthenticated by design —
# the booking token in the path is the credential.
api_router.include_router(interviews.public_router)
# Candidate sitting an assessment (design §4.4). Token-as-credential again.
api_router.include_router(assessments.public_router)
# Candidate opt-out. Also unauthenticated, and also token-as-credential, but
# the caller is usually a mail client acting on RFC 8058 one-click rather than
# a person with a browser.
api_router.include_router(outreach.public_router)
# Job-scoped pipeline views (`/jobs/{id}/board`, `/jobs/{id}/candidates`) live
# with the pipeline code but hang off the jobs prefix, per design §6.1.
api_router.include_router(applications.job_router)
