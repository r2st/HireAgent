"""v1 API router aggregation."""

from fastapi import APIRouter

from app.api.v1 import applications, auth, candidates, jobs

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(jobs.router)
api_router.include_router(candidates.router)
api_router.include_router(applications.router)
# Job-scoped pipeline views (`/jobs/{id}/board`, `/jobs/{id}/candidates`) live
# with the pipeline code but hang off the jobs prefix, per design §6.1.
api_router.include_router(applications.job_router)
