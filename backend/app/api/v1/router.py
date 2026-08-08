"""v1 API router aggregation."""

from fastapi import APIRouter

from app.api.v1 import auth, candidates, jobs

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(jobs.router)
api_router.include_router(candidates.router)
