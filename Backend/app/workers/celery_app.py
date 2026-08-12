"""
Celery application.

Redis (already required for caching/rate-limiting infra) doubles as the
broker and result backend — this avoids standing up RabbitMQ purely for a
single background task type. The spec lists RabbitMQ as optional for
exactly this reason; swapping the broker to RabbitMQ later is a one-line
`CELERY_BROKER_URL` change, nothing in `tasks.py` needs to know or care.
"""
from __future__ import annotations

from celery import Celery

from app.core.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "ai_sap_assistant",
    broker=_settings.CELERY_BROKER_URL or _settings.REDIS_URL,
    backend=_settings.CELERY_RESULT_BACKEND or _settings.REDIS_URL,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=60 * 10,  # a stuck document-processing task shouldn't hang a worker forever
    worker_max_tasks_per_child=100,  # bounds memory growth from repeatedly loading the embedding model
)
