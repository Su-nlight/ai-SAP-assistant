from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.audit_feedback import Feedback
from app.models.chat import ChatSession, Message
from app.repositories.base import BaseRepository


class ChatRepository(BaseRepository[ChatSession]):
    model = ChatSession

    async def get_session_with_messages(self, session_id: uuid.UUID, user_id: uuid.UUID) -> ChatSession | None:
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .options(selectinload(ChatSession.messages))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sessions_for_user(self, user_id: uuid.UUID) -> list[ChatSession]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .options(selectinload(ChatSession.messages))
            .order_by(ChatSession.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_recent_messages(self, session_id: uuid.UUID, limit: int = 10) -> list[Message]:
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.timestamp.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return list(reversed(result.scalars().all()))

    async def add_message(self, message: Message) -> Message:
        self._session.add(message)
        await self._session.flush()
        return message

    async def add_feedback(self, feedback: Feedback) -> Feedback:
        self._session.add(feedback)
        await self._session.flush()
        return feedback


def get_chat_repository(session: AsyncSession) -> ChatRepository:
    return ChatRepository(session)
