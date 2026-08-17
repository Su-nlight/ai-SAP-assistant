from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import Role, User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    async def get_by_email(self, email: str) -> User | None:
        stmt = select(User).where(User.email == email).options(selectinload(User.role))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()


class RoleRepository(BaseRepository[Role]):
    model = Role

    async def get_by_name(self, name: str) -> Role | None:
        result = await self._session.execute(select(Role).where(Role.name == name))
        return result.scalar_one_or_none()


def get_user_repository(session: AsyncSession) -> UserRepository:
    return UserRepository(session)


def get_role_repository(session: AsyncSession) -> RoleRepository:
    return RoleRepository(session)
