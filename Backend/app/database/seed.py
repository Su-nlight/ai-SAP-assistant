"""
One-off seed script: creates default roles (admin/analyst/user/viewer) and a
demo admin account. Run with `python -m app.database.seed` after migrations.
Idempotent — safe to run multiple times.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select

from app.core.security import hash_password
from app.database.session import AsyncSessionLocal
from app.models.user import Role, User

_DEFAULT_ROLES = ["admin", "analyst", "user", "viewer"]


async def seed() -> None:
    async with AsyncSessionLocal() as session:
        role_by_name: dict[str, Role] = {}
        for name in _DEFAULT_ROLES:
            existing = (await session.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
            if existing is None:
                existing = Role(name=name, description=f"{name.capitalize()} role")
                session.add(existing)
                await session.flush()
            role_by_name[name] = existing

        admin_email = "admin@ai-sap-assistant.local"
        existing_admin = (await session.execute(select(User).where(User.email == admin_email))).scalar_one_or_none()
        if existing_admin is None:
            session.add(
                User(
                    email=admin_email,
                    full_name="Demo Administrator",
                    hashed_password=hash_password("ChangeMe123!"),
                    role_id=role_by_name["admin"].id,
                )
            )
            print(f"Seeded demo admin: {admin_email} / ChangeMe123!  (CHANGE THIS PASSWORD)")
        else:
            print("Demo admin already exists — skipping.")

        await session.commit()
        print("Roles seeded:", ", ".join(_DEFAULT_ROLES))


if __name__ == "__main__":
    asyncio.run(seed())
