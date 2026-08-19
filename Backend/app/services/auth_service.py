"""Application-layer auth logic. Routes call this; this never touches HTTP concerns."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.repositories.user_repository import RoleRepository, UserRepository


class AuthError(Exception):
    """Raised for any auth failure (duplicate email, bad credentials, inactive user)."""


@dataclass(slots=True)
class AuthResult:
    user: User
    access_token: str
    expires_in_minutes: int


class AuthService:
    def __init__(self, user_repo: UserRepository, role_repo: RoleRepository, settings: Settings) -> None:
        self._user_repo = user_repo
        self._role_repo = role_repo
        self._settings = settings

    async def register(self, email: str, full_name: str, password: str) -> User:
        existing = await self._user_repo.get_by_email(email)
        if existing:
            raise AuthError("An account with this email already exists.")

        default_role = await self._role_repo.get_by_name("user")
        if default_role is None:
            raise AuthError("Default 'user' role is not seeded. Run database seed script.")

        user = User(
            email=email,
            full_name=full_name,
            hashed_password=hash_password(password),
            role_id=default_role.id,
        )
        user = await self._user_repo.add(user)
        await self._user_repo.commit()
        user.role = default_role
        return user

    async def login(self, email: str, password: str) -> AuthResult:
        user = await self._user_repo.get_by_email(email)
        if user is None or not verify_password(password, user.hashed_password):
            raise AuthError("Invalid email or password.")
        if not user.is_active:
            raise AuthError("This account has been deactivated.")

        token = create_access_token(
            subject=str(user.id),
            role=user.role.name,
            secret_key=self._settings.JWT_SECRET_KEY,
            algorithm=self._settings.JWT_ALGORITHM,
            expires_minutes=self._settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        )
        return AuthResult(user=user, access_token=token, expires_in_minutes=self._settings.ACCESS_TOKEN_EXPIRE_MINUTES)
