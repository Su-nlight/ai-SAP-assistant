"""Role-based access control. FastAPI dependency factory `require_role` gates routes by role."""
from __future__ import annotations

from enum import Enum

from fastapi import Depends, HTTPException, status

from app.api.deps import CurrentUser, get_current_user


class Role(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    USER = "user"
    VIEWER = "viewer"


# Simple hierarchy: admin can do anything a lower role can. Kept explicit
# (no clever bit-flag hierarchy) so it stays readable to a new engineer.
_ROLE_RANK = {Role.VIEWER: 0, Role.USER: 1, Role.ANALYST: 2, Role.ADMIN: 3}


def require_role(minimum_role: Role):
    def _dependency(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        try:
            user_role = Role(current_user.role)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown role.")
        if _ROLE_RANK[user_role] < _ROLE_RANK[minimum_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role '{minimum_role.value}' or higher.",
            )
        return current_user

    return _dependency
