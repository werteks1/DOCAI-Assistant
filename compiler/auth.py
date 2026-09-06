"""FastAPI-зависимости авторизации поверх auth_store.

Роли проверяются на сервере: `current_user` (любой вошедший), `enforce_change`
(вошедший и сменивший обязательный стартовый пароль), `require_admin`.
Матрица доступа эндпоинтов — в api.py; здесь только проверки и HTTP-статусы.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Depends, Header, HTTPException

from . import auth_store

_401 = HTTPException(status_code=401, detail="Требуется вход (авторизация не пройдена)")
_403_CHANGE = HTTPException(
    status_code=403,
    detail="Сначала смените временный пароль (обязательная смена пароля при первом входе).",
)


def _bearer_token(authorization: str) -> Optional[str]:
    """Достаёт токен из «Authorization: Bearer <token>»."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def current_user(authorization: str = Header(default="")) -> Dict[str, Any]:
    """Возвращает пользователя по токену сессии или 401."""
    token = _bearer_token(authorization)
    user = auth_store.get_user_by_token(token) if token else None
    if user is None:
        raise _401
    return user


def enforce_change(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """Пускает только тех, кто уже сменил обязательный стартовый пароль."""
    if user.get("must_change"):
        raise _403_CHANGE
    return user


def require_admin(user: Dict[str, Any] = Depends(enforce_change)) -> Dict[str, Any]:
    """Только администратор (и после смены стартового пароля)."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Недостаточно прав: требуется администратор")
    return user
