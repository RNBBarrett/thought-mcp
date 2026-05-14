"""Tiny fixture module — known-good structure for AST extractor tests.

Intentionally compact. The extractor should produce:
  module:   auth
  function: authenticate_user
  function: _decode_token         (private — leading underscore)
  class:    AuthError
  method:   AuthError.__init__
  method:   AuthError.with_context
  class:    JWTAuth        (inherits from AuthBackend)
  method:   JWTAuth.verify

Imports:
  import jwt
  from datetime import datetime
  from .errors import UpstreamError

Calls (Phase 2 will pick these up):
  authenticate_user -> _decode_token
  authenticate_user -> JWTAuth.verify
  authenticate_user -> AuthError
  JWTAuth.verify    -> jwt.decode

Inheritance:
  JWTAuth INHERITS_FROM AuthBackend
"""
from __future__ import annotations

from datetime import datetime

import jwt

from .errors import UpstreamError  # noqa: F401 — fixture exercises relative-import extraction


class AuthBackend:
    """Base class for auth backends (defined in another file conceptually)."""


class AuthError(Exception):
    """Raised when authentication fails."""

    def __init__(self, message: str, status: int = 401) -> None:
        super().__init__(message)
        self.status = status

    def with_context(self, ctx: dict) -> AuthError:
        self.context = ctx
        return self


class JWTAuth(AuthBackend):
    """JWT-based auth backend."""

    def __init__(self, secret: str) -> None:
        self.secret = secret

    def verify(self, token: str) -> dict:
        return jwt.decode(token, self.secret, algorithms=["HS256"])


def _decode_token(token: str, secret: str) -> dict:
    """Internal helper — leading underscore signals private."""
    return jwt.decode(token, secret, algorithms=["HS256"])


def authenticate_user(token: str, secret: str) -> dict:
    """Verify a JWT and return the user's claims.

    Public entry point. Combines _decode_token and a JWTAuth.verify path so
    Phase 2's call-graph extraction has a real branchy target.
    """
    try:
        claims = _decode_token(token, secret)
        backend = JWTAuth(secret)
        backend.verify(token)
        return claims
    except Exception as e:
        raise AuthError(f"auth failed at {datetime.now()}: {e}") from e
