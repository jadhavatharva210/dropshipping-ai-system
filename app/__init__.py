"""Dropshipping AI System - modular monolith application package.

Layering (each layer may only import from the ones above it)::

    app.core    enums + settings, no framework dependencies
    app.db      declarative base, engine, session dependency
    app.models  SQLAlchemy 2.0 ORM models
    app.main    FastAPI composition root
"""

from __future__ import annotations

from app.core.config import APP_VERSION as __version__

__all__ = ["__version__"]
