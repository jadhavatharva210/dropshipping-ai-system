"""ORM models.

Importing this package registers every table on ``Base.metadata``, which is
what Alembic's ``env.py`` and ``init_db()`` rely on - so new models must be
exported here.
"""

from __future__ import annotations

from app.models.core import ApprovalRequest, AuditLog, SharedMemory, Task

__all__ = ["ApprovalRequest", "AuditLog", "SharedMemory", "Task"]
