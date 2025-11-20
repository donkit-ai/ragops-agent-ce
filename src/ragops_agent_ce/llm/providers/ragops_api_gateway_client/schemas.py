from uuid import UUID

from pydantic import BaseModel


class ProjectInfo(BaseModel):
    id: UUID
    rag_use_case: str | None = None
