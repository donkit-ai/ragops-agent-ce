from __future__ import annotations

import datetime
from collections.abc import Iterable
from pathlib import Path

from sqlmodel import Field, Session, SQLModel, create_engine, select

from .config import Settings, load_settings


class KV(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )


class MessageHistory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    sequence_num: int
    role: str
    content: str
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


class ConversationSummary(SQLModel, table=True):
    session_id: str = Field(primary_key=True)
    summary: str
    updated_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.datetime.now(datetime.timezone.utc)},
    )


class DB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._engine = create_engine(
            f"sqlite:///{self.path}", connect_args={"check_same_thread": False}
        )
        self._session: Session | None = None

    def __enter__(self) -> DB:
        self._session = Session(self._engine)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self._session is not None:
            self._session.close()
            self._session = None


def open_db(settings: Settings | None = None) -> DB:
    cfg = settings or load_settings()
    db_path = Path(cfg.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DB(path=db_path)
    migrate(db)
    return db


def migrate(db: DB) -> None:
    SQLModel.metadata.create_all(db._engine)


def kv_set(db: DB, key: str, value: str) -> None:
    with Session(db._engine) as session:
        stmt = select(KV).where(KV.key == key)
        db_kv = session.exec(stmt).first()
        if db_kv:
            db_kv.value = value
            db_kv.updated_at = datetime.datetime.now(datetime.timezone.utc)
        else:
            db_kv = KV(key=key, value=value)
        session.add(db_kv)
        session.commit()


def kv_get(db: DB, key: str) -> str | None:
    with Session(db._engine) as session:
        obj = session.get(KV, key)
        return None if obj is None else obj.value


def kv_delete(db: DB, key: str) -> bool:
    """Delete a key-value pair. Returns True if deleted, False if not found."""
    with Session(db._engine) as session:
        obj = session.get(KV, key)
        if obj is None:
            return False
        session.delete(obj)
        session.commit()
        return True


def kv_all(db: DB) -> Iterable[tuple[str, str]]:
    with Session(db._engine) as session:
        rows = session.exec(select(KV).order_by(KV.key)).all()
        for row in rows:
            yield row.key, row.value


def kv_all_by_prefix(db: DB, prefix: str) -> list[tuple[str, str]]:
    """Get all key-value pairs where the key starts with a given prefix."""
    with Session(db._engine) as session:
        stmt = select(KV).where(KV.key.startswith(prefix))
        results = session.exec(stmt).all()
        return [(r.key, r.value) for r in results]


def close(db: DB) -> None:
    try:
        # Dispose underlying connections
        db._engine.dispose()
    except Exception:
        pass


_db: DB | None = None
