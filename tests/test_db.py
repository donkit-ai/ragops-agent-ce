"""Unit tests for database layer (KV store and migrations)."""

from __future__ import annotations

import concurrent.futures
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from donkit_ragops.config import Settings
from donkit_ragops.db import DB
from donkit_ragops.db import KV
from donkit_ragops.db import close
from donkit_ragops.db import kv_all
from donkit_ragops.db import kv_all_by_prefix
from donkit_ragops.db import kv_delete
from donkit_ragops.db import kv_get
from donkit_ragops.db import kv_set
from donkit_ragops.db import migrate
from sqlmodel import Session
from sqlmodel import select

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path
        # Cleanup
        if db_path.exists():
            db_path.unlink()


@pytest.fixture
def db(temp_db_path: Path) -> Generator[DB, None, None]:
    """Create an in-memory database for testing."""
    database = DB(path=temp_db_path)
    migrate(database)
    yield database
    close(database)


@pytest.fixture
def settings(temp_db_path: Path) -> Settings:
    """Create settings with temporary database path."""
    return Settings(db_path=str(temp_db_path))


# ============================================================================
# Tests: KV Operations
# ============================================================================


def test_kv_set_and_get(db: DB) -> None:
    """Test setting and getting a key-value pair."""
    kv_set(db, "test_key", "test_value")
    result = kv_get(db, "test_key")

    assert result == "test_value"


def test_kv_get_nonexistent(db: DB) -> None:
    """Test getting a non-existent key returns None."""
    result = kv_get(db, "nonexistent_key")

    assert result is None


def test_kv_set_overwrites_existing(db: DB) -> None:
    """Test that setting an existing key overwrites the value."""
    kv_set(db, "key", "value1")
    kv_set(db, "key", "value2")
    result = kv_get(db, "key")

    assert result == "value2"


def test_kv_delete_existing(db: DB) -> None:
    """Test deleting an existing key-value pair."""
    kv_set(db, "key_to_delete", "value")
    deleted = kv_delete(db, "key_to_delete")
    result = kv_get(db, "key_to_delete")

    assert deleted is True
    assert result is None


def test_kv_delete_nonexistent(db: DB) -> None:
    """Test deleting a non-existent key returns False."""
    deleted = kv_delete(db, "nonexistent_key")

    assert deleted is False


def test_kv_all_empty(db: DB) -> None:
    """Test kv_all returns empty list when no keys exist."""
    result = list(kv_all(db))

    assert result == []


def test_kv_all_multiple_keys(db: DB) -> None:
    """Test kv_all returns all key-value pairs."""
    kv_set(db, "key1", "value1")
    kv_set(db, "key2", "value2")
    kv_set(db, "key3", "value3")

    result = list(kv_all(db))

    assert len(result) == 3
    assert ("key1", "value1") in result
    assert ("key2", "value2") in result
    assert ("key3", "value3") in result


def test_kv_all_ordered_by_key(db: DB) -> None:
    """Test kv_all returns results ordered by key."""
    kv_set(db, "zebra", "z_value")
    kv_set(db, "apple", "a_value")
    kv_set(db, "banana", "b_value")

    result = list(kv_all(db))

    assert result[0][0] == "apple"
    assert result[1][0] == "banana"
    assert result[2][0] == "zebra"


def test_kv_all_by_prefix_empty(db: DB) -> None:
    """Test kv_all_by_prefix returns empty list when no matches."""
    kv_set(db, "other_key", "value")
    result = kv_all_by_prefix(db, "prefix_")

    assert result == []


def test_kv_all_by_prefix_single_match(db: DB) -> None:
    """Test kv_all_by_prefix returns matching keys."""
    kv_set(db, "prefix_key1", "value1")
    kv_set(db, "prefix_key2", "value2")
    kv_set(db, "other_key", "value3")

    result = kv_all_by_prefix(db, "prefix_")

    assert len(result) == 2
    assert ("prefix_key1", "value1") in result
    assert ("prefix_key2", "value2") in result


def test_kv_all_by_prefix_no_false_matches(db: DB) -> None:
    """Test kv_all_by_prefix doesn't match keys that don't start with prefix."""
    kv_set(db, "prefix_key", "value1")
    kv_set(db, "my_prefix_key", "value2")  # Contains but doesn't start with
    kv_set(db, "other", "value3")

    result = kv_all_by_prefix(db, "prefix_")

    assert len(result) == 1
    assert result[0][0] == "prefix_key"


def test_kv_all_by_prefix_empty_prefix(db: DB) -> None:
    """Test kv_all_by_prefix with empty prefix returns all keys."""
    kv_set(db, "key1", "value1")
    kv_set(db, "key2", "value2")

    result = kv_all_by_prefix(db, "")

    assert len(result) == 2


def test_kv_operations_with_special_characters(db: DB) -> None:
    """Test KV operations with special characters in values."""
    special_value = "test\"value'with\\special\nchars"
    kv_set(db, "special_key", special_value)
    result = kv_get(db, "special_key")

    assert result == special_value


def test_kv_operations_with_empty_string(db: DB) -> None:
    """Test KV operations with empty string values."""
    kv_set(db, "empty_key", "")
    result = kv_get(db, "empty_key")

    assert result == ""


def test_kv_operations_with_large_value(db: DB) -> None:
    """Test KV operations with large string values."""
    large_value = "x" * 10000
    kv_set(db, "large_key", large_value)
    result = kv_get(db, "large_key")

    assert result == large_value
    assert len(result) == 10000


# ============================================================================
# Tests: Migrations
# ============================================================================


def test_migrate_creates_tables(temp_db_path: Path) -> None:
    """Test that migrate creates all required tables."""
    db = DB(path=temp_db_path)
    migrate(db)

    # Check that tables exist by trying to query them
    with Session(db._engine) as session:
        # Try to query KV table
        result = session.exec(select(KV)).all()
        assert isinstance(result, list)


def test_migrate_idempotent(db: DB) -> None:
    """Test that calling migrate multiple times is safe."""
    # First migration already done in fixture
    migrate(db)  # Call again
    migrate(db)  # And again

    # Should still be able to use the database
    kv_set(db, "key", "value")
    result = kv_get(db, "key")

    assert result == "value"


def test_migrate_preserves_data(temp_db_path: Path) -> None:
    """Test that migrate doesn't delete existing data."""
    db = DB(path=temp_db_path)
    migrate(db)

    # Add data
    kv_set(db, "key1", "value1")
    kv_set(db, "key2", "value2")

    # Call migrate again
    migrate(db)

    # Data should still exist
    assert kv_get(db, "key1") == "value1"
    assert kv_get(db, "key2") == "value2"


# ============================================================================
# Tests: Concurrent Access
# ============================================================================


def test_concurrent_kv_set_get(db: DB) -> None:
    """Test concurrent set and get operations."""
    num_threads = 5
    num_operations = 20

    def worker(thread_id: int) -> None:
        for i in range(num_operations):
            key = f"thread_{thread_id}_key_{i}"
            value = f"value_{i}"
            kv_set(db, key, value)
            result = kv_get(db, key)
            assert result == value

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Raise any exceptions


def test_concurrent_kv_operations_mixed(db: DB) -> None:
    """Test concurrent mixed KV operations (set, get, delete)."""
    num_threads = 3

    def worker(thread_id: int) -> None:
        for i in range(10):
            key = f"key_{thread_id}_{i}"
            # Set
            kv_set(db, key, f"value_{i}")
            # Get
            result = kv_get(db, key)
            assert result == f"value_{i}"
            # Delete
            deleted = kv_delete(db, key)
            assert deleted is True
            # Verify deleted
            result = kv_get(db, key)
            assert result is None

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def test_concurrent_kv_all(db: DB) -> None:
    """Test concurrent kv_all with concurrent modifications."""
    # Pre-populate database
    for i in range(10):
        kv_set(db, f"initial_key_{i}", f"value_{i}")

    def reader() -> None:
        for _ in range(5):
            result = list(kv_all(db))
            assert len(result) >= 10

    def writer(thread_id: int) -> None:
        for i in range(5):
            kv_set(db, f"thread_{thread_id}_key_{i}", f"value_{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit 2 readers and 2 writers
        futures = []
        futures.append(executor.submit(reader))
        futures.append(executor.submit(reader))
        futures.append(executor.submit(writer, 1))
        futures.append(executor.submit(writer, 2))

        for future in concurrent.futures.as_completed(futures):
            future.result()


def test_concurrent_kv_all_by_prefix(db: DB) -> None:
    """Test concurrent kv_all_by_prefix with concurrent modifications."""
    # Pre-populate
    for i in range(5):
        kv_set(db, f"prefix_key_{i}", f"value_{i}")

    def reader() -> None:
        for _ in range(5):
            result = kv_all_by_prefix(db, "prefix_")
            assert len(result) >= 5

    def writer() -> None:
        for i in range(5):
            kv_set(db, f"prefix_new_{i}", f"value_{i}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        futures.append(executor.submit(reader))
        futures.append(executor.submit(reader))
        futures.append(executor.submit(writer))

        for future in concurrent.futures.as_completed(futures):
            future.result()


def test_concurrent_delete_same_key(db: DB) -> None:
    """Test concurrent delete operations on the same key.

    Note: With SQLite and check_same_thread=False, each thread gets its own
    session, so multiple threads may see the row as existing and attempt to
    delete it. The important thing is that the final state is consistent.
    """
    kv_set(db, "shared_key", "value")

    results = []

    def deleter() -> None:
        deleted = kv_delete(db, "shared_key")
        results.append(deleted)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(deleter) for _ in range(3)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    # After all deletes, the key should not exist
    final_value = kv_get(db, "shared_key")
    assert final_value is None
    # At least one delete should have succeeded
    assert sum(results) >= 1


# ============================================================================
# Tests: Edge Cases
# ============================================================================


def test_db_context_manager(temp_db_path: Path) -> None:
    """Test DB context manager properly opens and closes session."""
    db = DB(path=temp_db_path)
    migrate(db)

    with db as session_db:
        assert session_db._session is not None
        kv_set(session_db, "key", "value")

    # Session should be closed after context
    assert db._session is None


def test_kv_set_updates_timestamp(db: DB) -> None:
    """Test that kv_set updates the timestamp."""

    kv_set(db, "key", "value1")

    # Get the first timestamp
    with Session(db._engine) as session:
        obj1 = session.get(KV, "key")
        timestamp1 = obj1.updated_at if obj1 else None

    # Wait a bit and update
    import time

    time.sleep(0.01)
    kv_set(db, "key", "value2")

    # Get the second timestamp
    with Session(db._engine) as session:
        obj2 = session.get(KV, "key")
        timestamp2 = obj2.updated_at if obj2 else None

    # Timestamps should be different
    assert timestamp1 is not None
    assert timestamp2 is not None
    assert timestamp2 > timestamp1


def test_multiple_db_instances(temp_db_path: Path) -> None:
    """Test that multiple DB instances can access the same database."""
    db1 = DB(path=temp_db_path)
    db2 = DB(path=temp_db_path)

    migrate(db1)
    migrate(db2)

    # Set via db1
    kv_set(db1, "key", "value")

    # Get via db2
    result = kv_get(db2, "key")

    assert result == "value"

    close(db1)
    close(db2)
