"""Root conftest — session-scoped testcontainer fixtures.

Neo4j Community Edition and Redis 7 containers, shared across the entire
test session.  Individual tests clean each database via autouse fixtures.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import time

import pytest
import redis as sync_redis
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase
from redis.asyncio import Redis
from testcontainers.core.container import DockerContainer

logger = logging.getLogger(__name__)

# Load repository-root .env for test opt-ins (existing env vars stay authoritative).
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach suite markers from test path.

    - `tests/unit/*` -> `unit`
    - `tests/integration/*` -> `integration`
    - `tests/scenarios/*` -> `scenario`
    """
    root = Path(__file__).resolve().parents[1]
    for item in items:
        item_path = Path(str(item.fspath)).resolve()
        try:
            rel = item_path.relative_to(root)
        except ValueError:
            continue

        parts = rel.parts
        if len(parts) < 2 or parts[0] != "tests":
            continue
        if parts[1] == "unit":
            item.add_marker(pytest.mark.unit)
        elif parts[1] == "integration":
            item.add_marker(pytest.mark.integration)
        elif parts[1] == "scenarios":
            item.add_marker(pytest.mark.scenario)


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def neo4j_container():
    """Spin up a Neo4j Community container and yield its bolt URI.

    Session-scoped: one container for the entire test run.
    """
    container = (
        DockerContainer("neo4j:community")
        .with_exposed_ports(7687)
        .with_env("NEO4J_AUTH", "none")
    )
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(7687)
        uri = f"bolt://{host}:{port}"

        async def wait_for_neo4j():
            driver = AsyncGraphDatabase.driver(uri)
            max_attempts = 30
            for attempt in range(max_attempts):
                try:
                    await driver.verify_connectivity()
                    await driver.close()
                    return
                except Exception as exc:
                    if attempt == max_attempts - 1:
                        await driver.close()
                        raise
                    logger.debug(
                        "Neo4j not ready (attempt %d/%d): %s",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    time.sleep(1)

        asyncio.run(wait_for_neo4j())
        yield uri


@pytest.fixture()
async def neo4j_driver(neo4j_container):
    """Yield an async Neo4j driver connected to the test container."""
    driver = AsyncGraphDatabase.driver(neo4j_container)
    yield driver
    await driver.close()


@pytest.fixture(autouse=True)
async def clean_neo4j(neo4j_driver):
    """Wipe all nodes and relationships before each test."""
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield


@pytest.fixture(scope="session")
def _graph_schema_initialized(neo4j_container):
    """Initialize the Neo4j schema once per session (indexes + constraints)."""
    from engramcp.graph.schema import init_schema

    async def _init():
        driver = AsyncGraphDatabase.driver(neo4j_container)
        await init_schema(driver)
        await driver.close()

    asyncio.run(_init())
    return True


@pytest.fixture()
async def graph_store(neo4j_driver, _graph_schema_initialized):
    """Yield a GraphStore connected to the test Neo4j container."""
    from engramcp.graph.store import GraphStore

    return GraphStore(neo4j_driver)


@pytest.fixture()
async def traceability(neo4j_driver, _graph_schema_initialized):
    """Yield a SourceTraceability connected to the test Neo4j container."""
    from engramcp.graph.traceability import SourceTraceability

    return SourceTraceability(neo4j_driver)


@pytest.fixture()
async def confidence_engine(graph_store, traceability):
    """Yield a ConfidenceEngine wired to graph_store and traceability."""
    from engramcp.engine.confidence import ConfidenceEngine

    return ConfidenceEngine(graph_store, traceability)


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_container():
    """Spin up a Redis 7 container and yield its URL.

    Session-scoped: one container for the entire test run.
    """
    container = DockerContainer("redis:7-alpine").with_exposed_ports(6379)
    with container as c:
        host = c.get_container_host_ip()
        port = c.get_exposed_port(6379)
        url = f"redis://{host}:{port}"

        # Wait for Redis readiness
        r = sync_redis.Redis(host=host, port=int(port))
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                r.ping()
                r.close()
                break
            except Exception as exc:
                if attempt == max_attempts - 1:
                    r.close()
                    raise
                logger.debug(
                    "Redis not ready (attempt %d/%d): %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                time.sleep(1)

        yield url


@pytest.fixture()
async def redis_client(redis_container):
    """Yield an async Redis client connected to the test container."""
    client = Redis.from_url(redis_container)
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
async def clean_redis(redis_client):
    """Flush Redis between tests."""
    await redis_client.flushdb()
    yield
    await redis_client.flushdb()
