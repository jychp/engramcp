"""Root conftest — session-scoped Neo4j testcontainers fixture.

Neo4j Community Edition container, shared across the entire test session.
Individual tests clean the database via the autouse ``clean_neo4j`` fixture.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from neo4j import AsyncGraphDatabase
from testcontainers.core.container import DockerContainer


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
                except Exception:
                    if attempt == max_attempts - 1:
                        await driver.close()
                        raise
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
