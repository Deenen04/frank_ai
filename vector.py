from __future__ import annotations
import os
import asyncio
import logging
from typing import Any, Dict, List, Optional
import numpy as np
from neo4j import AsyncGraphDatabase
from apiEmbedding import (
    APIKeyManager,
    make_openai_embedding_request,
)
from config import EMBEDDING_MODEL,NEO4J_URI_BOLT, NEO4J_USER,NEO4J_PASSWORD

# --------------------------------------------------------------------------- #
# Configuration & logging                                                     #
# --------------------------------------------------------------------------- #

if not all([NEO4J_URI_BOLT, NEO4J_USER, NEO4J_PASSWORD]):
    raise EnvironmentError("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD must be set")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
log = logging.getLogger("vector_neo4j")

embedding_api_key_manager = APIKeyManager.from_env("EMBEDDING")

# --------------------------------------------------------------------------- #
# Connection Pool Manager Class                                               #
# --------------------------------------------------------------------------- #
class Neo4jConnectionManager:
    """Manages Neo4j connection with connection pooling and keep-alive."""
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        max_connection_lifetime: int = 3600,  # 1 hour
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
        keep_alive: bool = True
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._lock = asyncio.Lock()
        
        # Connection pool configuration - only include supported settings
        self.config = {
            "max_connection_lifetime": max_connection_lifetime,
            "max_connection_pool_size": max_connection_pool_size,
            "connection_acquisition_timeout": connection_acquisition_timeout,
            "keep_alive": keep_alive,
        }
        
        # Add encryption settings only if using appropriate URI schemes
        # Add encryption settings only if using appropriate URI schemes
        self._configure_encryption()
    
    def _configure_encryption(self):
        """Configure encryption settings based on URI scheme."""
        uri_lower = self.uri.lower()
        
        # Encryption settings only work with these URI schemes:
        # bolt, neo4j (for manual encryption config)
        # bolt+s, bolt+ssc, neo4j+s, neo4j+ssc (for automatic encryption)
        
        if any(scheme in uri_lower for scheme in ['bolt+s', 'bolt+ssc', 'neo4j+s', 'neo4j+ssc']):
            # These schemes automatically handle encryption, no additional config needed
            log.info(f"Using encrypted URI scheme: {uri_lower}")
        elif any(scheme in uri_lower for scheme in ['bolt://', 'neo4j://']):
            # These schemes support manual encryption configuration
            # Only add encryption settings if you want encrypted connections
            # For local development, you might not need encryption
            if os.getenv("NEO4J_ENCRYPTED", "false").lower() == "true":
                self.config.update({
                    "encrypted": True,
                    "trust": os.getenv("NEO4J_TRUST_STRATEGY", "TRUST_ALL_CERTIFICATES")
                })
                log.info("Encryption enabled for bolt/neo4j URI")
            else:
                log.info("Encryption disabled for local development")
        else:
            # Other schemes like bolt+routing, etc.
            log.info(f"Using URI scheme: {uri_lower} (no encryption config needed)")
    
    async def get_driver(self):
        """Get or create the Neo4j driver with connection pooling."""
        if self._driver is None:
            async with self._lock:
                if self._driver is None:  # Double-check locking
                    log.info("Creating new Neo4j driver with connection pooling")
                    self._driver = AsyncGraphDatabase.driver(
                        self.uri,
                        auth=(self.user, self.password),
                        **self.config
                    )
                    # Verify connectivity
                    try:
                        await self._driver.verify_connectivity()
                        log.info("Neo4j connection verified successfully")
                    except Exception as e:
                        log.error(f"Failed to verify Neo4j connectivity: {e}")
                        await self._driver.close()
                        self._driver = None
                        raise
        return self._driver
    
    async def close(self):
        """Close the driver and all connections."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            log.info("Neo4j driver closed")
    
    async def get_session(self, **kwargs):
        """Get a session from the connection pool."""
        driver = await self.get_driver()
        return driver.session(**kwargs)

# Global connection manager instance
_connection_manager = Neo4jConnectionManager(
    uri=NEO4J_URI_BOLT,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    max_connection_lifetime=3600,  # 1 hour
    max_connection_pool_size=20,   # Adjust based on your needs
    keep_alive=True
)

# --------------------------------------------------------------------------- #
# Session Context Manager for Reuse                                          #
# --------------------------------------------------------------------------- #
class Neo4jSessionPool:
    """Manages a pool of reusable Neo4j sessions."""
    
    def __init__(self, connection_manager: Neo4jConnectionManager, pool_size: int = 5):
        self.connection_manager = connection_manager
        self.pool_size = pool_size
        self._sessions = asyncio.Queue(maxsize=pool_size)
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def _initialize_pool(self):
        """Initialize the session pool."""
        if not self._initialized:
            async with self._lock:
                if not self._initialized:
                    log.info(f"Initializing Neo4j session pool with {self.pool_size} sessions")
                    for _ in range(self.pool_size):
                        session = await self.connection_manager.get_session()
                        await self._sessions.put(session)
                    self._initialized = True
    
    async def get_session(self):
        """Get a session from the pool."""
        await self._initialize_pool()
        try:
            # Try to get a session with timeout
            session = await asyncio.wait_for(self._sessions.get(), timeout=5.0)
            return session
        except asyncio.TimeoutError:
            # If no session available, create a new one
            log.warning("Session pool exhausted, creating new session")
            return await self.connection_manager.get_session()
    
    async def return_session(self, session):
        """Return a session to the pool."""
        try:
            # Check if session is still valid
            if session and not session.closed():
                await self._sessions.put(session)
            else:
                # Session is closed, create a new one for the pool
                new_session = await self.connection_manager.get_session()
                await self._sessions.put(new_session)
        except Exception as e:
            log.warning(f"Error returning session to pool: {e}")
    
    async def close_all(self):
        """Close all sessions in the pool."""
        while not self._sessions.empty():
            try:
                session = await self._sessions.get()
                await session.close()
            except Exception as e:
                log.warning(f"Error closing session: {e}")

# Global session pool
_session_pool = Neo4jSessionPool(_connection_manager, pool_size=5)

# --------------------------------------------------------------------------- #
# Optimized semantic search function                                         #
# --------------------------------------------------------------------------- #
async def semantic_search(
    question: str,
    index_name: str,
    doc_id: Optional[str] = None,
    *,
    num_results: int = 5,
    use_session_pool: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform semantic search using Neo4j's native vector index with optimized connections.
    
    Args:
        question: The search query
        index_name: Name of the Neo4j vector index
        doc_id: Optional document ID filter
        num_results: Number of results to return
        use_session_pool: Whether to use session pooling (recommended for high-frequency requests)
    """
    log.info(f"Performing Neo4j semantic search for: '{question[:50]}...' on index '{index_name}' with num_results={num_results}")
    
    # 1️⃣ Embed the question
    question_embedding_np: Optional[np.ndarray] = await make_openai_embedding_request(
        embedding_api_key_manager,
        question,
        model=EMBEDDING_MODEL,
    )
    
    if question_embedding_np is None:
        log.error("Failed to get embedding for the question.")
        return []
    
    question_embedding_list: List[float] = question_embedding_np.tolist()
    
    # 2️⃣ Prepare Cypher query
    cypher_query = """
    CALL db.index.vector.queryNodes($index_name, $k_neighbors, $embedding)
    YIELD node AS entity, score
    WHERE $doc_id IS NULL OR entity.doc_id = $doc_id
    
    OPTIONAL MATCH (entity)-[:MENTIONS]-(c:Chunk)
    WITH entity, score, c.content AS resolved_content 
    
    RETURN
        entity.descriptions AS descriptions,
        entity.type AS type,
        entity.routing_status AS routing_status,
        entity.gathering_requirement_status AS gathering_requirement_status,
        entity.gathering_requirement_information AS gathering_requirement_information,
        entity.how_to_gather_requirement AS how_to_gather_requirement,
        entity.step_to_assist_user AS step_to_assist_user,
        resolved_content AS content,
        score AS similarity
    LIMIT $limit_val
    """
    
    params = {
        "index_name": index_name,
        "k_neighbors": num_results,
        "embedding": question_embedding_list,
        "doc_id": doc_id,
        "limit_val": num_results
    }
    
    results: List[Dict[str, Any]] = []
    session = None
    
    try:
        # 3️⃣ Execute query with optimized connection handling
        if use_session_pool:
            session = await _session_pool.get_session()
        else:
            session = await _connection_manager.get_session()
        
        log.debug(f"Executing Cypher query with params: index_name={params['index_name']}, k_neighbors={params['k_neighbors']}, doc_id={params['doc_id']}, limit_val={params['limit_val']}")
        
        records = await session.run(cypher_query, params)
        results = await records.data()
        
        log.info(f"Neo4j search returned {len(results)} results.")
        
    except Exception as e:
        log.error(f"Error during Neo4j semantic search: {e}")
        return []
    
    finally:
        # 4️⃣ Return session to pool or close it
        if session:
            if use_session_pool:
                await _session_pool.return_session(session)
            else:
                await session.close()
    
    return results

# --------------------------------------------------------------------------- #
# Batch processing for multiple queries                                      #
# --------------------------------------------------------------------------- #
async def batch_semantic_search(
    questions: List[str],
    index_name: str,
    doc_id: Optional[str] = None,
    *,
    num_results: int = 5,
    max_concurrent: int = 10
) -> List[List[Dict[str, Any]]]:
    """
    Process multiple semantic search queries concurrently with connection reuse.
    
    Args:
        questions: List of search queries
        index_name: Name of the Neo4j vector index
        doc_id: Optional document ID filter
        num_results: Number of results per query
        max_concurrent: Maximum number of concurrent requests
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(question: str):
        async with semaphore:
            return await semantic_search(
                question=question,
                index_name=index_name,
                doc_id=doc_id,
                num_results=num_results,
                use_session_pool=True
            )
    
    log.info(f"Processing {len(questions)} queries with max_concurrent={max_concurrent}")
    
    tasks = [search_with_semaphore(q) for q in questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            log.error(f"Error processing question {i}: {result}")
            processed_results.append([])
        else:
            processed_results.append(result)
    
    return processed_results

# --------------------------------------------------------------------------- #
# Cleanup function                                                           #
# --------------------------------------------------------------------------- #
async def cleanup_connections():
    """Clean up all connections and pools."""
    log.info("Cleaning up Neo4j connections...")
    await _session_pool.close_all()
    await _connection_manager.close()

# --------------------------------------------------------------------------- #
# Context manager for automatic cleanup                                      #
# --------------------------------------------------------------------------- #
class Neo4jSearchContext:
    """Context manager for Neo4j operations with automatic cleanup."""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await cleanup_connections()
    
    async def search(self, *args, **kwargs):
        return await semantic_search(*args, **kwargs)
    
    async def batch_search(self, *args, **kwargs):
        return await batch_semantic_search(*args, **kwargs)

# --------------------------------------------------------------------------- #
# Demo function                                                              #
# --------------------------------------------------------------------------- #
async def _demo() -> None:
    log.info("Running Neo4j vector search demo with optimized connections...")
    
    async with Neo4jSearchContext() as search_ctx:
        # Get first available index
        session = await _connection_manager.get_session()
        try:
            result = await session.run("SHOW VECTOR INDEXES YIELD name")
            indexes = await result.single()
            demo_index_name = indexes["name"] if indexes and indexes["name"] else "placeholder_index_name"
        except Exception as e:
            log.error(f"Error fetching vector indexes: {e}")
            demo_index_name = "placeholder_index_name"
        finally:
            await session.close()
        
        print(f"Using index: {demo_index_name} for demo.")
        
        # Single query test
        question = "what is your location"
        results = await search_ctx.search(
            question=question,
            index_name=demo_index_name,
            doc_id="document_terms_description",
            num_results=5
        )
        
        if not results:
            print("\nNo results found for the demo query.")
            return
        
        print(f"\nFound {len(results)} chunks for single query:")
        for i, r in enumerate(results, 1):
            print(f"\nChunk {i} (sim={r.get('similarity', 0.0):.3f})")
            print(f"  Type: {r.get('type', 'N/A')}")
            print(f"  Description: {r.get('descriptions', 'N/A')}")
            content = r.get('content', '')
            print(f"  Content: {content[:120] if content else 'N/A'}...")
        
        # Batch query test
        batch_questions = [
            "what is your location",
            "how can I help you",
            "what services do you provide"
        ]
        
        batch_results = await search_ctx.batch_search(
            questions=batch_questions,
            index_name=demo_index_name,
            doc_id="document_terms_description",
            num_results=3
        )
        
        print(f"\n\nBatch processing results for {len(batch_questions)} questions:")
        for i, (question, results) in enumerate(zip(batch_questions, batch_results)):
            print(f"\nQuestion {i+1}: '{question}' -> {len(results)} results")

if __name__ == "__main__":
    try:
        asyncio.run(_demo())
    except KeyboardInterrupt:
        log.info("Demo interrupted by user")
    except Exception as e:
        log.error(f"Demo failed: {e}")
        raise