import logging
import asyncio
from neo4j import AsyncGraphDatabase
from config import NEO4J_URI_BOLT, NEO4J_USER, NEO4J_PASSWORD
from utils import configure_logging


# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

# Initialize Neo4j asynchronous driver
driver = AsyncGraphDatabase.driver(NEO4J_URI_BOLT, auth=(NEO4J_USER, NEO4J_PASSWORD))

async def print_entity_descriptions_and_type(doc_id="document_terms_description"):
    """
    Standalone function to print the first entity's descriptions and type
    """
    try:
        async with driver.session() as session:
            query = """
            MATCH (e:Entity)
            WHERE e.doc_id = $doc_id
            RETURN e.name AS entity_name, e.descriptions AS descriptions, e.type AS type
            LIMIT 1
            """
            result = await session.run(query, doc_id=doc_id)
            record = await result.single()
            
            if record:
                entity_name = record["entity_name"]
                descriptions = record["descriptions"]
                entity_type = record["type"]
                
                print("\n=============================================")
                print(f"Entity: {entity_name}")
                print("---------------------------------------------")
                print(f"Description: {descriptions}")
                print(f"Type: {entity_type}")
                print("=============================================\n")
                
                return {
                    "entity_name": entity_name,
                    "descriptions": descriptions,
                    "type": entity_type
                }
            else:
                print(f"No entities found for document ID: {doc_id}")
                return None
    except Exception as e:
        logging.error(f"Error fetching entity details: {e}")
        print(f"Error fetching entity details: {e}")
        return None

async def close_resources():
    """
    Close all asynchronous resources gracefully.
    """
    await driver.close()
    logging.info("Neo4j connection closed.")

async def main():
    try:
        # Print the first entity's descriptions and type field
        await print_entity_descriptions_and_type()
    finally:
        await close_resources()

if __name__ == "__main__":
    asyncio.run(main()) 