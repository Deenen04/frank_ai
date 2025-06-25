import os
import redis
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def delete_all_redis_data():
    redis_url = os.getenv("REDIS")

    if not redis_url:
        raise ValueError("Environment variable 'REDIS' not set")

    # Parse the redis URL
    parsed_url = urlparse(redis_url)

    # Extract connection info
    host = parsed_url.hostname
    port = parsed_url.port or 6379
    password = parsed_url.password
    db = int(parsed_url.path.strip('/')) if parsed_url.path else 0

    # Connect to Redis
    r = redis.Redis(
        host=host,
        port=port,
        password=password,
        db=db,
        decode_responses=True
    )

    try:
        r.ping()
        print("Connected to Redis.")
        deleted_keys = r.flushdb()
        print("All data deleted.")
    except Exception as e:
        print("Failed to connect or delete data:", e)

if __name__ == "__main__":
    delete_all_redis_data()
