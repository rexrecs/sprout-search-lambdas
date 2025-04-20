import os
import json
import logging
import psycopg2
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the embedding model once at cold start
# Assumes the model files are bundled in /opt/model or similar
MODEL_PATH = os.getenv("MODEL_PATH", "model/all-MiniLM-L6-v2")
logger.info(f"Loading embedding model from {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH)

# Database connection parameters from environment
DB_PARAMS = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
}

# SQL for inserting a new person
INSERT_SQL = """
INSERT INTO people (
    name,
    university,
    major,
    hobbies,
    sports_teams,
    description,
    embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s)
RETURNING id;
"""


def lambda_handler(event, context):
    try:
        # Parse JSON body
        payload = json.loads(event.get("body", "{}"))
        name = payload["name"]
        university = payload.get("university", "")
        major = payload.get("major", "")
        hobbies = payload.get("hobbies", "")
        sports_teams = payload.get("sports_teams", "")
        description = payload.get("description", "")

        # Combine fields into a single text for embedding
        text = (
            f"{name} is a student at {university}, majoring in {major}. "
            f"Hobbies: {hobbies}. Sports: {sports_teams}. {description}"
        )

        # Generate embedding vector (list of floats)
        embedding = model.encode(text).tolist()

        # Insert into Postgres
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute(INSERT_SQL, (
            name,
            university,
            major,
            hobbies,
            sports_teams,
            description,
            embedding
        ))
        new_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        # Return successful response
        return {
            "statusCode": 201,
            "body": json.dumps({"id": new_id}),
            "headers": {"Content-Type": "application/json"}
        }

    except Exception as e:
        logger.error(f"Error inserting person: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
