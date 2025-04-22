from sentence_transformers import SentenceTransformer
import psycopg2


class Searcher:
    def __init__(
        self,
        dbname="auth_db",
        user="user",
        password="password",
        host="localhost",
        port=5432,
        model_name="notebooks/model/all-MiniLM-L6-v2"
    ):
        self.conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.model = SentenceTransformer(model_name)

    def search(self, query, limit=25, page=0):
        embedding = self.model.encode(query).tolist()
        embedding_sql = f"[{', '.join(str(x) for x in embedding)}]"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, school, major, minor, school_location, bio, image,
                       embedding <-> %s AS distance
                FROM students
                ORDER BY embedding <-> %s
                LIMIT %s OFFSET %s;
                """,
                (embedding_sql, embedding_sql, limit, page * limit)
            )
            results = cur.fetchall()
            return [
                {
                    "data": row,
                    "similarity_score": 1 / (1 + row[7])
                }
                for row in results
            ]

    def close(self):
        self.conn.close()
