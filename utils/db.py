import sqlite3
import numpy as np

class SQLiteUtils:
    def __init__(self, db_path):
        """Initialize the SQLiteUtils class."""
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            embedding_vector BLOB,
            first_name TEXT,
            last_name TEXT
        )
        """)
        self.conn.commit()

    def fetch_all(self):
        """Fetch all records from the users table."""
        cursor = self.conn.execute("SELECT * FROM users")
        return cursor.fetchall()

    def delete_all(self):
        """Delete all records from the users table."""
        self.conn.execute("DELETE FROM users")
        self.conn.commit()

    def delete_by_id(self, user_id):
        """Delete a user by ID."""
        self.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()

    def distance_similarity_fetch(self, vector, limit, max_distance):
        """
        Fetch users based on distance similarity to the input vector.

        :param vector: Input embedding vector (as a NumPy array).
        :param limit: Maximum number of results to return.
        :param max_distance: Maximum allowed distance.
        :return: List of users within the distance threshold.
        """
        cursor = self.conn.execute("SELECT id, embedding_vector, first_name, last_name FROM users")
        results = []

        for row in cursor.fetchall():
            user_id, embedding_blob, first_name, last_name = row
            stored_vector = np.frombuffer(embedding_blob, dtype=np.float32)
            distance = np.linalg.norm(stored_vector - vector)
            if distance <= max_distance:
                results.append((user_id, first_name, last_name, distance))

        # Sort by distance and apply limit
        results.sort(key=lambda x: x[3])
        return results[:limit]

    def insert_user(self, image_path, embedding_vector, first_name, last_name):
        """Insert a new user into the database."""
        embedding_blob = embedding_vector.tobytes()
        self.conn.execute(
            "INSERT INTO users (image_path, embedding_vector, first_name, last_name) VALUES (?, ?, ?, ?)",
            (image_path, embedding_blob, first_name, last_name)
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()