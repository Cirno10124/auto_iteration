import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pymysql  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pymysql = None


class MySQLEmbeddingStore:
    def __init__(self, config: dict, similarity_threshold: float = 0.85):
        self._config = config or {}
        self.similarity_threshold = similarity_threshold
        self.enabled = bool(self._config.get("enabled", False))
        self._connections: Dict[str, object] = {}
        self._namespace_default = self._config.get("namespace", "default")
        self._databases = self._config.get("databases")

        if not self.enabled:
            return
        if pymysql is None:
            print("警告：pymysql 未安装，跨运行聚类持久化已禁用。")
            self.enabled = False
            return
        if not self._config.get("database") and not self._databases:
            print("警告：MySQL database 未配置，跨运行聚类持久化已禁用。")
            self.enabled = False
            return
        if self._databases is not None and not isinstance(
            self._databases, dict
        ):
            print("警告：MySQL databases 配置无效，跨运行聚类持久化已禁用。")
            self.enabled = False
            return

        try:
            self._get_connection(self._namespace_default)
        except Exception as exc:
            print(f"警告：MySQL 初始化失败，将回退到单次聚类: {exc}")
            self.enabled = False
            self._connections = {}

    def _resolve_database(self, namespace: str) -> Optional[str]:
        if isinstance(self._databases, dict) and self._databases:
            if namespace in self._databases:
                return self._databases.get(namespace)
            if self._namespace_default in self._databases:
                return self._databases.get(self._namespace_default)
            for db in self._databases.values():
                return db
            return None
        return self._config.get("database")

    def _connect(self, database: str):
        return pymysql.connect(
            host=self._config.get("host", "127.0.0.1"),
            port=int(self._config.get("port", 3306)),
            user=self._config.get("user", "root"),
            password=self._config.get("password", ""),
            database=database,
            charset=self._config.get("charset", "utf8mb4"),
            autocommit=True,
        )

    def _get_connection(self, namespace: str):
        if not self.enabled:
            return None
        database = self._resolve_database(namespace)
        if not database:
            raise ValueError("MySQL database 未配置")
        if database in self._connections:
            return self._connections[database]
        conn = self._connect(database)
        self._ensure_tables(conn)
        self._connections[database] = conn
        return conn

    def _ensure_tables(self, conn):
        if conn is None:
            return
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS speaker_clusters (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    namespace VARCHAR(255) NOT NULL,
                    speaker_name VARCHAR(255) NOT NULL DEFAULT '',
                    centroid LONGBLOB NOT NULL,
                    dim INT NOT NULL,
                    count INT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS speaker_segments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    namespace VARCHAR(255) NOT NULL,
                    audio_key VARCHAR(1024) NOT NULL,
                    start_sec DOUBLE NOT NULL,
                    end_sec DOUBLE NOT NULL,
                    cluster_id INT NOT NULL,
                    embedding LONGBLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_audio_key (audio_key),
                    INDEX idx_cluster_id (cluster_id),
                    INDEX idx_namespace (namespace),
                    CONSTRAINT fk_cluster
                        FOREIGN KEY (cluster_id) REFERENCES speaker_clusters(id)
                        ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
            self._ensure_namespace_column(cur, "speaker_clusters")
            self._ensure_namespace_column(cur, "speaker_segments")
            self._ensure_speaker_name_column(cur)

    def _ensure_namespace_column(self, cursor, table: str):
        cursor.execute(
            f"SHOW COLUMNS FROM {table} LIKE 'namespace'"  # noqa: S608
        )
        row = cursor.fetchone()
        if row:
            return
        cursor.execute(
            f"""
            ALTER TABLE {table}
            ADD COLUMN namespace VARCHAR(255) NOT NULL DEFAULT 'default',
            ADD INDEX idx_namespace (namespace)
            """  # noqa: S608
        )

    def _ensure_speaker_name_column(self, cursor):
        cursor.execute("SHOW COLUMNS FROM speaker_clusters LIKE 'speaker_name'")
        row = cursor.fetchone()
        if row:
            return
        cursor.execute(
            """
            ALTER TABLE speaker_clusters
            ADD COLUMN speaker_name VARCHAR(255) NOT NULL DEFAULT ''
            """
        )

    @staticmethod
    def _serialize_vector(vec: np.ndarray) -> bytes:
        return np.asarray(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_vector(blob: bytes, dim: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32, count=dim).astype(
            np.float32, copy=False
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm <= 0 or b_norm <= 0:
            return -1.0
        return float(np.dot(a, b) / (a_norm * b_norm))

    def _load_clusters(self, conn, namespace: str) -> List[Dict[str, object]]:
        if not self.enabled or conn is None:
            return []
        clusters: List[Dict[str, object]] = []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, centroid, dim, count
                FROM speaker_clusters
                WHERE namespace = %s
                """,
                (namespace,),
            )
            for row in cur.fetchall():
                cluster_id, centroid_blob, dim, count = row
                centroid = self._deserialize_vector(centroid_blob, int(dim))
                clusters.append(
                    {
                        "id": int(cluster_id),
                        "centroid": centroid,
                        "count": int(count),
                        "dim": int(dim),
                    }
                )
        return clusters

    def match_and_update_clusters(
        self,
        local_stats: Dict[int, Dict[str, np.ndarray]],
        namespace: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[int, int]:
        """
        local_stats: {local_label: {"centroid": vec, "sum": vec, "count": int}}
        """
        if not self.enabled:
            return {}
        namespace = namespace or self._namespace_default
        threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        conn = self._get_connection(namespace)
        if conn is None:
            return {}

        mapping: Dict[int, int] = {}
        clusters = self._load_clusters(conn, namespace)

        with conn.cursor() as cur:
            for local_label, stats in local_stats.items():
                centroid = stats["centroid"]
                local_sum = stats["sum"]
                local_count = int(stats["count"])

                best_id = None
                best_score = -1.0
                for cluster in clusters:
                    score = self._cosine_similarity(
                        centroid, cluster["centroid"]
                    )
                    if score > best_score:
                        best_score = score
                        best_id = cluster["id"]

                if best_id is None or best_score < threshold:
                    cur.execute(
                        """
                        INSERT INTO speaker_clusters
                            (namespace, centroid, dim, count)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            namespace,
                            self._serialize_vector(centroid),
                            int(centroid.shape[0]),
                            local_count,
                        ),
                    )
                    new_id = int(cur.lastrowid)
                    cur.execute(
                        """
                        UPDATE speaker_clusters
                        SET speaker_name = %s
                        WHERE id = %s AND namespace = %s
                        """,
                        (f"speaker_{new_id}", new_id, namespace),
                    )
                    mapping[local_label] = new_id
                    clusters.append(
                        {
                            "id": new_id,
                            "centroid": centroid,
                            "count": local_count,
                            "dim": int(centroid.shape[0]),
                        }
                    )
                else:
                    for cluster in clusters:
                        if cluster["id"] == best_id:
                            old_count = int(cluster["count"])
                            old_centroid = cluster["centroid"]
                            new_count = old_count + local_count
                            new_centroid = (
                                old_centroid * old_count + local_sum
                            ) / max(new_count, 1)
                            cur.execute(
                                """
                                UPDATE speaker_clusters
                                SET centroid = %s, dim = %s, count = %s
                                WHERE id = %s AND namespace = %s
                                """,
                                (
                                    self._serialize_vector(new_centroid),
                                    int(new_centroid.shape[0]),
                                    new_count,
                                    best_id,
                                    namespace,
                                ),
                            )
                            cluster["centroid"] = new_centroid
                            cluster["count"] = new_count
                            mapping[local_label] = best_id
                            break

        return mapping

    def save_segments(
        self,
        audio_key: str,
        segments: Iterable[object],
        embeddings: List[np.ndarray],
        cluster_ids: List[int],
        namespace: Optional[str] = None,
    ):
        if not self.enabled:
            return
        namespace = namespace or self._namespace_default
        conn = self._get_connection(namespace)
        if conn is None:
            return

        audio_key = os.fspath(audio_key)
        rows: List[Tuple] = []
        for turn, embedding, cluster_id in zip(
            segments, embeddings, cluster_ids
        ):
            rows.append(
                (
                    namespace,
                    audio_key,
                    float(turn.start),
                    float(turn.end),
                    int(cluster_id),
                    self._serialize_vector(embedding),
                )
            )

        if not rows:
            return
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO speaker_segments
                    (namespace, audio_key, start_sec, end_sec, cluster_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    def get_speaker_names(
        self, namespace: str, speaker_ids: List[int]
    ) -> Dict[int, str]:
        if not self.enabled or not speaker_ids:
            return {}
        conn = self._get_connection(namespace)
        if conn is None:
            return {}
        placeholders = ", ".join(["%s"] * len(speaker_ids))
        query = f"""
            SELECT id, speaker_name
            FROM speaker_clusters
            WHERE namespace = %s AND id IN ({placeholders})
        """
        with conn.cursor() as cur:
            cur.execute(query, [namespace, *speaker_ids])
            rows = cur.fetchall()
        return {int(row[0]): row[1] or f"speaker_{row[0]}" for row in rows}

    def list_speakers(
        self, namespace: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, object]]:
        if not self.enabled:
            return []
        conn = self._get_connection(namespace)
        if conn is None:
            return []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, speaker_name, count, updated_at
                FROM speaker_clusters
                WHERE namespace = %s
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
                """,
                (namespace, int(limit), int(offset)),
            )
            rows = cur.fetchall()
        return [
            {
                "speaker_id": int(row[0]),
                "speaker_name": row[1] or f"speaker_{row[0]}",
                "count": int(row[2]),
                "updated_at": row[3].isoformat() if row[3] else None,
            }
            for row in rows
        ]

    def get_speaker(
        self, namespace: str, speaker_id: int
    ) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        conn = self._get_connection(namespace)
        if conn is None:
            return None
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, speaker_name, count, updated_at
                FROM speaker_clusters
                WHERE namespace = %s AND id = %s
                """,
                (namespace, int(speaker_id)),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "speaker_id": int(row[0]),
            "speaker_name": row[1] or f"speaker_{row[0]}",
            "count": int(row[2]),
            "updated_at": row[3].isoformat() if row[3] else None,
        }

    def update_speaker_name(
        self, namespace: str, speaker_id: int, speaker_name: str
    ) -> Optional[Dict[str, object]]:
        if not self.enabled:
            return None
        conn = self._get_connection(namespace)
        if conn is None:
            return None
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE speaker_clusters
                SET speaker_name = %s
                WHERE namespace = %s AND id = %s
                """,
                (speaker_name, namespace, int(speaker_id)),
            )
        return self.get_speaker(namespace, speaker_id)
