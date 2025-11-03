import json
import sqlite3
import time
from typing import Optional, Dict


class StateStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS status_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    status_json TEXT NOT NULL
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def save_system_status(self, status: Dict) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO status_snapshots (ts, status_json) VALUES (?, ?)",
                (time.time(), json.dumps(status)),
            )
            con.commit()
        finally:
            con.close()

    def load_last_status(self) -> Optional[Dict]:
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT status_json FROM status_snapshots ORDER BY id DESC LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])
        finally:
            con.close()

