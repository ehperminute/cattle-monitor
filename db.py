import os
import sqlite3
from typing import Any, Iterable

from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cows (
            cow_id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_id TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            age REAL NOT NULL,
            weight REAL NOT NULL,
            body_temperature REAL NOT NULL,
            heart_rate REAL NOT NULL,
            appetite_loss INTEGER NOT NULL,
            vomiting INTEGER NOT NULL,
            diarrhea INTEGER NOT NULL,
            coughing INTEGER NOT NULL,
            prediction TEXT,
            healthy_probability REAL,
            sick_probability REAL,
            status TEXT,
            recommendation TEXT,
            source TEXT DEFAULT 'seed',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cow_id) REFERENCES cows(cow_id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS case_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_id TEXT NOT NULL,
            observation_id INTEGER,
            note_type TEXT NOT NULL,
            note_text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cow_id) REFERENCES cows(cow_id),
            FOREIGN KEY (observation_id) REFERENCES observations(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS follow_up_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_id TEXT NOT NULL,
            observation_id INTEGER,
            action_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (cow_id) REFERENCES cows(cow_id),
            FOREIGN KEY (observation_id) REFERENCES observations(id)
        )
        """
    )

    conn.commit()
    conn.close()


def query_all(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    conn.close()
    return rows


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    row = cur.fetchone()
    conn.close()
    return row


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    conn.commit()
    lastrowid = cur.lastrowid
    conn.close()
    return lastrowid
