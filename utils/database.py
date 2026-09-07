"""
Database Module for Sentiment Analysis History
Uses SQLite for storing analysis results
"""
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager
import os
import json

import config

logger = logging.getLogger(__name__)


def init_database():
    """Initialize database and create tables"""
    os.makedirs(os.path.dirname(config.DATABASE_PATH) if os.path.dirname(config.DATABASE_PATH) else '.', exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                query TEXT,
                total_count INTEGER NOT NULL,
                positive_count INTEGER NOT NULL,
                negative_count INTEGER NOT NULL,
                neutral_count INTEGER NOT NULL,
                model_used TEXT NOT NULL,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_platform ON analyses(platform)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at)
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(config.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def save_analysis(platform: str, query: str, total_count: int,
                   positive_count: int, negative_count: int, neutral_count: int,
                   model_used: str, results: List[tuple] = None) -> int:
    """
    Save an analysis to the database.
    
    Returns:
        Analysis ID
    """
    results_json = None
    if results:
        try:
            results_json = json.dumps([
                {"text": text[:500], "sentiment": sentiment}
                for text, sentiment in results[:1000]
            ])
        except Exception as e:
            logger.warning(f"Failed to serialize results: {e}")
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analyses 
            (platform, query, total_count, positive_count, negative_count, 
             neutral_count, model_used, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (platform, query, total_count, positive_count, negative_count,
              neutral_count, model_used, results_json))
        
        conn.commit()
        return cursor.lastrowid


def get_analysis_history(platform: str = None, limit: int = 50) -> List[Dict]:
    """Get analysis history, optionally filtered by platform"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if platform:
            cursor.execute('''
                SELECT * FROM analyses 
                WHERE platform = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (platform, limit))
        else:
            cursor.execute('''
                SELECT * FROM analyses 
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id: int) -> Optional[Dict]:
    """Get a specific analysis by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
        row = cursor.fetchone()
        
        return dict(row) if row else None


def delete_analysis(analysis_id: int) -> bool:
    """Delete an analysis by ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analyses WHERE id = ?', (analysis_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_statistics() -> Dict:
    """Get overall statistics"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_analyses,
                SUM(total_count) as total_posts,
                SUM(positive_count) as total_positive,
                SUM(negative_count) as total_negative,
                SUM(neutral_count) as total_neutral
            FROM analyses
        ''')
        
        row = cursor.fetchone()
        
        cursor.execute('''
            SELECT platform, COUNT(*) as count
            FROM analyses
            GROUP BY platform
            ORDER BY count DESC
        ''')
        
        platform_counts = cursor.fetchall()
        
        return {
            "total_analyses": row[0] or 0,
            "total_posts": row[1] or 0,
            "total_positive": row[2] or 0,
            "total_negative": row[3] or 0,
            "total_neutral": row[4] or 0,
            "platforms": [dict(row) for row in platform_counts]
        }


def clear_history() -> int:
    """Clear all analysis history"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM analyses')
        conn.commit()
        return cursor.rowcount
