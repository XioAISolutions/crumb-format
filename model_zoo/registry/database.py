"""
Model Registry Database Backend

SQLite/PostgreSQL backend for model registry.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metadata import ModelMetadata, VersionInfo


logger = logging.getLogger(__name__)


class RegistryDatabase:
    """Database backend for model registry."""
    
    def __init__(self, db_path: str = "model_registry.db"):
        """Initialize database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                architecture TEXT,
                created_at TEXT,
                updated_at TEXT,
                license TEXT,
                tags TEXT
            )
        """)
        
        # Versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                num_parameters INTEGER,
                hidden_dim INTEGER,
                num_layers INTEGER,
                num_heads INTEGER,
                training_tokens INTEGER,
                training_steps INTEGER,
                training_duration TEXT,
                benchmarks TEXT,
                perplexity REAL,
                inference_speed REAL,
                checkpoint_url TEXT,
                checkpoint_size_mb INTEGER,
                checkpoint_sha256 TEXT,
                created_at TEXT,
                changelog TEXT,
                deprecated INTEGER DEFAULT 0,
                base_model TEXT,
                specialization TEXT,
                UNIQUE(model_name, version),
                FOREIGN KEY(model_name) REFERENCES models(name)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON models(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_version ON versions(model_name, version)")
        
        self.conn.commit()
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        """Register a new model or version.
        
        Args:
            metadata: Model metadata
        
        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        
        try:
            # Insert or update model
            cursor.execute("""
                INSERT OR REPLACE INTO models (name, description, architecture, created_at, updated_at, license, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.name,
                metadata.description,
                metadata.architecture,
                metadata.created_at,
                metadata.updated_at,
                metadata.license,
                json.dumps(metadata.tags),
            ))
            
            # Insert version
            cursor.execute("""
                INSERT INTO versions (
                    model_name, version, num_parameters, hidden_dim, num_layers, num_heads,
                    training_tokens, training_steps, training_duration,
                    benchmarks, perplexity, inference_speed,
                    checkpoint_url, checkpoint_size_mb, checkpoint_sha256,
                    created_at, changelog, base_model, specialization
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.name,
                metadata.version,
                metadata.num_parameters,
                metadata.hidden_dim,
                metadata.num_layers,
                metadata.num_heads,
                metadata.training_tokens,
                metadata.training_steps,
                metadata.training_duration,
                json.dumps(metadata.benchmarks),
                metadata.perplexity,
                metadata.inference_speed,
                metadata.checkpoint_url,
                metadata.checkpoint_size_mb,
                metadata.checkpoint_sha256,
                metadata.created_at,
                "",  # changelog
                metadata.base_model,
                metadata.specialization,
            ))
            
            self.conn.commit()
            logger.info(f"Registered model {metadata.name} version {metadata.version}")
            return True
            
        except sqlite3.IntegrityError as e:
            logger.error(f"Failed to register model: {e}")
            return False
    
    def get_model(self, name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get model metadata.
        
        Args:
            name: Model name
            version: Model version (latest if None)
        
        Returns:
            Model metadata or None
        """
        cursor = self.conn.cursor()
        
        if version:
            cursor.execute("""
                SELECT m.*, v.*
                FROM models m
                JOIN versions v ON m.name = v.model_name
                WHERE m.name = ? AND v.version = ?
            """, (name, version))
        else:
            cursor.execute("""
                SELECT m.*, v.*
                FROM models m
                JOIN versions v ON m.name = v.model_name
                WHERE m.name = ?
                ORDER BY v.created_at DESC
                LIMIT 1
            """, (name,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return self._row_to_metadata(row)
    
    def list_models(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List all models.
        
        Args:
            tags: Filter by tags
        
        Returns:
            List of model summaries
        """
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM models ORDER BY name")
        rows = cursor.fetchall()
        
        models = []
        for row in rows:
            model_tags = json.loads(row['tags']) if row['tags'] else []
            
            # Filter by tags if specified
            if tags and not any(tag in model_tags for tag in tags):
                continue
            
            models.append({
                'name': row['name'],
                'description': row['description'],
                'architecture': row['architecture'],
                'tags': model_tags,
            })
        
        return models
    
    def list_versions(self, name: str) -> List[VersionInfo]:
        """List all versions of a model.
        
        Args:
            name: Model name
        
        Returns:
            List of version info
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT version, created_at, changelog, deprecated
            FROM versions
            WHERE model_name = ?
            ORDER BY created_at DESC
        """, (name,))
        
        versions = []
        for row in cursor.fetchall():
            versions.append(VersionInfo(
                version=row['version'],
                created_at=row['created_at'],
                changelog=row['changelog'] or "",
                deprecated=bool(row['deprecated']),
            ))
        
        return versions
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search models by name or description.
        
        Args:
            query: Search query
        
        Returns:
            List of matching models
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM models
            WHERE name LIKE ? OR description LIKE ?
            ORDER BY name
        """, (f"%{query}%", f"%{query}%"))
        
        models = []
        for row in cursor.fetchall():
            models.append({
                'name': row['name'],
                'description': row['description'],
                'architecture': row['architecture'],
            })
        
        return models
    
    def _row_to_metadata(self, row: sqlite3.Row) -> ModelMetadata:
        """Convert database row to ModelMetadata."""
        return ModelMetadata(
            name=row['name'],
            version=row['version'],
            description=row['description'],
            architecture=row['architecture'],
            num_parameters=row['num_parameters'],
            hidden_dim=row['hidden_dim'],
            num_layers=row['num_layers'],
            num_heads=row['num_heads'],
            training_tokens=row['training_tokens'],
            training_steps=row['training_steps'],
            training_duration=row['training_duration'],
            benchmarks=json.loads(row['benchmarks']) if row['benchmarks'] else {},
            perplexity=row['perplexity'],
            inference_speed=row['inference_speed'],
            checkpoint_url=row['checkpoint_url'],
            checkpoint_size_mb=row['checkpoint_size_mb'],
            checkpoint_sha256=row['checkpoint_sha256'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            license=row['license'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            base_model=row['base_model'],
            specialization=row['specialization'],
        )
    
    def close(self):
        """Close database connection."""
        self.conn.close()

# Made with Bob
