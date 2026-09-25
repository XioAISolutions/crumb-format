"""
Model Registry REST API

REST API for accessing the model registry.
"""

import json
import logging
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

from .registry import ModelRegistry


logger = logging.getLogger(__name__)


def create_app(db_path: str = "model_registry.db") -> Flask:
    """Create Flask app for registry API.
    
    Args:
        db_path: Path to registry database
    
    Returns:
        Flask app
    """
    app = Flask(__name__)
    registry = ModelRegistry(db_path)
    
    @app.route('/api/models', methods=['GET'])
    def list_models():
        """List all models."""
        tags = request.args.getlist('tags')
        models = registry.list(tags=tags if tags else None)
        return jsonify({'models': models})
    
    @app.route('/api/models/<name>', methods=['GET'])
    def get_model(name: str):
        """Get model details."""
        version = request.args.get('version')
        metadata = registry.get(name, version)
        
        if not metadata:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify(metadata.to_dict())
    
    @app.route('/api/models/<name>/versions', methods=['GET'])
    def list_versions(name: str):
        """List model versions."""
        versions = registry.list_versions(name)
        return jsonify({
            'versions': [v.to_dict() for v in versions]
        })
    
    @app.route('/api/models', methods=['POST'])
    def register_model():
        """Register a new model."""
        data = request.json
        
        required_fields = ['name', 'version', 'checkpoint_path', 'config_path']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        success = registry.register(
            name=data['name'],
            version=data['version'],
            checkpoint_path=data['checkpoint_path'],
            config_path=data['config_path'],
            eval_results_path=data.get('eval_results_path'),
            checkpoint_url=data.get('checkpoint_url', ''),
            license=data.get('license', 'Apache 2.0'),
            tags=data.get('tags', []),
        )
        
        if success:
            return jsonify({'message': 'Model registered successfully'}), 201
        else:
            return jsonify({'error': 'Failed to register model'}), 500
    
    @app.route('/api/models/<name>/download', methods=['GET'])
    def get_download_url(name: str):
        """Get download URL for a model."""
        version = request.args.get('version')
        url = registry.get_download_url(name, version)
        
        if not url:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'download_url': url})
    
    @app.route('/api/models/<name>/compare', methods=['GET'])
    def compare_versions(name: str):
        """Compare two versions of a model."""
        version1 = request.args.get('version1')
        version2 = request.args.get('version2')
        
        if not version1 or not version2:
            return jsonify({'error': 'Both version1 and version2 required'}), 400
        
        comparison = registry.compare_versions(name, version1, version2)
        
        if 'error' in comparison:
            return jsonify(comparison), 404
        
        return jsonify(comparison)
    
    @app.route('/api/search', methods=['GET'])
    def search_models():
        """Search models."""
        query = request.args.get('q', '')
        
        if not query:
            return jsonify({'error': 'Query parameter q required'}), 400
        
        results = registry.search(query)
        return jsonify({'results': results})
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({'status': 'healthy'})
    
    return app


def run_server(host: str = '0.0.0.0', port: int = 5000, db_path: str = "model_registry.db"):
    """Run the registry API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        db_path: Path to registry database
    """
    app = create_app(db_path)
    app.run(host=host, port=port, debug=False)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Registry API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--db-path', default='model_registry.db', help='Database path')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting registry API server on {args.host}:{args.port}")
    
    run_server(args.host, args.port, args.db_path)

# Made with Bob
