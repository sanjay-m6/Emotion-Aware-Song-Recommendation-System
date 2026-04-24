"""
Flask application entry point.

Registers auth and API blueprints, initialises the EmotionService
singleton, and configures CORS for the React frontend.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so backend.* imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

# Load .env from backend/ directory
load_dotenv(Path(__file__).parent / ".env")

from backend.routes.auth_routes import auth_bp
from backend.routes.api_routes import api_bp
from backend.services.emotion_service import emotion_service


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

    # CORS — allow the React dev server
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    CORS(app, resources={r"/api/*": {"origins": [frontend_url, "http://localhost:5173"]}})

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    # Health check
    @app.route("/api/health")
    def health():
        return {
            "status": "ok",
            "model_ready": emotion_service.is_ready,
        }

    return app


# ── Startup ──────────────────────────────────────────────────────────────────

app = create_app()

# Initialise emotion detection model
try:
    emotion_service.initialize()
    print("[OK] Flask backend ready")
except Exception as e:
    print(f"[WARN] EmotionService init failed: {e}")
    print("   Backend will run but /api/detect-emotion will return 503")


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
