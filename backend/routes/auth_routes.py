"""
Spotify OAuth 2.0 Authorization Code Flow routes.

Endpoints:
    GET  /api/auth/login    — Redirect user to Spotify authorize page
    GET  /api/auth/callback — Exchange auth code for tokens, redirect to frontend
    POST /api/auth/refresh  — Refresh an expired access token
    GET  /api/auth/me       — Get current user profile from Spotify
"""

import os
import base64
from urllib.parse import urlencode

import requests
from flask import Blueprint, redirect, request, jsonify

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

# Spotify OAuth endpoints
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

# Required scopes for full functionality
SCOPES = " ".join([
    "user-read-private",
    "user-read-email",
    "user-top-read",
    "playlist-modify-public",
    "playlist-modify-private",
    "user-library-modify",
    "streaming",
])


def _get_credentials():
    """Read Spotify credentials from environment."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:5000/api/auth/callback")
    return client_id, client_secret, redirect_uri


@auth_bp.route("/login")
def login():
    """
    Redirect user to Spotify authorization page.

    Query params are built from environment variables. After the user
    authorises, Spotify redirects to /api/auth/callback.
    """
    client_id, _, redirect_uri = _get_credentials()

    if not client_id:
        return jsonify({"error": "SPOTIFY_CLIENT_ID not configured"}), 500

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "show_dialog": "true",
    }

    return redirect(f"{SPOTIFY_AUTH_URL}?{urlencode(params)}")


@auth_bp.route("/callback")
def callback():
    """
    Handle Spotify OAuth callback.

    Exchanges the authorization code for access + refresh tokens,
    then redirects to the frontend with tokens as query params.
    """
    code = request.args.get("code")
    error = request.args.get("error")

    if error:
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        return redirect(f"{frontend_url}?error={error}")

    if not code:
        return jsonify({"error": "No authorization code received"}), 400

    client_id, client_secret, redirect_uri = _get_credentials()

    # Exchange code for tokens
    auth_str = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        },
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=10,
    )

    if resp.status_code != 200:
        return jsonify({
            "error": "Token exchange failed",
            "details": resp.json(),
        }), resp.status_code

    tokens = resp.json()
    access_token = tokens.get("access_token", "")
    refresh_token = tokens.get("refresh_token", "")
    expires_in = tokens.get("expires_in", 3600)

    # Redirect to frontend with tokens
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    params = urlencode({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
    })

    return redirect(f"{frontend_url}/callback?{params}")


@auth_bp.route("/refresh", methods=["POST"])
def refresh():
    """
    Refresh an expired Spotify access token.

    Expects JSON body: { "refresh_token": "..." }
    Returns new access_token and expires_in.
    """
    data = request.get_json(silent=True) or {}
    refresh_token = data.get("refresh_token")

    if not refresh_token:
        return jsonify({"error": "refresh_token required"}), 400

    client_id, client_secret, _ = _get_credentials()
    auth_str = f"{client_id}:{client_secret}"
    auth_b64 = base64.b64encode(auth_str.encode()).decode()

    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        headers={
            "Authorization": f"Basic {auth_b64}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=10,
    )

    if resp.status_code != 200:
        return jsonify({"error": "Refresh failed"}), resp.status_code

    tokens = resp.json()
    return jsonify({
        "access_token": tokens.get("access_token"),
        "expires_in": tokens.get("expires_in", 3600),
    })


@auth_bp.route("/me")
def me():
    """
    Get the authenticated user's Spotify profile.

    Expects header: Authorization: Bearer <access_token>
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Authorization header required"}), 401

    access_token = auth_header.split(" ", 1)[1]

    from backend.services.spotify_service import get_user_profile

    profile = get_user_profile(access_token)
    if profile:
        return jsonify(profile)

    return jsonify({"error": "Failed to fetch profile"}), 401
