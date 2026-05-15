"""Generate JWT token for a user.

Usage:
  python gen_token.py user@example.com              # no expiry
  python gen_token.py user@example.com 3600          # expires in 1 hour

Requires JWT_SECRET env var to match the server's.
Reads user.json to get the user's salt.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import jwt

JWT_SECRET = os.environ.get("JWT_SECRET", "")
USER_DB_PATH = os.environ.get("USER_DB_PATH", "user.json")


def main():
    if not JWT_SECRET:
        print("Error: JWT_SECRET env var not set")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python gen_token.py <email> [expires_in_seconds]")
        sys.exit(1)

    email = sys.argv[1]
    expires_in = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Load user.json
    if not os.path.exists(USER_DB_PATH):
        print(f"Error: {USER_DB_PATH} not found. Start the server first to create it.")
        sys.exit(1)

    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)

    if email not in users:
        print(f"Error: User '{email}' not found in {USER_DB_PATH}")
        print(f"Available users: {', '.join(users.keys())}")
        sys.exit(1)

    salt = users[email]["salt"]
    key = hashlib.sha256((JWT_SECRET + salt).encode()).digest()

    payload = {"email": email, "salt": salt, "iat": datetime.now(timezone.utc)}
    expires_at = None
    if expires_in is not None and expires_in > 0:
        exp_time = datetime.now(timezone.utc).timestamp() + expires_in
        payload["exp"] = exp_time
        expires_at = datetime.fromtimestamp(exp_time, tz=timezone.utc).isoformat()

    token = jwt.encode(payload, key, algorithm="HS256")

    print(f"Email:      {email}")
    print(f"Role:       {users[email].get('role', 'user')}")
    print(f"Expires:    {expires_at or 'never'}")
    print(f"Token:      {token}")
    print()
    print("Example curl:")
    print(f'  curl -H "Authorization: Bearer {token}" http://localhost:8399/health')


if __name__ == "__main__":
    main()
