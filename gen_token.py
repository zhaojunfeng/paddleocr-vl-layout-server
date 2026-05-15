"""Generate Authorization header from API token.

Usage:
  python gen_token.py                    # Generate a random token
  python gen_token.py my-secret-token    # Use provided token
"""
import secrets
import sys

token = sys.argv[1] if len(sys.argv) > 1 else secrets.token_urlsafe(32)

print(f"Token:       {token}")
print(f"Header:      Authorization: Bearer {token}")
print()
print("Set in environment:")
print(f'  set API_TOKEN={token}' if sys.platform == "win32" else f'  export API_TOKEN="{token}"')
print()
print("Example curl:")
print(f'  curl -H "Authorization: Bearer {token}" http://localhost:8399/health')
