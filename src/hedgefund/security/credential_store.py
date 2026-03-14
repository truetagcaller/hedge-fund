"""Encrypted credential storage for broker API keys and secrets.

Uses Fernet symmetric encryption (AES-128-CBC via the ``cryptography`` library)
when available, falling back to base64 obfuscation with a loud warning when it
is not installed.  Master key is derived from the HEDGEFUND_MASTER_KEY
environment variable or auto-generated and persisted at ~/.hedgefund/.master_key.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from hedgefund.logger import get_logger

log = get_logger(__name__)

_HEDGEFUND_DIR = Path.home() / ".hedgefund"
_MASTER_KEY_FILE = _HEDGEFUND_DIR / ".master_key"
_CREDENTIALS_FILE = _HEDGEFUND_DIR / "credentials.enc"

# ---------------------------------------------------------------------------
# Encryption backend selection
# ---------------------------------------------------------------------------

try:
    from cryptography.fernet import Fernet, InvalidToken

    _HAS_FERNET = True
except ImportError:  # pragma: no cover
    _HAS_FERNET = False
    InvalidToken = Exception  # type: ignore[misc,assignment]


class _FernetBackend:
    """Encryption backend using Fernet (AES-128-CBC + HMAC-SHA256)."""

    def __init__(self, key: bytes) -> None:
        # Fernet requires a url-safe base64-encoded 32-byte key.
        derived = hashlib.sha256(key).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(derived))

    def encrypt(self, data: bytes) -> bytes:
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        return self._fernet.decrypt(token)


class _Base64Backend:
    """Obfuscation-only fallback -- NOT secure.

    XORs the data with a repeating key then base64-encodes the result.
    This is *not* encryption; it only deters casual reading.
    """

    def __init__(self, key: bytes) -> None:
        self._key = hashlib.sha256(key).digest()
        warnings.warn(
            "cryptography library not installed -- credentials are only "
            "base64 obfuscated, NOT encrypted.  Install 'cryptography' for "
            "production use.",
            UserWarning,
            stacklevel=3,
        )

    def encrypt(self, data: bytes) -> bytes:
        xored = bytes(b ^ self._key[i % len(self._key)] for i, b in enumerate(data))
        return base64.b64encode(xored)

    def decrypt(self, token: bytes) -> bytes:
        xored = base64.b64decode(token)
        return bytes(b ^ self._key[i % len(self._key)] for i, b in enumerate(xored))


# ---------------------------------------------------------------------------
# CredentialStore
# ---------------------------------------------------------------------------


class CredentialStore:
    """Encrypted, namespace-based credential storage.

    Credentials are kept in-memory as a nested dict and flushed to
    ``~/.hedgefund/credentials.enc`` on every write.

    Namespaces follow the convention ``"broker:zerodha"``,
    ``"datasource:twitter"``, etc.
    """

    def __init__(self, *, credentials_path: Optional[Path] = None) -> None:
        self._credentials_path = credentials_path or _CREDENTIALS_FILE
        self._master_key = self._resolve_master_key()
        self._backend = self._make_backend(self._master_key)
        self._data: Dict[str, Dict[str, str]] = {}
        self._load()

    # -- key management -----------------------------------------------------

    @staticmethod
    def _resolve_master_key() -> bytes:
        """Return the master key, creating one if necessary."""
        env_key = os.environ.get("HEDGEFUND_MASTER_KEY")
        if env_key:
            log.info("credential_store.master_key_source", source="environment")
            return env_key.encode()

        _HEDGEFUND_DIR.mkdir(parents=True, exist_ok=True)

        if _MASTER_KEY_FILE.exists():
            key = _MASTER_KEY_FILE.read_bytes().strip()
            log.info("credential_store.master_key_source", source="file")
            return key

        # Generate a new key
        key = Fernet.generate_key() if _HAS_FERNET else base64.urlsafe_b64encode(os.urandom(32))
        _MASTER_KEY_FILE.write_bytes(key)
        os.chmod(_MASTER_KEY_FILE, 0o600)
        log.info("credential_store.master_key_generated", path=str(_MASTER_KEY_FILE))
        return key

    @staticmethod
    def _make_backend(key: bytes) -> Any:
        if _HAS_FERNET:
            return _FernetBackend(key)
        return _Base64Backend(key)

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if not self._credentials_path.exists():
            self._data = {}
            return
        try:
            raw = self._credentials_path.read_bytes()
            decrypted = self._backend.decrypt(raw)
            self._data = json.loads(decrypted.decode())
            log.info(
                "credential_store.loaded",
                namespace_count=len(self._data),
            )
        except (InvalidToken, json.JSONDecodeError, Exception) as exc:
            log.error("credential_store.load_failed", error=type(exc).__name__)
            self._data = {}

    def _flush(self) -> None:
        """Persist current state to disk (encrypted)."""
        _HEDGEFUND_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data).encode()
        encrypted = self._backend.encrypt(payload)
        self._credentials_path.write_bytes(encrypted)
        os.chmod(self._credentials_path, 0o600)

    # -- public API ---------------------------------------------------------

    def store(self, namespace: str, key: str, value: str) -> None:
        """Store a credential under *namespace* / *key*.

        The value is never logged.
        """
        if namespace not in self._data:
            self._data[namespace] = {}
        self._data[namespace][key] = value
        self._flush()
        log.info("credential_store.stored", namespace=namespace, key=key)

    def retrieve(self, namespace: str, key: str) -> Optional[str]:
        """Retrieve a decrypted credential, or ``None`` if missing."""
        ns = self._data.get(namespace)
        if ns is None:
            return None
        return ns.get(key)

    def delete(self, namespace: str, key: str) -> None:
        """Delete a single credential."""
        ns = self._data.get(namespace)
        if ns is None:
            return
        ns.pop(key, None)
        if not ns:
            del self._data[namespace]
        self._flush()
        log.info("credential_store.deleted", namespace=namespace, key=key)

    def list_namespaces(self) -> List[str]:
        """Return all stored namespaces."""
        return list(self._data.keys())

    def list_keys(self, namespace: str) -> List[str]:
        """Return all keys within a namespace."""
        ns = self._data.get(namespace, {})
        return list(ns.keys())

    def export_encrypted(self, filepath: str | Path) -> None:
        """Export all credentials to *filepath* (encrypted with current key).

        The export file can be used as a backup and re-imported with
        ``import_encrypted``.
        """
        filepath = Path(filepath)
        payload = json.dumps(self._data).encode()
        encrypted = self._backend.encrypt(payload)
        filepath.write_bytes(encrypted)
        os.chmod(filepath, 0o600)
        log.info("credential_store.exported", path=str(filepath))

    def import_encrypted(self, filepath: str | Path, master_key: str) -> None:
        """Import credentials from a backup file encrypted with *master_key*.

        Existing credentials for overlapping namespaces/keys will be
        overwritten.
        """
        filepath = Path(filepath)
        backend = self._make_backend(master_key.encode())
        try:
            raw = filepath.read_bytes()
            decrypted = backend.decrypt(raw)
            imported: Dict[str, Dict[str, str]] = json.loads(decrypted.decode())
        except (InvalidToken, json.JSONDecodeError, Exception) as exc:
            log.error("credential_store.import_failed", error=type(exc).__name__)
            raise ValueError("Failed to decrypt/parse import file") from exc

        for ns, keys in imported.items():
            if ns not in self._data:
                self._data[ns] = {}
            self._data[ns].update(keys)

        self._flush()
        log.info(
            "credential_store.imported",
            path=str(filepath),
            namespace_count=len(imported),
        )

    # -- key rotation -------------------------------------------------------

    def rotate_key(self, new_master_key: str) -> None:
        """Re-encrypt all credentials with a new master key.

        The old data is decrypted in memory, the backend is replaced, and
        everything is flushed under the new key.  The master key file is
        updated as well.
        """
        new_key = new_master_key.encode()
        self._master_key = new_key
        self._backend = self._make_backend(new_key)
        self._flush()

        # Update persisted master key file (unless sourced from env)
        if not os.environ.get("HEDGEFUND_MASTER_KEY"):
            _MASTER_KEY_FILE.write_bytes(new_key)
            os.chmod(_MASTER_KEY_FILE, 0o600)

        log.info("credential_store.key_rotated")
