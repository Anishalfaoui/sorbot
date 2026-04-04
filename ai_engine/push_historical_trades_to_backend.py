"""
Push historical simulation trades (CSV) to backend DB via secured API.

Usage example:
python push_historical_trades_to_backend.py \
  --csv data/historical_sim_2026_present_trades.csv \
  --summary data/historical_sim_2026_present_summary.json \
  --backend-url http://localhost:8081 \
  --username your_user --password your_pass \
  --clear-existing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests


def _sanitize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                out[k] = None
            else:
                out[k] = v
        cleaned.append(out)
    return cleaned


def _login(base_url: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{base_url}/api/auth/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError("Login response has no token")
    return token


def _register(base_url: str, username: str, password: str, email: str) -> str:
    resp = requests.post(
        f"{base_url}/api/auth/register",
        json={"username": username, "password": password, "email": email},
        timeout=30,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Register failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError("Register response has no token")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Push historical simulation trades into backend DB via API")
    parser.add_argument("--csv", required=True, help="Path to trades CSV")
    parser.add_argument("--summary", help="Path to summary JSON (optional, used for finalBalance)")
    parser.add_argument("--backend-url", default="http://localhost:8081")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--email", help="Needed only if --register-if-missing is used")
    parser.add_argument("--register-if-missing", action="store_true")
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("CSV contains no trades")

    rows = _sanitize_rows(df.to_dict(orient="records"))

    payload: Dict[str, Any] = {
        "clearExistingTrades": bool(args.clear_existing),
        "trades": rows,
    }

    if args.summary:
        summary_path = Path(args.summary)
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            final_balance = summary.get("final_balance")
            if final_balance is not None:
                payload["finalBalance"] = final_balance

    base_url = args.backend_url.rstrip("/")
    try:
        token = _login(base_url, args.username, args.password)
    except Exception as login_error:
        if not args.register_if_missing:
            raise
        if not args.email:
            raise RuntimeError(
                f"Login failed and --register-if-missing requires --email. Root error: {login_error}"
            )
        token = _register(base_url, args.username, args.password, args.email)

    resp = requests.post(
        f"{base_url}/api/trades/import",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Import failed ({resp.status_code}): {resp.text}")

    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
