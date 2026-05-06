"""Fyers app credentials loaded from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class FyersCreds:
    app_id: str          # e.g. "XYZ123ABC-100"
    secret_id: str
    redirect_uri: str    # must match what's registered in myapi.fyers.in


def load_creds() -> FyersCreds:
    load_dotenv()
    required = ("FYERS_APP_ID", "FYERS_SECRET_ID", "FYERS_REDIRECT_URI")
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(
            f"Missing Fyers env vars: {', '.join(missing)}. "
            f"Copy env.example to .env and fill them in."
        )
    return FyersCreds(
        app_id=os.environ["FYERS_APP_ID"].strip(),
        secret_id=os.environ["FYERS_SECRET_ID"].strip(),
        redirect_uri=os.environ["FYERS_REDIRECT_URI"].strip(),
    )
