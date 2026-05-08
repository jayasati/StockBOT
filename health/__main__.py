"""``python -m health`` entrypoint."""
from .monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
