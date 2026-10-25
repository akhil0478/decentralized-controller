import sys
from pathlib import Path

# Support direct execution without requiring an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.planner.main import main

if __name__ == "__main__":
    raise SystemExit(main())
