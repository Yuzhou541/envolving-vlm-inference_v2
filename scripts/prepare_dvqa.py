"""Prepare DVQA transfer manifest."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_transfer_dataset import main

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv[1:1] = ["--dataset", "DVQA"]
    main()
