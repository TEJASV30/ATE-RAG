from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ate_rag.ingestion.ingest import main


if __name__ == "__main__":
    main()
