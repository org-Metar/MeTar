"""Allow the same CLI through ``python -m metar`` and ``metar``."""

from .cli import main

raise SystemExit(main())
