"""Compatibility launcher: `python cli.py ...` == `python -m veriscript ...`."""
from veriscript.cli import main

if __name__ == "__main__":
    main()
