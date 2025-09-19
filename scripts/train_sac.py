#!/usr/bin/env python3
"""
Quick training script for SAC algorithm
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.scripts.train import main
import sys

if __name__ == "__main__":
    # Override sys.argv to use SAC by default
    if len(sys.argv) == 1:
        sys.argv.extend(["--algorithm", "sac"])
    elif "--algorithm" not in " ".join(sys.argv):
        sys.argv.extend(["--algorithm", "sac"])
    
    main()