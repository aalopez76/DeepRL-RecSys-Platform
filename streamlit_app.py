import os
import sys

# Append the project root so internal imports from 'src' work gracefully
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.deeprl_recsys.ui.app import main

if __name__ == "__main__":
    main()
