import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ui import render_app

if __name__ == "__main__":
    render_app()
