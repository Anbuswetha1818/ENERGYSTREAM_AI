import sys
import os

# Add power_consumption subdirectory to python path so imports resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'power_consumption'))

from app import app
