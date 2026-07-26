import sys
import os

# Remove 'app' from sys.modules if it's already registered to prevent circular importing
if 'app' in sys.modules:
    del sys.modules['app']

# Remove the root directory and current working directory from sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if p != root_dir and p != '']

# Insert the power_consumption folder at the beginning of the path
sys.path.insert(0, os.path.join(root_dir, 'power_consumption'))

# Now import the actual app from power_consumption/app.py
import app as power_app
app = power_app.app
