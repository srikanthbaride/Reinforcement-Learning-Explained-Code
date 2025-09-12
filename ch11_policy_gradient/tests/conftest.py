# Ensure repo root on sys.path so `import ch11_policy_gradient` works from any cwd
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
