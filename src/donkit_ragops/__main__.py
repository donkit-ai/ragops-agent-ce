from __future__ import annotations

import warnings

# Suppress warnings from transitive dependencies we don't control
#
# 1. pypdf (via donkit-read-engine) uses deprecated ARC4 import from cryptography
#    - Issue: pypdf 3.17.4 uses old cryptography API
#    - Fix: Would require updating pypdf, but donkit-read-engine may not support newer versions
#    - Status: Known issue in dependency, safe to suppress
#
# 2. matplotlib Axes3D import warning
#    - Issue: Multiple matplotlib installations or version conflict
#    - Fix: matplotlib not a direct dependency, installed in user directory
#    - Status: Warning doesn't affect functionality (3D plots not used)
#
# These warnings come from dependencies and don't affect our code functionality.
# Suppressing them is a pragmatic solution until dependencies are updated.
warnings.filterwarnings("ignore", message=".*Unable to import Axes3D.*", module="matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.projections")
warnings.filterwarnings("ignore", message=".*ARC4 has been moved.*")
try:
    from cryptography.utils import CryptographyDeprecationWarning

    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    pass

from donkit_ragops.cli import app

if __name__ == "__main__":
    # Allow running as: python -m donkit_ragops
    app()
