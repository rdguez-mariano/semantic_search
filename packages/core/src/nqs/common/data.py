import hashlib
import os
from typing import Optional


def hash_from_file(filepath: str) -> Optional[str]:
    if os.path.exists(filepath):
        with open(filepath, "br") as fp:
            hash = hashlib.sha256(fp.read()).hexdigest()
        return hash
    else:
        return None
