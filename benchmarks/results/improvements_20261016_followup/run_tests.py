"""Pin the repository/native build and safely run tests on Windows spawn."""
import hashlib
import json
from pathlib import Path
import sys


def main():
    package = Path(sys.argv[1]).resolve()
    sys.meta_path = [finder for finder in sys.meta_path
                    if type(finder).__module__ != "_editable_skbc_ctboost"]
    sys.path.insert(0, str(package))
    import ctboost
    import pytest

    assert Path(ctboost.__file__).resolve().is_relative_to(package)
    assert Path(ctboost._core.__file__).resolve().is_relative_to(package)
    print(json.dumps({"package": ctboost.__file__, "native": ctboost._core.__file__,
                      "native_sha256": hashlib.sha256(Path(ctboost._core.__file__).read_bytes()).hexdigest(),
                      "build_info": ctboost.build_info()}), flush=True)
    return pytest.main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
