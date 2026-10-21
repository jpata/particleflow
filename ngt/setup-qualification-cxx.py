"""Restore the pinned userspace C++ runtime required by native ORT on Alma9."""

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://api.anaconda.org/download/conda-forge/libstdcxx/14.3.0/linux-64/libstdcxx-14.3.0-h934c35e_20.conda"
SHA256 = "83447d514a21052674806585721fef05979dd6700ca199dbfdef713b6113cb2b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path("/tmp/qualification/cxx"))
    args = parser.parse_args()
    args.prefix.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qualification-cxx-") as name:
        temp = Path(name)
        archive = temp / "runtime.conda"
        urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("C++ runtime package checksum mismatch")
        with zipfile.ZipFile(archive) as package:
            member = next(n for n in package.namelist() if n.startswith("pkg-") and n.endswith(".tar.zst"))
            compressed = temp / "package.tar.zst"
            compressed.write_bytes(package.read(member))
        tar = temp / "package.tar"
        with tar.open("wb") as output:
            subprocess.run(["zstd", "-d", "-c", str(compressed)], stdout=output, check=True)
        with tarfile.open(tar) as package:
            package.extractall(args.prefix, filter="data")
    (args.prefix / "source.json").write_text(json.dumps({"url": URL, "sha256": SHA256}, indent=2) + "\n")
    print(args.prefix / "lib/libstdc++.so.6")


if __name__ == "__main__":
    main()
