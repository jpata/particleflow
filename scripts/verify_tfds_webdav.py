#!/usr/bin/env python3
"""Compare CLD, CLIC, IDEA PF and CLD scan files with WebDAV copies.

Run with Python 3.8+ and an X.509 proxy containing the private key.
"""

import argparse
import base64
import hashlib
import os
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REMOTE = "https://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds"
DAV = "{DAV:}"


def selected_versions(root):
    campaign = "v1.2.5_key4hep_2025-05-29"
    for detector, names in (
        ("cld", ("ttbar", "ww_fullhad", "qq", "zz")),
        ("clic", ("ttbar", "ww_fullhad", "qq")),
    ):
        for name in names:
            dataset = f"{detector}_edm_{name}_pf"
            for config in range(1, 11):
                rel = f"{detector}/{campaign}/{dataset}/{config}/3.3.0"
                local = root / detector / campaign / "tfds" / dataset / str(config) / "3.3.0"
                yield rel, local
    scan = "granularity_scan_2025-05-29"
    for kind, sizes, sample in (
        ("ecal", ("1p5", "2", "2p5", "3", "4", "5p1"), "z_tautau"),
        ("hcal", ("10", "12", "15", "20", "25", "30"), "ttbar"),
    ):
        for size in sizes:
            point = f"{kind}_{size}mm"
            for suffix, version in (("pf", "3.3.0"), ("hits", "3.2.2")):
                dataset = f"cld_edm_{sample}_{suffix}"
                for config in range(1, 11):
                    rel = f"cld/{scan}/{point}/{dataset}/{config}/{version}"
                    local = root / "cld" / scan / point / "tfds" / dataset / str(config) / version
                    yield rel, local
    idea = "IDEA_o1_v03_fccconfig_a05a3a9"
    for name in ("ttbar", "ww_fullhad", "qq"):
        dataset = f"idea_edm_{name}_pf"
        for version in ("0.2.0", "0.2.1"):
            rel = f"idea/{idea}/{dataset}/1/{version}"
            local = root / "idea" / idea / "tfds" / dataset / "1" / version
            yield rel, local


def remote_listing(proxy, rel):
    url = REMOTE + "/" + "/".join(urllib.parse.quote(p, safe="") for p in rel.split("/"))
    result = subprocess.run(
        [
            "curl",
            "-fsSL",
            "--retry",
            "3",
            "--max-time",
            "60",
            "--cert",
            str(proxy),
            "--key",
            str(proxy),
            "-X",
            "PROPFIND",
            "-H",
            "Depth: 1",
            url + "/",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(f"{rel}: {result.stderr.decode(errors='replace').strip()}")
    root = ET.fromstring(result.stdout)
    files = {}
    for response in root.findall(DAV + "response"):
        href = response.findtext(DAV + "href")
        if href is None or href.rstrip("/").endswith("/" + rel):
            continue
        kind = response.find(".//" + DAV + "resourcetype/" + DAV + "collection")
        if kind is not None:
            raise RuntimeError(f"unexpected remote subdirectory: {href}")
        name = urllib.parse.unquote(href.rstrip("/").rsplit("/", 1)[-1])
        size = response.findtext(".//" + DAV + "getcontentlength")
        if not name or size is None:
            raise RuntimeError(f"missing name/size in {rel}: {href}")
        files[name] = int(size)
    if not files or "dataset_info.json" not in files:
        raise RuntimeError(f"incomplete remote TFDS version: {rel}")
    return url, files


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(4 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


def server_md5(proxy, url):
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--head",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--max-time",
            "900",
            "--header",
            "Want-Digest: md5",
            "--cert",
            str(proxy),
            "--key",
            str(proxy),
            url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"remote MD5 request failed for {url}: {result.stderr.strip()}")
    blocks = [block for block in result.stdout.replace("\r\n", "\n").split("\n\n") if block.strip()]
    if not blocks:
        raise RuntimeError(f"remote MD5 response has no headers: {url}")
    lines = blocks[-1].splitlines()
    if len(lines) < 2 or not lines[0].startswith("HTTP/") or lines[0].split()[1] != "200":
        raise RuntimeError(f"unexpected remote MD5 response: {lines[0] if lines else 'empty'}")
    headers = {name.lower(): value.strip() for line in lines[1:] if ":" in line for name, _, value in [line.partition(":")]}
    digest = next((item.partition("=")[2] for item in headers.get("digest", "").split(",") if item.strip().lower().startswith("md5=")), None)
    if digest is None or "content-length" not in headers:
        raise RuntimeError(f"server did not return MD5 digest and size: {url}")
    return int(headers["content-length"]), base64.b64decode(digest, validate=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, default=Path(os.environ.get("X509_USER_PROXY", str(Path.home() / "x509-cms-2048"))))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--mode", choices=("md5-server", "sha256-readback"), default="md5-server")
    parser.add_argument("--scope", choices=("all", "cld-existing", "scan", "new"), default="all")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 8:
        parser.error("--jobs must be between 1 and 8")
    context = ssl.create_default_context()
    context.load_cert_chain(str(args.proxy))
    versions = list(selected_versions(args.local_root))
    if args.scope == "cld-existing":
        versions = [
            (rel, local)
            for rel, local in versions
            if rel.startswith("cld/v1.2.5_key4hep_2025-05-29/") and ("/cld_edm_ttbar_pf/" in rel or "/cld_edm_qq_pf/" in rel)
        ]
    elif args.scope == "scan":
        versions = [(rel, local) for rel, local in versions if rel.startswith("cld/granularity_scan_2025-05-29/")]
    elif args.scope == "new":
        versions = [
            (rel, local)
            for rel, local in versions
            if rel.startswith("clic/")
            or rel.startswith("idea/")
            or (rel.startswith("cld/v1.2.5_key4hep_2025-05-29/") and ("/cld_edm_ww_fullhad_pf/" in rel or "/cld_edm_zz_pf/" in rel))
        ]
    manifest = []
    errors = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(remote_listing, args.proxy, rel): (rel, local) for rel, local in versions}
        for done, future in enumerate(as_completed(futures), 1):
            rel, local = futures[future]
            try:
                url, expected = future.result()
                actual = {p.name: p.stat().st_size for p in local.iterdir() if p.is_file()} if local.is_dir() else {}
                if actual != expected:
                    missing = sorted(expected.keys() - actual.keys())
                    extra = sorted(actual.keys() - expected.keys())
                    wrong = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
                    raise RuntimeError(f"file manifest differs: missing={missing}, extra={extra}, wrong_size={wrong}")
                manifest.extend((url + "/" + urllib.parse.quote(name, safe=""), local / name, size) for name, size in expected.items())
            except Exception as exc:
                errors.append(f"{rel}: {exc}")
            if done % 40 == 0 or done == len(versions):
                print(f"Listed {done}/{len(versions)} versions; {len(errors)} errors", flush=True)
    if errors:
        for error in errors:
            print("ERROR", error, file=sys.stderr)
        raise SystemExit(f"Manifest check failed for {len(errors)} version(s)")
    print(f"Comparing {len(manifest)} files, {sum(item[2] for item in manifest):,} bytes using {args.mode}", flush=True)

    def compare(item):
        url, path, expected_size = item
        if args.mode == "md5-server":
            local_hash = hashlib.md5()
            with path.open("rb") as stream:
                while True:
                    block = stream.read(4 * 1024 * 1024)
                    if not block:
                        break
                    local_hash.update(block)
            if path.stat().st_size != expected_size:
                raise RuntimeError(f"local size changed: {path}")
            remote_size, remote_hash = server_md5(args.proxy, url)
            if remote_size != expected_size or local_hash.digest() != remote_hash:
                raise RuntimeError(f"MD5 mismatch: {path} local={local_hash.hexdigest()} remote={remote_hash.hex()}")
        else:
            with path.open("rb") as stream:
                local_size, local_hash = digest_stream(stream)
            if local_size != expected_size:
                raise RuntimeError(f"local size changed: {path}")
            with urllib.request.urlopen(url, context=context, timeout=120) as stream:
                remote_size, remote_hash = digest_stream(stream)
            if remote_size != expected_size or local_hash != remote_hash:
                raise RuntimeError(f"SHA-256 mismatch: {path} local={local_hash} remote={remote_hash}")
        return expected_size

    verified = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(compare, item): item for item in manifest}
        for future in as_completed(futures):
            try:
                total_bytes += future.result()
                verified += 1
            except Exception as exc:
                errors.append(str(exc))
                print("ERROR", exc, file=sys.stderr, flush=True)
            if (verified + len(errors)) % 100 == 0:
                print(f"Verified {verified}/{len(manifest)} files, {total_bytes:,} bytes; {len(errors)} errors", flush=True)
    if errors:
        raise SystemExit(f"Checksum comparison failed for {len(errors)} file(s)")
    print(f"PASS: {verified} files and {total_bytes:,} bytes match remote using {args.mode}", flush=True)


if __name__ == "__main__":
    main()
