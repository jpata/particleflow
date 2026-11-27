import base64
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

import pytest
import requests

DAV = "{DAV:}"


@pytest.fixture
def webdav(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import tfds_webdav

    return tfds_webdav


class MemoryWebDAV(requests.adapters.BaseAdapter):
    """Exercise Requests preparation and WebDAV redirects against an in-memory store."""

    def __init__(self):
        self.collections = {"/data"}
        self.files = {}
        self.calls = []
        self.allow_ranges = True

    def close(self):
        pass

    def listing(self, path, depth):
        tree = ET.Element(DAV + "multistatus")
        children = [path]
        if depth == "1":
            children += sorted(name for name in self.collections.union(self.files) if name.rsplit("/", 1)[0] == path)
        for name in children:
            row = ET.SubElement(tree, DAV + "response")
            ET.SubElement(row, DAV + "href").text = quote(name)
            propstat = ET.SubElement(row, DAV + "propstat")
            prop = ET.SubElement(propstat, DAV + "prop")
            kind = ET.SubElement(prop, DAV + "resourcetype")
            if name in self.collections:
                ET.SubElement(kind, DAV + "collection")
            else:
                ET.SubElement(prop, DAV + "getcontentlength").text = str(len(self.files[name]))
                ET.SubElement(prop, DAV + "getlastmodified").text = "Fri, 27 Nov 2026 00:00:00 GMT"
            ET.SubElement(propstat, DAV + "status").text = "HTTP/1.1 200 OK"
        return ET.tostring(tree)

    def send(self, request, **kwargs):
        response = requests.Response()
        response.url = request.url
        response.request = request
        response._content = b""
        response._content_consumed = True
        method = request.method
        path = urlsplit(request.url).path.rstrip("/")
        self.calls.append((method, path, request.headers.get("Range")))
        if not path.startswith("/storage/") and method in ("GET", "HEAD", "PUT"):
            response.status_code = 302 if method == "GET" else 307
            response.headers["Location"] = "/storage" + path
            if method == "PUT" and hasattr(request.body, "read"):
                request.body.read(2)
            return response

        path = path.removeprefix("/storage")
        if method == "PROPFIND":
            response.status_code = 207 if path in self.collections else 404
            if response.ok:
                response._content = self.listing(path, request.headers["Depth"])
        elif method == "MKCOL":
            if path.rsplit("/", 1)[0] in self.collections:
                self.collections.add(path)
                response.status_code = 201
            else:
                response.status_code = 409
        elif method == "PUT":
            payload = request.body.read(1024) if hasattr(request.body, "read") else request.body or b""
            assert request.headers["Content-Length"] == str(len(payload))
            self.files[path] = payload
            response.status_code = 201
        elif path not in self.files:
            response.status_code = 404
        else:
            payload = self.files[path]
            response.status_code = 200
            if method == "HEAD":
                response.headers["Digest"] = "md5=" + base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")
            elif self.allow_ranges and "Range" in request.headers:
                offset = int(request.headers["Range"].removeprefix("bytes=").removesuffix("-"))
                response.status_code = 206
                response.headers["Content-Range"] = f"bytes {offset}-{len(payload) - 1}/{len(payload)}"
                payload = payload[offset:]
            response.headers["Content-Length"] = str(len(payload))
            if method == "GET":
                response._content = payload
        return response


def test_transfer_resume_and_json(webdav, tmp_path, monkeypatch, capsys):
    from tfds_webdav_transfer import partial_paths

    storage = MemoryWebDAV()
    session = requests.Session()
    session.mount("https://example.org", storage)
    monkeypatch.setattr(webdav.WebDAV, "session", lambda self: session)
    proxy = tmp_path / "test-proxy.pem"
    proxy.touch()
    relative = "sample/1/3.3.0"
    local = tmp_path / "source" / relative
    local.mkdir(parents=True)
    (local / "dataset_info.json").write_text("{}")
    (local / "shard").write_bytes(b"correct")
    remote = "https://example.org/data"
    remote_path = f"/data/{relative}"
    storage.collections.update({"/data/sample", "/data/sample/1", remote_path})
    storage.files[f"{remote_path}/shard"] = b"corrupt"  # Same size as the local shard.
    status = tmp_path / "status.json"
    common = ["--remote-url", remote, "--dataset", "sample", "--proxy", str(proxy), "--jobs", "1", "--attempts", "1", "--json"]
    upload = ["upload", "--local-root", str(tmp_path / "source"), "--status-file", str(status), *common]
    webdav.main(upload)
    assert json.loads(capsys.readouterr().out)["transferred_files"] == 2
    assert storage.files[f"{remote_path}/shard"] == b"correct"
    assert [path for method, path, _ in storage.calls if method == "PUT" and path.startswith("/storage/")] == [
        f"/storage{remote_path}/shard",
        f"/storage{remote_path}/dataset_info.json",
    ]

    webdav.main(upload)
    assert json.loads(capsys.readouterr().out)["transferred_files"] == 0

    destination = tmp_path / "target" / relative
    destination.mkdir(parents=True)
    (destination / "dataset_info.json").write_text("{}")
    part, state = partial_paths(destination / "shard")
    part.write_bytes(b"cor")
    state.write_text(json.dumps({"url": f"{remote}/{relative}/shard", "size": 7, "etag": None, "modified": "Fri, 27 Nov 2026 00:00:00 GMT"}))
    download = ["download", "--local-root", str(tmp_path / "target"), "--status-file", str(status), "--resume", *common]
    webdav.main(download)
    assert json.loads(capsys.readouterr().out)["transferred_files"] == 1
    assert ("GET", f"/storage{remote_path}/shard", "bytes=3-") in storage.calls
    assert (destination / "shard").read_bytes() == b"correct"
    assert not part.exists()
    assert not state.exists()
    webdav.main(download)
    assert json.loads(capsys.readouterr().out)["transferred_files"] == 0


def test_collection_creation_and_full_get_fallback(webdav, tmp_path, monkeypatch):
    from tfds_webdav_transfer import download_file, partial_paths

    storage = MemoryWebDAV()
    storage.allow_ranges = False
    session = requests.Session()
    session.mount("https://example.org", storage)
    monkeypatch.setattr(webdav.WebDAV, "session", lambda self: session)
    proxy = tmp_path / "test-proxy.pem"
    proxy.touch()
    root = "https://example.org/data"
    version = f"{root}/sample/3.4.0"
    address = f"{version}/shard"
    source = tmp_path / "source"
    source.write_bytes(b"whole")
    path = tmp_path / "destination"

    with webdav.WebDAV(root, proxy=proxy) as client:
        client.ensure_collection(version)
        client.put(address, source, 5)
        assert client.checksum(address) == (5, hashlib.md5(b"whole").digest())
        entry = client.list(version)["shard"]
        part, state = partial_paths(path)
        part.write_bytes(b"wh")
        state.write_text(json.dumps({"url": address, "size": entry.size, "etag": entry.etag, "modified": entry.modified}))
        download_file(client, address, path, entry)

    assert "/data/sample/3.4.0" in storage.collections
    assert ("GET", "/storage/data/sample/3.4.0/shard", "bytes=2-") in storage.calls
    assert path.read_bytes() == b"whole"
    assert not part.exists()


def test_remote_inventory_filters_datasets_and_configs(webdav):
    from tfds_webdav_client import Entry
    from tfds_webdav_transfer import Selection, remote_versions

    class Client:
        def list(self, address):
            entries = {
                "campaign": {"wanted": Entry(), "ignored": Entry()},
                "wanted": {"1": Entry(), "2": Entry()},
                "1": {"3.4.0": Entry()},
                "2": {"3.4.0": Entry()},
                "3.4.0": {"dataset_info.json": Entry(10), "shard": Entry(20)},
            }
            return entries[address.rsplit("/", 1)[-1]]

    root = "https://example.org/campaign"
    selection = Selection(("wanted",), ("2",), ("3.4.0",))
    versions = remote_versions(Client(), root, selection, jobs=2, dataset_root=True)
    assert [(version.path, version.size) for version in versions] == [(PurePosixPath("wanted/2/3.4.0"), 30)]


def test_listing_rejects_paths_outside_requested_directory(webdav):
    from tfds_webdav_client import WebDAVError, parse_listing

    with pytest.raises(WebDAVError, match="unexpected child"):
        parse_listing(b'<multistatus xmlns="DAV:"><response><href>/data/%2e%2e</href></response></multistatus>', "https://example.org/data/")
