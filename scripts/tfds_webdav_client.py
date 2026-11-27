"""Requests transport for Grid WebDAV collections and files."""

import base64
import hashlib
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlsplit

import requests

DEFAULT_REMOTE = "https://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds"
CHUNK_SIZE = 1024 * 1024
DAV = "{DAV:}"
PROPERTIES = b'<propfind xmlns="DAV:"><prop><resourcetype/><getcontentlength/><getetag/><getlastmodified/></prop></propfind>'
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}


class WebDAVError(RuntimeError):
    def __init__(self, message, status=None, *, retryable=False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable or status in TRANSIENT_STATUS


@dataclass(frozen=True)
class Entry:
    size: int | None = None  # None denotes a collection.
    etag: str | None = None
    modified: str | None = None


class UploadBody:
    """A rewindable, interruptible body with a known length for Requests."""

    def __init__(self, stream, size, check_running):
        self.stream = stream
        self.size = size
        self.check_running = check_running

    def __len__(self):
        return self.size

    def read(self, size):
        self.check_running()
        return self.stream.read(size)

    def tell(self):
        return self.stream.tell()

    def seek(self, offset):
        return self.stream.seek(offset)


def safe_component(name):
    return name not in ("", ".", "..") and not any(char in name for char in "/\\\x00")


def https_url(value):
    parts = urlsplit(value)
    if (
        parts.scheme not in ("davs", "https")
        or not parts.hostname
        or parts.username is not None
        or parts.query
        or parts.fragment
        or any(not safe_component(unquote(part)) for part in parts.path.strip("/").split("/") if parts.path.strip("/"))
    ):
        raise ValueError("use a davs:// or https:// URL without credentials, a query, or a fragment")
    return parts._replace(scheme="https", path=parts.path.rstrip("/")).geturl()


def child_url(root, *names):
    if not all(safe_component(name) for name in names):
        raise ValueError("invalid WebDAV path component")
    return "/".join((root.rstrip("/"), *(quote(name, safe="") for name in names)))


def default_proxy():
    return Path(os.environ.get("X509_USER_PROXY") or "~/x509-cms-2048").expanduser()


def retry(operation, attempts):
    for attempt in range(attempts):
        try:
            return operation()
        except WebDAVError as exc:
            if not exc.retryable or attempt + 1 == attempts:
                raise
        time.sleep(min(2**attempt, 15))
    raise ValueError("attempts must be positive")


def parse_listing(content, url):
    """Return the immediate children in a Depth: 1 Multi-Status response."""
    prefix = tuple(unquote(part) for part in urlsplit(url).path.strip("/").split("/") if part)
    try:
        tree = ET.fromstring(content)
        if tree.tag != DAV + "multistatus":
            raise ValueError("expected a DAV multistatus response")
        entries = {}
        for response in tree.findall(DAV + "response"):
            href = response.findtext(DAV + "href")
            if not href:
                raise ValueError("missing DAV href")
            path = tuple(unquote(part) for part in urlsplit(href).path.strip("/").split("/") if part)
            if path == prefix:
                continue
            if path[:-1] != prefix or not safe_component(path[-1]):
                raise ValueError(f"unexpected child path {href!r}")
            name = path[-1]
            if name in entries:
                raise ValueError(f"duplicate child {name!r}")
            properties = {}
            for propstat in response.findall(DAV + "propstat"):
                status = propstat.findtext(DAV + "status", "").split()
                prop = propstat.find(DAV + "prop")
                if len(status) >= 2 and status[1].startswith("2") and prop is not None:
                    properties.update((element.tag, element) for element in prop)
            kind = properties.get(DAV + "resourcetype")
            if kind is None:
                raise ValueError(f"missing resource type for {name!r}")
            if kind.find(DAV + "collection") is not None:
                entries[name] = Entry()
                continue
            size = int(properties[DAV + "getcontentlength"].text)
            if size < 0:
                raise ValueError(f"invalid size for {name!r}")
            etag = properties.get(DAV + "getetag")
            modified = properties.get(DAV + "getlastmodified")
            entries[name] = Entry(size, etag.text if etag is not None else None, modified.text if modified is not None else None)
    except (ET.ParseError, KeyError, ValueError, TypeError, IndexError) as exc:
        raise WebDAVError(f"invalid WebDAV listing for {url}: {exc}") from exc
    return entries


def digest_blocks(blocks, algorithm):
    digest = hashlib.new(algorithm)
    size = 0
    for block in blocks:
        size += len(block)
        digest.update(block)
    return size, digest.digest()


def file_digest(path, algorithm, check_running=None):
    with path.open("rb") as stream:

        def blocks():
            while block := stream.read(CHUNK_SIZE):
                if check_running:
                    check_running()
                yield block

        return digest_blocks(blocks(), algorithm)


class WebDAV:
    def __init__(self, root, *, proxy=None, capath=None, timeout=120, attempts=5, redirect_hosts=()):
        self.root = https_url(root)
        self.proxy = (proxy or default_proxy()).expanduser()
        if not self.proxy.is_file():
            raise ValueError(f"X.509 proxy not found at {self.proxy}. Set X509_USER_PROXY or pass --proxy.")
        self.capath = str(capath.expanduser()) if capath else True
        if capath and not capath.expanduser().exists():
            raise ValueError(f"CA bundle or directory not found: {capath}")
        self.timeout = (min(10, timeout), timeout)
        self.attempts = attempts
        self.redirect_hosts = {host.lower() for host in redirect_hosts}
        self.threads = threading.local()
        self.sessions = []
        self.session_lock = threading.Lock()
        self.stopped = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for session in self.sessions:
            session.close()

    def session(self):
        if not hasattr(self.threads, "session"):
            session = requests.Session()
            session.cert = str(self.proxy)
            session.headers["Accept-Encoding"] = "identity"
            self.threads.session = session
            with self.session_lock:
                self.sessions.append(session)
        return self.threads.session

    def cancel(self):
        self.stopped.set()

    def check_running(self):
        if self.stopped.is_set():
            raise WebDAVError("transfer interrupted")

    def pace(self, delay):
        if self.stopped.wait(delay):
            self.check_running()

    def trusted_redirect(self, address):
        origin = urlsplit(self.root)
        target = urlsplit(address)
        if target.scheme != "https" or target.username is not None or not target.hostname:
            return False
        if target.hostname in self.redirect_hosts:
            return True
        if (target.port or 443) != (origin.port or 443):
            return False
        if target.hostname == origin.hostname:
            return True
        return origin.hostname == "xrootd.hep.kbfi.ee" and re.fullmatch(r"xrootd[0-9]+\.hep\.kbfi\.ee", target.hostname) is not None

    def request(self, method, url, expected, *, data=None, **kwargs):
        address = url
        for _ in range(9):
            self.check_running()
            if hasattr(data, "seek"):
                data.seek(0)
            try:
                response = self.session().request(
                    method, address, data=data, verify=self.capath, timeout=self.timeout, allow_redirects=False, **kwargs
                )
            except requests.exceptions.SSLError as exc:
                raise WebDAVError(f"{method} {url}: TLS verification or client certificate failed ({exc.__class__.__name__})") from exc
            except requests.RequestException as exc:
                raise WebDAVError(f"{method} {url}: {exc.__class__.__name__}", retryable=True) from exc
            status = response.status_code
            if status in expected:
                return response
            response.close()
            if status in (307, 308) or (method in ("GET", "HEAD") and status in (301, 302, 303)):
                target = urljoin(address, response.headers.get("Location", "")) if "Location" in response.headers else ""
                try:
                    trusted = self.trusted_redirect(target)
                except ValueError:
                    trusted = False
                if not trusted:
                    raise WebDAVError(f"{method} {url}: redirect to an untrusted HTTPS host; set --redirect-host for a trusted storage server")
                address = target
                continue
            hint = ". Check the Grid proxy and permissions." if status in (401, 403) else ""
            raise WebDAVError(f"{method} {url}: HTTP {status}{hint}", status)
        raise WebDAVError(f"{method} {url}: too many redirects")

    def list(self, url):
        url = url.rstrip("/") + "/"
        headers = {"Depth": "1", "Content-Type": "application/xml"}

        def fetch():
            with self.request("PROPFIND", url, (207,), data=PROPERTIES, headers=headers) as response:
                return parse_listing(response.content, url)

        return retry(fetch, self.attempts)

    def ensure_collection(self, url):
        url = url.rstrip("/") + "/"

        def exists():
            try:
                with self.request("PROPFIND", url, (207,), headers={"Depth": "0"}):
                    return True
            except WebDAVError as exc:
                if exc.status != 404:
                    raise
                return False

        def create():
            with self.request("MKCOL", url, (200, 201, 204, 405, 409)) as response:
                return response.status_code

        if retry(exists, self.attempts):
            return
        status = retry(create, self.attempts)
        if status == 409:
            parent = url.rstrip("/").rsplit("/", 1)[0]
            if not urlsplit(parent).path:
                raise WebDAVError(f"cannot create parent collection for {url}", status)
            self.ensure_collection(parent)
            status = retry(create, self.attempts)
        if status in (405, 409) and not retry(exists, self.attempts):
            raise WebDAVError(f"cannot create collection {url}: HTTP {status}", status)

    def put(self, url, path, size):
        with path.open("rb") as stream:
            with self.request(
                "PUT",
                url,
                (200, 201, 204),
                data=UploadBody(stream, size, self.check_running) if size else b"",
                headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"},
            ):
                pass

    def blocks(self, response, url):
        try:
            for block in response.iter_content(CHUNK_SIZE):
                self.check_running()
                yield block
        except requests.RequestException as exc:
            raise WebDAVError(f"GET {url}: interrupted response ({exc.__class__.__name__})", retryable=True) from exc

    def checksum(self, url):
        """Size and MD5 computed by the server, without transferring the file."""
        with self.request("HEAD", url, (200,), headers={"Want-Digest": "md5"}) as response:
            fields = dict(value.strip().partition("=")[::2] for value in response.headers.get("Digest", "").split(","))
            md5 = next((value for name, value in fields.items() if name.lower() == "md5"), "")
            try:
                size = int(response.headers["Content-Length"])
                digest = base64.b64decode(md5, validate=True)
                if size < 0 or len(digest) != 16:
                    raise ValueError("invalid size or digest")
            except (KeyError, ValueError) as exc:
                raise WebDAVError(f"{url}: server did not return a valid MD5 and file size") from exc
            return size, digest
