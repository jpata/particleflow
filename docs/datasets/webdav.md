# Transfer datasets with Grid WebDAV

The Tallinn Grid stores large TFDS productions, including CMS data, Key4hep campaigns, and CLD granularity scans. Use `scripts/tfds_webdav.py` to find datasets and copy them between the Grid and a local TFDS data root. Transfers use Python Requests over HTTPS with your X.509 Grid proxy.

## Setup

Run the examples with Python 3.11 in the ParticleFlow environment. To run the CLI in a lightweight environment, use:

```bash
uv run --no-project --python 3.11 --with requests --with pyyaml \
  python scripts/tfds_webdav.py --help
```

Create a proxy on a machine where your Grid certificate and `voms-proxy-init` are set up:

```bash
voms-proxy-init -rfc -voms cms -bits 2048 -out "$HOME/x509-cms-2048"
export X509_USER_PROXY="$HOME/x509-cms-2048"
```

The CLI uses `$X509_USER_PROXY` or `~/x509-cms-2048`. Pass `--proxy PATH` to select a different proxy. Local inventories and upload dry runs can run without a proxy.

## Find datasets and platform paths

```bash
python scripts/tfds_webdav.py platforms
python scripts/tfds_webdav.py list local
python scripts/tfds_webdav.py list local --data-key cld --dataset cld_edm_ttbar_pf
python scripts/tfds_webdav.py list remote --detector cld --campaign v1.4.0_50k
```

The CLI reads `data_dir` from the YAML files in `configs/training/platforms/`. The default profile is `local`, which maps `cld`, `clic`, `ild`, and `idea` to their directories under `/mnt/work/mlpf/tensorflow_datasets`. Use `--platform NAME` on another system, `--data-key KEY` when the profile key differs from the remote detector name, or `--local-root PATH` to select a directory explicitly. `platforms --json` reports both configured path templates and paths with available environment variables expanded.

`list remote` can scan the full Grid TFDS root. Add `--detector`, `--campaign`, or `--variant` to narrow the scan. Both listings accept repeatable `--dataset`, `--config`, and `--version` filters. Each row identifies a directory containing `dataset_info.json` and reports its observed file count and byte size. Add `--files` for the list of files and byte sizes, or `--json` for one JSON document on stdout. Version paths in the result are relative to the reported root.

## Download

Preview one CLD version using the desktop's configured data root:

```bash
python scripts/tfds_webdav.py download \
  --detector cld --campaign v1.4.0_50k \
  --dataset cld_edm_ttbar_pf --config 1 --version 3.4.0 \
  --resume --dry-run --json
```

Remove `--dry-run` to download. The plan shows the selected versions and totals, plus the files already present by size. Repeat `--dataset` to select multiple datasets or pass `--all` to copy a whole TFDS data root. Omit `--config` or `--version` to include all available configurations or versions in the selection. Pass the resulting local root to MLPF as `--data-dir`.

Use `--resume` to continue an interrupted download or to add data to an existing version directory. Files with the expected size are skipped. Downloads stage incomplete files next to the destination and use HTTP Range when the remote file still has the same size and validator. If the server returns the whole file, the CLI restarts that file. A file moves into place after its size is verified, and the CLI copies TFDS metadata after the other files. Use `--jobs` to set concurrency and `--limit-rate` to cap aggregate download throughput in bytes per second.

### Remote and local directory layout

The default remote base is `https://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds`. Under it, WebDAV and TFDS data are organized as:

```text
<remote-base>/<detector>/[<campaign>/][<variant>/]<dataset>/[<config>/]<version>/
<local-root>/<dataset>/[<config>/]<version>/
```

Use `--remote-url` to specify the collection directly above dataset directories when downloading or uploading. Both `https://` and `davs://` URLs are accepted. For a variant, pass its exact local TFDS directory with `--local-root`. On this desktop, the CLD granularity variant `ecal_1p5mm` is under `/mnt/work/mlpf/tensorflow_datasets/cld/granularity_scan_2025-05-29/ecal_1p5mm`.

After downloading, see [Verify metadata and one event](download.md#verify-metadata-and-one-event) for a TFDS check.

## Upload a campaign

```bash
python scripts/tfds_webdav.py upload \
  --local-root /path/to/workspace/tfds \
  --detector cld --campaign v1.4.0_50k \
  --dataset cld_edm_ttbar_pf --config 1 --version 3.4.0 \
  --dry-run --json
```

Remove `--dry-run` to create the destination directories and upload. Review the remote path first, since the command replaces a remote file when its contents differ from the local source. Uploads always resume. Files with the expected size are checked against the local checksum before being skipped. Incomplete or mismatching files are sent again in full. Rerun the same command after a failure. The CLI verifies every file and checks the final remote directory listing.

| `--verify-mode` | Verification |
|---|---|
| `sha256-readback` (default) | Stream the remote file and compare its SHA-256 with the local file |
| `md5-server` | Compare the size and server-provided MD5 using `HEAD` with `Want-Digest: md5` |

For all twelve CLD granularity-scan roots, use `scripts/upload_granularity_scan.py`. Repeat `--point` to select specific scan points.

## Monitor and automate

```bash
python scripts/tfds_webdav.py status
python scripts/tfds_webdav.py status --watch
python scripts/tfds_webdav.py status /path/to/job.json --json
```

Transfers write an atomic status file under `~/.local/state/mlpf-tfds-webdav/` or `$XDG_STATE_HOME/mlpf-tfds-webdav/`. Use `--status-file PATH` to choose a location. `status --watch --json` emits one snapshot per line until the transfer completes or fails. Transfer commands with `--json` write the plan or final status to stdout and progress to stderr. A failed command exits with a nonzero status.

The existing `tfds_webdav_tree.py`, `tfds_webdav_upload.py`, and `tfds_webdav_status.py` entry points also work with the Requests-based workflow.

## Troubleshooting

- **HTTP 401 or 403.** Check the selected X.509 proxy and its expiry with `voms-proxy-info`. Uploads also require write access at the destination.
- **TLS certificate verification.** Set `REQUESTS_CA_BUNDLE` or pass `--capath PATH` for a PEM bundle or an OpenSSL-hashed CA directory on this host. For a separate trusted WebDAV endpoint that redirects to additional storage hostnames, repeat `--redirect-host HOSTNAME` for those hosts.
- **TFDS cannot find the dataset.** Point `--data-dir` to the directory immediately above the dataset names, and check that the requested configuration and version exist there.
