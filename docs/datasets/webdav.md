# Transfer TFDS datasets with Grid WebDAV

Use the repository's transfer scripts to upload locally prepared TensorFlow
Datasets (TFDS) to a Grid WebDAV endpoint or download them to another system.
The scripts preserve the TFDS directory layout, so the local data can be used
with MLPF's `--data-dir` option after download.

## Prerequisites

Run the examples from the repository root. You need Python 3, Apptainer, a
Davix container image, and a valid X.509 proxy with access to the endpoint.
Uploads also require write permission on the remote destination. Choose local
storage with enough free space for the datasets you download.

Create the X.509 proxy on a system with `voms-proxy-init` and your Grid
certificate configured:

```bash
voms-proxy-init -rfc -voms cms -bits 2048 -out "$HOME/x509-cms-2048"
export X509_USER_PROXY="$HOME/x509-cms-2048"
```

Renew the proxy with the same command if it expires. The scripts use
`$X509_USER_PROXY`, falling back to `~/x509-cms-2048`; use `--proxy /path/to/proxy`
to override it. They mount the proxy read-only at `/tmp/x509` inside Apptainer.

The Davix image is available at <https://jpata.web.cern.ch/jpata/davix.sif>.
The examples assume it is saved as `./davix.sif`; pass `--image /path/to/davix.sif`
if it is elsewhere. The image must contain the Grid CA certificates at
`/etc/grid-security/certificates`, or use `--capath` to specify their location
inside the container.

## Local and remote directory layout

A local TFDS data root directly contains dataset-name directories:

```text
/path/to/tfds/
└── cld_edm_ttbar_pf/
    └── 1/
        └── 3.3.0/
            ├── dataset_info.json
            ├── features.json
            └── ... dataset shards ...
```

Always pass `/path/to/tfds` as `--local-root`, rather than the dataset,
configuration, or version directory. A root can contain multiple datasets,
configurations, and versions.

The default remote root is:

```text
davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds
```

Uploads place files below it using this layout:

```text
<remote-root>/<detector>/[<campaign>/][<variant>/]<dataset>/<configuration>/<version>/
```

The campaign and variant directories are optional. Use a campaign to group a
production, and a variant to separate independent data roots such as detector
scan points. Keep the campaign name consistent across related datasets. The
remote TFDS data root is the directory immediately above the dataset names:
this is the detector directory when neither campaign nor variant is used.

The examples below use detector `cld`, campaign `v1.2.5_key4hep_2025-05-29`,
and dataset `cld_edm_ttbar_pf/1:3.3.0`. Substitute the detector, campaign,
dataset, configuration, and version for your transfer.

## Upload local datasets to WebDAV

Preview an upload with `scripts/tfds_webdav_upload.py`:

```bash
python3 scripts/tfds_webdav_upload.py \
  --local-root /path/to/source/tfds \
  --detector cld --campaign v1.2.5_key4hep_2025-05-29 \
  --dataset cld_edm_ttbar_pf --config 1 --version 3.3.0 \
  --image ./davix.sif --jobs 8 --dry-run
```

The preview reports the selected files, bytes, and destination without
transferring data. Remove `--dry-run` to upload. This example copies files from
`/path/to/source/tfds/cld_edm_ttbar_pf/1/3.3.0/` to:

```text
davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds/cld/v1.2.5_key4hep_2025-05-29/cld_edm_ttbar_pf/1/3.3.0/
```

Repeat `--dataset` to upload several datasets and `--config` to select several
configurations. Omit `--config` to select every local configuration containing
the requested version and `dataset_info.json`. Run the command separately for
each version.

Omit `--campaign` to upload directly below the detector directory. Add
`--variant ecal_1p5mm` to put a variant directory before the dataset names;
`--local-root` must point to that variant's local TFDS data root. Use
`--remote-root davs://host:port/path` to upload to another endpoint or base
directory whose parent already exists.

The uploader copies files in parallel, retries failures, and verifies remote
sizes and checksums. By default, `--verify-mode sha256-readback` streams every
remote file back and compares its SHA-256 digest with the local source. Allow
for a full download in addition to the upload traffic.

For large transfers, `--verify-mode md5-server` compares local MD5 digests and
sizes with the endpoint's `Want-Digest: md5` response without downloading the
files again. This requires endpoint support and a host installation of `curl`
with TLS trust for the endpoint.

To retry an interrupted or failed upload, rerun the same command. Existing
files with matching sizes are checksum-verified; missing files and files with
size or checksum mismatches are uploaded again. Both verification modes end
with a remote directory size check.

## Download datasets from WebDAV

### Download a data root or selected datasets

Use `scripts/tfds_webdav_tree.py` to download a remote TFDS data root:

```bash
python3 scripts/tfds_webdav_tree.py \
  --remote-url davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds/cld/v1.2.5_key4hep_2025-05-29/ \
  --local-root /path/to/downloaded/tfds \
  --image ./davix.sif --jobs 8 --dry-run
```

Remove `--dry-run` to download all datasets, configurations, and versions under
that URL. The destination must be absent or empty for the initial download.
The resulting local root directly contains directories such as
`cld_edm_ttbar_pf/1/3.3.0/`.

To download only selected dataset directories, repeat `--dataset`:

```bash
python3 scripts/tfds_webdav_tree.py \
  --remote-url davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds/cld/v1.2.5_key4hep_2025-05-29/ \
  --local-root /path/to/downloaded/tfds \
  --dataset cld_edm_ttbar_pf --dataset cld_edm_qq_pf \
  --image ./davix.sif --jobs 8
```

This selects all configurations and versions within those datasets. Add
`--resume` to retry an interrupted transfer or add datasets to an existing
local root. The script skips files whose sizes match the remote listing and
downloads missing or wrong-size files again; it does not resume partial files
at a byte offset.

Set `--remote-url` to the TFDS data root you need. For uploads without a
campaign, use the detector URL ending in `/tfds/cld/`. For a variant, use the
URL ending in the campaign and variant, such as
`/tfds/cld/granularity_scan_2025-05-29/ecal_1p5mm/`, and choose a separate
local root for that variant. The same command also supports other endpoints
and legacy roots such as `/store/user/jpata/mlpf/tensorflow_datasets/2.8.0/`.

### Download one configuration and version

For a dataset within a campaign, use `scripts/tfds_webdav.py download`:

```bash
python3 scripts/tfds_webdav.py download cld v1.2.5_key4hep_2025-05-29 \
  cld_edm_ttbar_pf 1 3.3.0 \
  --local-root /path/to/downloaded/tfds \
  --image ./davix.sif --dry-run
```

Remove `--dry-run` to download. The positional arguments are direction,
detector, campaign, dataset, configuration, and version. Add `--variant` for a
variant within a campaign, or `--remote-root` to change the endpoint's base
directory. This command requires a campaign argument; use the tree downloader
for roots without a campaign.

The command uses eight Davix threads and refuses a nonempty destination
version directory. Other datasets or versions can already exist in the local
root. To recover an interrupted download into that version directory, use the
tree downloader with `--resume`.

### Verify and use downloaded data

Both download commands check downloaded file sizes against the remote listing.
These are size checks, not checksum checks; the tree downloader's `--resume`
option also identifies existing files by size. Upload checksum verification
confirms the remote bytes at upload time. Perform a separate checksum check
if you need end-to-end confirmation after download.

Follow the [TFDS metadata and one-event checks](download.md#verify-metadata-and-one-event)
using your downloaded dataset names, versions, and local root. Pass
`/path/to/downloaded/tfds` as `--data-dir` when using these examples with MLPF.

## Monitor transfers

Transfers report file and byte progress. Interactive terminals also show a
progress bar, rate, ETA, and error count. Each transfer writes JSON status under
`~/.local/state/mlpf-tfds-webdav/` (or
`$XDG_STATE_HOME/mlpf-tfds-webdav/`) and prints its status file path at startup.
Use `--status-file /path/to/job.json` to choose a predictable location.

Read the latest transfer's status or watch it from another terminal:

```bash
python3 scripts/tfds_webdav_status.py
python3 scripts/tfds_webdav_status.py --watch
python3 scripts/tfds_webdav_status.py /path/to/job.json --json
```

Status remains available after the process exits. The phases are `preparing`,
`transferring`, `verifying`, `complete`, and `failed`. A `complete` phase means
all selected files passed that command's verification. The JSON distinguishes
completed files and bytes from newly transferred files and bytes, which is
useful for resumed jobs. It contains no proxy credentials.
