# Transfer datasets with Grid WebDAV

Large TFDS datasets that are not on the Hugging Face Hub, such as the full ten-partition productions, CMS data, and the CLD granularity scans, are kept on the Tallinn Grid storage element. This page shows how to download them to a training site and how to upload a new production.

The transfer scripts run [Davix](https://davix.web.cern.ch/) in an Apptainer container, copy files in parallel, verify them, and keep the TFDS directory layout. A downloaded root can be passed directly to MLPF as `--data-dir`.

| Script | Use it to |
|---|---|
| `scripts/tfds_webdav_tree.py` | Download a whole TFDS data root, or selected datasets from it |
| `scripts/tfds_webdav.py` | Download (or upload) exactly one dataset configuration and version |
| `scripts/tfds_webdav_upload.py` | Upload one or more datasets, with checksum verification |
| `scripts/upload_granularity_scan.py` | Upload all twelve CLD granularity-scan roots |
| `scripts/tfds_webdav_status.py` | Show the progress of a running or finished transfer |

## Quick start

With a valid Grid proxy and the Davix image in the repository root, download the CLD ttbar and qq track/cluster datasets:

```bash
python3 scripts/tfds_webdav_tree.py \
  --remote-url davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds/cld/v1.2.5_key4hep_2025-05-29/ \
  --local-root /data/tfds/cld \
  --dataset cld_edm_ttbar_pf --dataset cld_edm_qq_pf \
  --dry-run
```

Check the listed files and total size, then remove `--dry-run`. Train with `--data-dir /data/tfds/cld`.

## Prerequisites

You need Python 3, Apptainer, the Davix container image, and an X.509 proxy with read access to the endpoint (write access for uploads). The scripts use only the Python standard library, so the MLPF environment is not required.

**Proxy.** Create it where your Grid certificate and `voms-proxy-init` are set up:

```bash
voms-proxy-init -rfc -voms cms -bits 2048 -out "$HOME/x509-cms-2048"
export X509_USER_PROXY="$HOME/x509-cms-2048"
```

The scripts read `$X509_USER_PROXY`, falling back to `~/x509-cms-2048`; `--proxy PATH` overrides both. Rerun the command when the proxy expires.

**Davix image.** Download <https://jpata.web.cern.ch/jpata/davix.sif> to the repository root, or pass `--image PATH`. The image must contain the Grid CA certificates at `/etc/grid-security/certificates`; use `--capath` if they are elsewhere inside the container.

## Remote layout

The default remote root is `davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds`. Below it, datasets are organized as:

```text
<remote-root>/<detector>/[<campaign>/][<variant>/]<dataset>/<configuration>/<version>/
```

- **campaign** groups one production, e.g. `v1.2.5_key4hep_2025-05-29` (optional);
- **variant** separates independent data roots within a campaign, such as the granularity-scan point `ecal_1p5mm` (optional).

The **TFDS data root** is the directory directly above the dataset names: `<detector>/<campaign>/` for a campaign, `<detector>/<campaign>/<variant>/` for a variant, or `<detector>/` when neither is used. A local data root has the same shape:

```text
/data/tfds/cld/                 <- --local-root and MLPF --data-dir
└── cld_edm_ttbar_pf/
    └── 1/
        └── 3.3.0/
            ├── dataset_info.json
            ├── features.json
            └── ...shards
```

Always pass the data root, never a dataset or version directory, and use a separate local root for each variant.

## Download

### A whole root or selected datasets

`tfds_webdav_tree.py` copies everything below `--remote-url`, which must be a TFDS data root. Repeat `--dataset` to restrict it to some datasets (all their configurations and versions):

```bash
python3 scripts/tfds_webdav_tree.py \
  --remote-url davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds/cld/granularity_scan_2025-05-29/ecal_1p5mm/ \
  --local-root /data/tfds/cld_ecal_1p5mm \
  --jobs 8
```

The destination must be empty. To continue an interrupted download, or to add datasets to an existing root, rerun with `--resume`: files whose size matches the remote listing are skipped, and missing or wrong-size files are downloaded again from the start. The same command works for older roots outside this layout, such as `/store/user/jpata/mlpf/tensorflow_datasets/2.8.0/`.

### One configuration and version

`tfds_webdav.py` copies exactly one `<dataset>/<configuration>/<version>` from a campaign:

```bash
python3 scripts/tfds_webdav.py download cld v1.2.5_key4hep_2025-05-29 \
  cld_edm_ttbar_pf 1 3.3.0 \
  --local-root /data/tfds/cld --dry-run
```

The positional arguments are direction, detector, campaign, dataset, configuration, and version. Add `--variant` for a variant. Other datasets and versions may already exist in the local root, but the target version directory must be empty or absent; to recover an interrupted download, use the tree downloader with `--resume`.

### Check the result

Downloads compare file sizes with the remote listing; they do not recompute checksums. The bytes on the server were checksum-verified at upload. Then open the data with TFDS as in [Download a dataset](download.md#verify-metadata-and-one-event), using your local root as `data_dir`.

## Upload

Upload one production, here CLD ttbar configuration 1, version 3.3.0:

```bash
python3 scripts/tfds_webdav_upload.py \
  --local-root /path/to/workspace/tfds \
  --detector cld --campaign v1.2.5_key4hep_2025-05-29 \
  --dataset cld_edm_ttbar_pf --config 1 --version 3.3.0 \
  --dry-run
```

The dry run prints the files, total size, and destination. Without `--dry-run`, files are copied to `<remote-root>/cld/v1.2.5_key4hep_2025-05-29/cld_edm_ttbar_pf/1/3.3.0/`.

- Repeat `--dataset` and `--config` to select several; omit `--config` to take every local configuration that has this version.
- One command handles one version; run it again for another version.
- `--variant NAME` adds a variant directory; `--local-root` must then be that variant's local root. For all twelve CLD granularity-scan points, use `scripts/upload_granularity_scan.py` (`--point` selects a subset).
- Omit `--campaign` to upload directly below the detector directory.
- `--remote-root` selects another endpoint or base directory; its parent must exist.

Use a consistent campaign name for all datasets of one production.

### Verification

Every uploaded file is verified before the command succeeds, followed by a check of the total remote directory size:

| `--verify-mode` | Check | Cost |
|---|---|---|
| `sha256-readback` (default) | Streams each file back and compares its SHA-256 with the local file | Downloads everything once more |
| `md5-server` | Compares local MD5 and size with the server's `Want-Digest: md5` response | No extra transfer; needs endpoint support and host `curl` that trusts the endpoint |

Use `md5-server` for large productions. Failed copies are retried (`--attempts`, default 5). To resume an interrupted upload, rerun the same command: existing files with the right size are verified, and missing or mismatching files are uploaded again.

## Monitor a transfer

Every transfer prints file and byte progress; interactive terminals also show a progress bar, rate, and ETA. Each job writes a JSON status file under `~/.local/state/mlpf-tfds-webdav/` (or `$XDG_STATE_HOME/mlpf-tfds-webdav/`) and prints its path at startup; `--status-file PATH` chooses the location.

From another terminal:

```bash
python3 scripts/tfds_webdav_status.py            # latest job
python3 scripts/tfds_webdav_status.py --watch    # refresh until done
python3 scripts/tfds_webdav_status.py /path/to/job.json --json
```

The phase is one of `preparing`, `transferring`, `verifying`, `complete`, or `failed`; `complete` means every selected file passed verification. The status separates files already present from newly transferred ones, which is useful after a resume. It contains no proxy credentials.

## Troubleshooting

Authentication or permission errors
: Check that the proxy exists, has not expired (`voms-proxy-info`), and is the one the script reports. Uploads also need write permission at the destination.

TLS or certificate errors
: The container cannot find the Grid CA certificates. Use an image that contains them or pass `--capath`. For `md5-server`, the host `curl` must also trust the endpoint.

`Dataset ... not found` in TFDS after download
: `--data-dir` must be the local data root that directly contains the dataset-name directories, and the version in your recipe must exist there.

Download refuses to start
: The tree downloader needs an empty destination unless `--resume` is given; `tfds_webdav.py` refuses a non-empty version directory.
