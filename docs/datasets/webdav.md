# Transfer TFDS datasets with Grid WebDAV

The Grid WebDAV tree for new production campaigns is:

```text
/store/user/jpata/mlpf/tfds/<detector>/<campaign>/<dataset>/<configuration>/<version>/
```

For a campaign with independent variants, insert `<variant>/` between
`<campaign>/` and `<dataset>/`. Each variant directory is then its own TFDS
data root. This is the layout used by the CLD granularity scan:

```text
/store/user/jpata/mlpf/tfds/cld/granularity_scan_2025-05-29/<scan-point>/<dataset>/<configuration>/<version>/
```

For ordinary campaigns, the campaign directory is a TFDS data root: it directly
contains dataset-name directories and can be used as `--data-dir` after download.
Keep a campaign name stable across all its datasets and configurations. The legacy
`mlpf/tensorflow_datasets/2.8.0/` tree is separate from this layout.

On 6 November 2026, the Tallinn CLD campaign
`cld/v1.2.5_key4hep_2025-05-29` was uploaded at version `3.3.0` for
`cld_edm_{ttbar,ww_fullhad,qq,zz}_pf`, configurations 1–10. All 930 files
(33,418,452,866 bytes) passed remote size and SHA-256 readback checks.

The matching CLIC campaign `clic/v1.2.5_key4hep_2025-05-29` was uploaded
at version `3.3.0` for `clic_edm_{ttbar,ww_fullhad,qq}_pf`, configurations
1–10. All 798 files (28,596,808,007 bytes) passed remote size and SHA-256
readback checks.

The CLD granularity scan `cld/granularity_scan_2025-05-29` was uploaded on
6 November 2026. It contains six `ecal_*mm` points with `cld_edm_z_tautau_pf`
and `cld_edm_z_tautau_hits`, and six `hcal_*mm` points with
`cld_edm_ttbar_pf` and `cld_edm_ttbar_hits`. Every point has configurations
1–10. PF uses version `3.3.0`; hits uses version `3.2.2`. All 2,258 files
(22,917,831,506 bytes) passed remote size and SHA-256 readback checks. The
remote top-level listing contains all 12 expected point directories.

The IDEA campaign `idea/IDEA_o1_v03_fccconfig_a05a3a9` was uploaded on
6 November 2026. It contains `idea_edm_{ttbar,ww_fullhad,qq}_pf`, configuration
`1`, at both versions `0.2.0` and `0.2.1`. All 498 files
(12,994,250,102 bytes) passed remote size and SHA-256 readback checks.
The two local versions have byte-identical shards and feature metadata;
only `dataset_info.json` records a different version. See the
[IDEA dataset note](idea.md#builder-and-recipe-compatibility) before choosing
between them.

## Upload a campaign

Use `scripts/tfds_webdav_upload.py`. It sends individual files in parallel,
retries failed transfers, checks the WebDAV-reported size, then downloads each
file as a stream and compares its SHA-256 digest with the source. Rerunning the
same command verifies existing files and uploads only missing or different-size
files. The readback adds one full download of the selected data, so allow for
both upload and download traffic.

For very large collections, `--verify-mode md5-server` compares the local MD5
with the endpoint's `Want-Digest: md5` response and its reported file size.
This verifies stored bytes without a full readback and requires host `curl`
with TLS trust for the endpoint. The default `sha256-readback` mode remains
available for smaller collections. Both modes verify final directory sizes.
Use `--progress-every 1000` to keep large transfer logs compact. Interactive
terminals show a progress bar with verified bytes, transfer rate, ETA, and error
count. Every run also writes atomic JSON status to
`~/.local/state/mlpf-tfds-webdav/` (or `$XDG_STATE_HOME/mlpf-tfds-webdav/`)
and prints the exact path at startup. Use `--status-file /path/to/job.json` for
a predictable path, especially in batch jobs. The JSON contains both verified
and newly transferred file/byte counts, so an agent can distinguish a resumed
job from new network traffic. It contains no proxy credentials.

Read the latest job from another terminal or agent session:

```bash
python3 scripts/tfds_webdav_status.py
python3 scripts/tfds_webdav_status.py --watch
python3 scripts/tfds_webdav_status.py /path/to/job.json --json
```

The status phases are `preparing`, `transferring`, `verifying`, `complete`, and
`failed`. Status remains available after the transfer process exits. A final
`complete` phase means all selected files passed the copy command's final
verification; `failed` means the job needs attention or a retry.

For the Tallinn CLD track/cluster campaign:

```bash
python3 scripts/tfds_webdav_upload.py \
  --local-root /local/joosep/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds \
  --detector cld --campaign v1.2.5_key4hep_2025-05-29 \
  --dataset cld_edm_ttbar_pf --dataset cld_edm_ww_fullhad_pf \
  --dataset cld_edm_qq_pf --dataset cld_edm_zz_pf \
  --version 3.3.0 --jobs 8 \
  --image /home/joosep/singularity/davix.sif --dry-run
```

Remove `--dry-run` to transfer. Use `--config 1` for a single configuration,
repeating `--config` to select more. The default selects every configuration
that has the chosen version and `dataset_info.json`.

The IDEA production has three datasets at configuration `1` in both `0.2.0`
and `0.2.1`. Use the uploader once per version, with `--detector idea`,
`--campaign IDEA_o1_v03_fccconfig_a05a3a9`, and repeat `--dataset` for
`idea_edm_ttbar_pf`, `idea_edm_ww_fullhad_pf`, and `idea_edm_qq_pf`.

The main CLD and CLIC hit productions each have `ttbar`, `ww_fullhad`, and
`qq` datasets at version `3.2.2`, configurations 1–10. Use the same
campaign names as their track/cluster datasets and select all three hit
dataset names for each detector. `cld_edm_zz_hits` is configured as an
additional builder but is absent from this production workspace.

For all 12 CLD granularity scan points, use the scan wrapper. It checks that
each point has the expected PF and hits datasets and all ten configurations
before creating remote directories. It uploads the PF version `3.3.0` and the
hits version `3.2.2` into separate TFDS trees:

```bash
python3 scripts/upload_granularity_scan.py \
  --image /home/joosep/singularity/davix.sif --jobs 8 --dry-run
```

Remove `--dry-run` to transfer. Use `--point ecal_1p5mm` to retry or inspect
one point. Individual uploads can use `tfds_webdav_upload.py --variant
ecal_1p5mm` with `--campaign granularity_scan_2025-05-29`.

## Download one version

On a receiving site, use the same campaign name. The destination directory is
the TFDS data root for that campaign:

```bash
python3 scripts/tfds_webdav.py download cld v1.2.5_key4hep_2025-05-29 \
  cld_edm_ttbar_pf 1 3.3.0 \
  --local-root /path/to/tfds/cld/v1.2.5_key4hep_2025-05-29 \
  --image /path/to/davix.sif --dry-run
```

Remove `--dry-run` to copy. The download command uses eight Davix
threads, refuses a nonempty destination, and shows progress based on received
files and bytes. It checks all downloaded file sizes against the remote listing.
Validate the downloaded version with
the [TFDS metadata and one-event checks](download.md#verify-metadata-and-one-event).
For a scan point, add `--variant ecal_1p5mm` and set `--local-root` to a
point-specific destination such as `/path/to/tfds/cld/granularity_scan_2025-05-29/ecal_1p5mm`.
The upload command's SHA-256 readback verifies the bytes on the endpoint at
publication time; receiving sites should perform their own integrity check if
they need end-to-end confirmation.

## Download a complete data root

Use `scripts/tfds_webdav_tree.py` for a WebDAV directory containing multiple
datasets and configurations. It inventories the remote tree, downloads in
parallel, and reports file/byte progress in the terminal and
the same atomic JSON status format as the other transfer commands. The local
destination must be empty for the initial recursive download. If a large
transfer is interrupted, rerun the same command with `--resume`: it checks
existing file sizes and downloads only missing or wrong-size files in parallel.
After Davix exits, it checks every downloaded file against the remote listing's
size. This is a size check, not a checksum check.

To download selected datasets from a campaign root, repeat `--dataset`. The
script lists the remote campaign once and copies only the selected dataset
directories. Use `--resume` when the local TFDS root already contains data.
For the three main CLD hit datasets:

```bash
python3 scripts/tfds_webdav_tree.py \
  --remote-url davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tfds/cld/v1.2.5_key4hep_2025-05-29/ \
  --local-root /mnt/ceph/users/jpata/mlpf/cld/v1.2.5_key4hep_2025-05-29/tfds \
  --dataset cld_edm_ttbar_hits --dataset cld_edm_ww_fullhad_hits \
  --dataset cld_edm_qq_hits --image ./davix.sif --jobs 8 --resume
```

For the legacy CMS TFDS root on another site:

```bash
python3 scripts/tfds_webdav_tree.py \
  --remote-url davs://xrootd.hep.kbfi.ee:1094/store/user/jpata/mlpf/tensorflow_datasets/2.8.0/ \
  --local-root /path/to/tfds --image /path/to/davix.sif --jobs 16
```

The local root then contains `cms_pf_ttbar/1/2.8.0/` and the other CMS
dataset directories. Set `$X509_USER_PROXY` or pass `--proxy` if the site's
proxy is not at `~/x509-cms-2048`. The legacy tree has 78,148 files and includes
six `.config/metadata.json` files; the tree downloader includes them.

The upload and download scripts use `$X509_USER_PROXY`, falling back to `~/x509-cms-2048`, and
mount it read-only at `/tmp/x509` in Apptainer. Supply `--proxy`, `--image`,
`--capath`, or `--remote-root` for other sites. The Davix image is available at
<https://jpata.web.cern.ch/jpata/davix.sif> if it is not already installed.
The image must contain the Grid CA directory at
`/etc/grid-security/certificates`.

## Why direct Davix copying

[DVC supports WebDAV remotes and client certificate/key paths](https://doc.dvc.org/user-guide/data-management/remote-storage/webdav),
but its `push`/`pull` workflow adds a content-addressed cache and Git-tracked
DVC metadata. The production campaign and TFDS version already identify these
datasets. Direct Davix copying keeps the remote tree browsable and puts the
files at the paths users need for TFDS.
