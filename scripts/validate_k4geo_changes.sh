#!/usr/bin/env bash
# Validate the HEP-KBFI k4geo and k4RecCalorimeter PRs, plus the upstream
# keyed-hit lookup fix, without modifying any checkout.
#
# All clones, source exports, builds, installs, simulation outputs, and reports
# live in a mktemp directory that is removed on exit unless --keep-workdir is
# explicitly requested. Builds and simulations are deliberately sequential.
set -Eeuo pipefail

readonly DEFAULT_REPOSITORY_URL="https://github.com/HEP-KBFI/k4geo.git"
readonly DEFAULT_RECCALO_REPOSITORY_URL="https://github.com/HEP-KBFI/k4RecCalorimeter.git"
readonly DEFAULT_KEY4HEP_RELEASE="2026-04-08"
validation_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly validation_script_dir

repository=""
repository_url="${DEFAULT_REPOSITORY_URL}"
reccalo_repository=""
reccalo_repository_url="${DEFAULT_RECCALO_REPOSITORY_URL}"
key4hep_release="${DEFAULT_KEY4HEP_RELEASE}"
main_ref="main"
truth_ref="idea-cherenkov-truth"
lookup_ref="idea-drc-map-lookups"
combined_ref="idea-mlpf-integration"
keyed_lookup_fix_ref="a9d5b725"
reccalo_main_ref="main"
reccalo_truth_ref="idea-cellid-truth-links"
reccalo_optical_ref="idea-optical-hit-links"
reccalo_cluster_ref="idea-dual-readout-clusters"
reccalo_combined_ref="idea-mlpf-integration"
reccalo_metadata_compat="auto"
build_jobs=2
events=10
speedup_events=3
speedup_energy_gev=50
qq_events=5
qq_seed=424242
minimum_speedup_factor=10
performance_repetitions=3
performance_tolerance_percent=15
truth_tolerance_percent=25
include_pions=1
keep_workdir=0
workdir=""
current_log=""
workflow_root="${validation_script_dir}/../mlpf/data/key4hep/gen"
allow_dirty_workflow_root=0

usage() {
  cat <<'EOF'
Usage: validate_k4geo_changes.sh [OPTIONS]

Sequentially build and validate HEP-KBFI's focused k4geo and
k4RecCalorimeter changes against CLIC, CLD, and IDEA. A controlled
current-minus-fix source export also reproduces the pre-#620 quadratic
calorimeter-hit lookup for physics-equivalence and speedup validation. Fixed-
seed 365 GeV qq events are generated once, simulated with the k4geo variants,
then reconstructed with baseline, three focused k4RecCalorimeter branches,
their combined branch, and reversed dual-readout input order. The source
repositories and caller's environment are never modified.
Temporary data are deleted on exit by default.

Options:
  --repo PATH                 Read refs from an existing local k4geo clone.
                              No fetch, checkout, or worktree is performed.
  --repository-url URL        Clone this repository into the temporary area
                              when --repo is not given.
  --reccalo-repo PATH         Read refs from an existing local
                              k4RecCalorimeter clone without modifying it.
  --reccalo-repository-url URL
                              Clone this k4RecCalorimeter repository when
                              --reccalo-repo is not given.
  --release NAME              Key4HEP release (default: 2026-04-08).
  --main-ref REF              Baseline ref (default: main).
  --truth-ref REF             Cherenkov-truth PR ref.
  --lookup-ref REF            dense-map lookup PR ref.
  --combined-ref REF          branch containing both focused changes.
  --keyed-lookup-fix-ref REF  keyed calorimeter-hit lookup commit used to
                              synthesize the pre-fix variant (default: a9d5b725).
  --reccalo-main-ref REF      k4RecCalorimeter baseline ref (default: main).
  --reccalo-truth-ref REF     cellID truth-link PR ref.
  --reccalo-optical-ref REF   optical digi/sim-link PR ref.
  --reccalo-cluster-ref REF   dual-readout cluster PR ref.
  --reccalo-combined-ref REF  branch containing all three reconstruction PRs.
  --reccalo-metadata-compat   apply the recorded April-2026 metadata API shim.
  --no-reccalo-metadata-compat
                              require the exact refs to build without the shim.
  --build-jobs N              Maximum parallel compile jobs (default: 2).
  --events N                  Events in every validation scenario (default: 10).
  --speedup-events N          50 GeV pi- events in the pre-fix/current benchmark
                              (default: 3).
  --speedup-energy N          Particle-gun energy in GeV for that benchmark
                              (default: 50).
  --qq-events N               Events in the qq simulation and reconstruction
                              validation (default: 5).
  --qq-seed N                 Generator and DDSim seed for qq (default: 424242).
  --workflow-root PATH        key4hep-sim gen directory containing CLDConfig
                              and the IDEA integration (default: the checkout
                              adjacent to this script).
  --allow-dirty-workflow-root permit a dirty workflow root for development
                              smoke tests; forbidden by default.
  --minimum-speedup N         Required processing-time speedup (default: 10).
  --performance-repetitions N Repetitions for e- timing cases (default: 3).
  --performance-tolerance PCT CPU/RSS regression gate (default: 15).
  --truth-tolerance PCT       IDEA truth-overhead gate (default: 25).
  --no-pions                  Skip the additional pi- cases.
  --quick                     One event, one speedup event, one qq event, one
                              repetition, and no additional pion cases.
                              Physics checks remain active; statistical
                              performance gates are reported but not enforced.
  --keep-workdir              Keep the temporary directory and print its path.
  -h, --help                  Show this help.

Default refs in HEP-KBFI/k4geo:
  main                      official-main baseline in the fork
  idea-cherenkov-truth      PR #1, Cherenkov ancestry
  idea-drc-map-lookups      PR #2, duplicate map lookup removal
  idea-mlpf-integration     both changes combined
  a9d5b725                  upstream PR #620, keyed hit lookups

Default refs in HEP-KBFI/k4RecCalorimeter:
  main                         official-main baseline in the fork
  idea-cellid-truth-links      PR #1, indexed cellID truth links
  idea-optical-hit-links       PR #2, optical digi/sim links
  idea-dual-readout-clusters   PR #3, channel-preserving clusters
  idea-mlpf-integration        all three changes combined

The script requires CVMFS, git, cmake, a C++ compiler, a clean key4hep-sim
workflow root, and enough temporary space for ten component builds plus
sequential detector outputs. The April-2026 stack lacks the metadata API used
by current k4RecCalorimeter main, so a narrowly scoped, recorded compatibility
shim is enabled automatically for release 2026-04-08. It never invokes Slurm,
Snakemake, git fetch on a supplied clone, or a remote write operation.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  if [[ -n "${current_log}" && -f "${current_log}" ]]; then
    printf '\nLast 80 lines of %s:\n' "${current_log}" >&2
    tail -n 80 "${current_log}" >&2 || true
  fi
  exit 1
}

require_positive_integer() {
  local label=$1
  local value=$2
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${label} must be a positive integer, got '${value}'"
}

while (($#)); do
  case "$1" in
    --repo)
      (($# >= 2)) || die "--repo requires a path"
      repository=$2
      shift 2
      ;;
    --repository-url)
      (($# >= 2)) || die "--repository-url requires a URL"
      repository_url=$2
      shift 2
      ;;
    --reccalo-repo)
      (($# >= 2)) || die "--reccalo-repo requires a path"
      reccalo_repository=$2
      shift 2
      ;;
    --reccalo-repository-url)
      (($# >= 2)) || die "--reccalo-repository-url requires a URL"
      reccalo_repository_url=$2
      shift 2
      ;;
    --release)
      (($# >= 2)) || die "--release requires a value"
      key4hep_release=$2
      shift 2
      ;;
    --main-ref)
      (($# >= 2)) || die "--main-ref requires a ref"
      main_ref=$2
      shift 2
      ;;
    --truth-ref)
      (($# >= 2)) || die "--truth-ref requires a ref"
      truth_ref=$2
      shift 2
      ;;
    --lookup-ref)
      (($# >= 2)) || die "--lookup-ref requires a ref"
      lookup_ref=$2
      shift 2
      ;;
    --combined-ref)
      (($# >= 2)) || die "--combined-ref requires a ref"
      combined_ref=$2
      shift 2
      ;;
    --keyed-lookup-fix-ref)
      (($# >= 2)) || die "--keyed-lookup-fix-ref requires a ref"
      keyed_lookup_fix_ref=$2
      shift 2
      ;;
    --reccalo-main-ref)
      (($# >= 2)) || die "--reccalo-main-ref requires a ref"
      reccalo_main_ref=$2
      shift 2
      ;;
    --reccalo-truth-ref)
      (($# >= 2)) || die "--reccalo-truth-ref requires a ref"
      reccalo_truth_ref=$2
      shift 2
      ;;
    --reccalo-optical-ref)
      (($# >= 2)) || die "--reccalo-optical-ref requires a ref"
      reccalo_optical_ref=$2
      shift 2
      ;;
    --reccalo-cluster-ref)
      (($# >= 2)) || die "--reccalo-cluster-ref requires a ref"
      reccalo_cluster_ref=$2
      shift 2
      ;;
    --reccalo-combined-ref)
      (($# >= 2)) || die "--reccalo-combined-ref requires a ref"
      reccalo_combined_ref=$2
      shift 2
      ;;
    --reccalo-metadata-compat)
      reccalo_metadata_compat=1
      shift
      ;;
    --no-reccalo-metadata-compat)
      reccalo_metadata_compat=0
      shift
      ;;
    --build-jobs)
      (($# >= 2)) || die "--build-jobs requires a value"
      build_jobs=$2
      shift 2
      ;;
    --events)
      (($# >= 2)) || die "--events requires a value"
      events=$2
      shift 2
      ;;
    --speedup-events)
      (($# >= 2)) || die "--speedup-events requires a value"
      speedup_events=$2
      shift 2
      ;;
    --speedup-energy)
      (($# >= 2)) || die "--speedup-energy requires a value"
      speedup_energy_gev=$2
      shift 2
      ;;
    --qq-events)
      (($# >= 2)) || die "--qq-events requires a value"
      qq_events=$2
      shift 2
      ;;
    --qq-seed)
      (($# >= 2)) || die "--qq-seed requires a value"
      qq_seed=$2
      shift 2
      ;;
    --workflow-root)
      (($# >= 2)) || die "--workflow-root requires a path"
      workflow_root=$2
      shift 2
      ;;
    --allow-dirty-workflow-root)
      allow_dirty_workflow_root=1
      shift
      ;;
    --minimum-speedup)
      (($# >= 2)) || die "--minimum-speedup requires a value"
      minimum_speedup_factor=$2
      shift 2
      ;;
    --performance-repetitions)
      (($# >= 2)) || die "--performance-repetitions requires a value"
      performance_repetitions=$2
      shift 2
      ;;
    --performance-tolerance)
      (($# >= 2)) || die "--performance-tolerance requires a value"
      performance_tolerance_percent=$2
      shift 2
      ;;
    --truth-tolerance)
      (($# >= 2)) || die "--truth-tolerance requires a value"
      truth_tolerance_percent=$2
      shift 2
      ;;
    --no-pions)
      include_pions=0
      shift
      ;;
    --quick)
      events=1
      speedup_events=1
      qq_events=1
      performance_repetitions=1
      include_pions=0
      shift
      ;;
    --keep-workdir)
      keep_workdir=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

require_positive_integer "--build-jobs" "${build_jobs}"
require_positive_integer "--events" "${events}"
require_positive_integer "--speedup-events" "${speedup_events}"
require_positive_integer "--speedup-energy" "${speedup_energy_gev}"
require_positive_integer "--qq-events" "${qq_events}"
require_positive_integer "--qq-seed" "${qq_seed}"
require_positive_integer "--minimum-speedup" "${minimum_speedup_factor}"
require_positive_integer "--performance-repetitions" "${performance_repetitions}"
require_positive_integer "--performance-tolerance" "${performance_tolerance_percent}"
require_positive_integer "--truth-tolerance" "${truth_tolerance_percent}"

if [[ "${reccalo_metadata_compat}" == auto ]]; then
  if [[ "${key4hep_release}" == "2026-04-08" ]]; then
    reccalo_metadata_compat=1
  else
    reccalo_metadata_compat=0
  fi
fi

for command_name in git cmake tar mktemp awk tail realpath stat dirname basename wget; do
  command -v "${command_name}" >/dev/null 2>&1 || die "missing required command: ${command_name}"
done

setup_script="/cvmfs/sw.hsf.org/key4hep/setup.sh"
[[ -r "${setup_script}" ]] || die "Key4HEP setup script is not readable: ${setup_script}"

workdir=$(mktemp -d "${TMPDIR:-/tmp}/key4hep-component-validation.XXXXXXXX")
[[ -n "${workdir}" && -d "${workdir}" ]] || die "failed to create temporary directory"

cleanup() {
  local status=$?
  if ((keep_workdir)); then
    printf '\nTemporary validation data kept at: %s\n' "${workdir}" >&2
  elif [[ "${workdir}" == "${TMPDIR:-/tmp}"/key4hep-component-validation.* && -d "${workdir}" ]]; then
    rm -rf -- "${workdir}"
  fi
  exit "${status}"
}
trap cleanup EXIT

# The Key4HEP setup script probes optional unset variables and is not nounset-safe.
set +u
source "${setup_script}" -r "${key4hep_release}"
set -u

for command_name in ddsim k4run k4_local_repo python3 c++ podio-dump; do
  command -v "${command_name}" >/dev/null 2>&1 || die "${command_name} is unavailable after loading Key4HEP ${key4hep_release}"
done

[[ -d "${workflow_root}" ]] || die "workflow root does not exist: ${workflow_root}"
workflow_root=$(realpath "${workflow_root}")
readonly workflow_root
readonly cld_config_dir="${workflow_root}/cld/CLDConfig"
readonly idea_integration_dir="${workflow_root}/idea"
readonly fcc_idea_dir="${idea_integration_dir}/FCC-config/FCCee/FullSim/IDEA/IDEA_o1_v03"
readonly pythia_qq_card="${cld_config_dir}/pythia/p8_ee_qq_ecm365.cmd"

[[ -r "${cld_config_dir}/pythia.py" ]] || die "missing CLD Pythia configuration under ${cld_config_dir}"
[[ -r "${pythia_qq_card}" ]] || die "missing qq Pythia card: ${pythia_qq_card}"
[[ -r "${fcc_idea_dir}/run_digi_reco.py" ]] || die "missing IDEA reconstruction steering under ${fcc_idea_dir}"

workflow_refs="${workdir}/workflow_refs.tsv"
printf 'path\tcommit\tstatus\n' >"${workflow_refs}"
if git -C "${workflow_root}" rev-parse --git-dir >/dev/null 2>&1; then
  workflow_status=$(git -C "${workflow_root}" status --porcelain --untracked-files=normal)
  if [[ -n "${workflow_status}" && "${allow_dirty_workflow_root}" -eq 0 ]]; then
    die "workflow root is dirty; use a fresh key4hep-sim clone or --allow-dirty-workflow-root for a non-acceptance smoke test"
  fi
  printf '.\t%s\t%s\n' "$(git -C "${workflow_root}" rev-parse HEAD)" "$(if [[ -n "${workflow_status}" ]]; then printf dirty; else printf clean; fi)" >>"${workflow_refs}"
  for dependency in cld/CLDConfig idea/FCC-config; do
    [[ -e "${workflow_root}/${dependency}" ]] || die "missing workflow dependency: ${dependency}"
    dependency_status=$(git -C "${workflow_root}/${dependency}" status --porcelain --untracked-files=normal)
    if [[ -n "${dependency_status}" && "${allow_dirty_workflow_root}" -eq 0 ]]; then
      die "workflow dependency ${dependency} is dirty"
    fi
    printf '%s\t%s\t%s\n' "${dependency}" "$(git -C "${workflow_root}/${dependency}" rev-parse HEAD)" \
      "$(if [[ -n "${dependency_status}" ]]; then printf dirty; else printf clean; fi)" >>"${workflow_refs}"
  done
else
  ((allow_dirty_workflow_root)) || die "workflow root is not a git checkout; pass --allow-dirty-workflow-root only for development"
  printf '.\tunknown\tnon-git\n' >>"${workflow_refs}"
fi
printf 'Workflow refs:\n'
column -t -s $'\t' "${workflow_refs}" 2>/dev/null || cat "${workflow_refs}"

export LC_ALL=C
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TBB_NUM_THREADS=1

if [[ -n "${repository}" ]]; then
  repository=$(realpath "${repository}")
  git -C "${repository}" rev-parse --git-dir >/dev/null 2>&1 || die "not a git repository: ${repository}"
  printf 'Using local repository read-only: %s\n' "${repository}"
else
  repository="${workdir}/repository"
  current_log="${workdir}/clone.log"
  printf 'Cloning %s into the disposable workspace...\n' "${repository_url}"
  git clone --quiet "${repository_url}" "${repository}" >"${current_log}" 2>&1 || die "git clone failed"
  current_log=""
fi

resolve_ref_in() {
  local target_repository=$1
  local requested=$2
  if git -C "${target_repository}" rev-parse --verify --quiet "${requested}^{commit}" >/dev/null; then
    git -C "${target_repository}" rev-parse "${requested}^{commit}"
  elif git -C "${target_repository}" rev-parse --verify --quiet "origin/${requested}^{commit}" >/dev/null; then
    git -C "${target_repository}" rev-parse "origin/${requested}^{commit}"
  else
    die "cannot resolve ref '${requested}' in ${target_repository}"
  fi
}

declare -A variant_sha
declare -A source_dir
declare -A install_dir
variant_sha[main]=$(resolve_ref_in "${repository}" "${main_ref}")
variant_sha[truth]=$(resolve_ref_in "${repository}" "${truth_ref}")
variant_sha[lookup]=$(resolve_ref_in "${repository}" "${lookup_ref}")
variant_sha[combined]=$(resolve_ref_in "${repository}" "${combined_ref}")
keyed_lookup_fix_sha=$(resolve_ref_in "${repository}" "${keyed_lookup_fix_ref}")
keyed_lookup_parent_sha=$(git -C "${repository}" rev-parse "${keyed_lookup_fix_sha}^")
# pre_fix is current main with only PR #620's two source files restored from
# its parent. This isolates the lookup algorithm from later unrelated changes.
variant_sha[pre_fix]="${variant_sha[main]}"
readonly -a variants=(main truth lookup combined)
readonly -a build_variants=(main truth lookup combined pre_fix)
readonly -a keyed_lookup_files=(
  plugins/CaloPreShowerSDAction.cpp
  plugins/FiberDRCaloSDAction.cpp
)

for variant in "${variants[@]}"; do
  printf '%-9s %s\n' "${variant}" "${variant_sha[${variant}]}"
done
printf '%-9s %s (main with %s reversed)\n' pre_fix \
  "${variant_sha[pre_fix]}" "${keyed_lookup_fix_sha}"

git -C "${repository}" merge-base --is-ancestor \
  "${keyed_lookup_fix_sha}" "${variant_sha[main]}" || \
  die "keyed lookup fix ${keyed_lookup_fix_sha} is not an ancestor of main"
changed_by_keyed_fix=$(git -C "${repository}" diff --name-only \
  "${keyed_lookup_parent_sha}" "${keyed_lookup_fix_sha}" -- plugins)
expected_keyed_fix_paths=$(printf '%s\n' "${keyed_lookup_files[@]}")
[[ "${changed_by_keyed_fix}" == "${expected_keyed_fix_paths}" ]] || \
  die "unexpected keyed-lookup fix scope: ${changed_by_keyed_fix}"
for path in "${keyed_lookup_files[@]}"; do
  git -C "${repository}" diff --quiet \
    "${keyed_lookup_fix_sha}" "${variant_sha[main]}" -- "${path}" || \
    die "${path} changed after keyed lookup fix; cannot synthesize an isolated pre-fix variant"
done
keyed_lookup_count=$(git -C "${repository}" show \
  "${keyed_lookup_fix_sha}:plugins/FiberDRCaloSDAction.cpp" | \
  awk 'index($0, "findByKey") {count++} END {print count+0}')
pre_fix_linear_count=$(git -C "${repository}" show \
  "${keyed_lookup_parent_sha}:plugins/FiberDRCaloSDAction.cpp" | \
  awk 'index($0, "CellIDCompare") {count++} END {print count+0}')
((keyed_lookup_count >= 3 && pre_fix_linear_count >= 3)) || \
  die "keyed/pre-fix lookup signatures were not found in FiberDRCaloSDAction.cpp"

[[ "${variant_sha[main]}" != "${variant_sha[truth]}" ]] || die "truth ref resolves to main"
[[ "${variant_sha[main]}" != "${variant_sha[lookup]}" ]] || die "lookup ref resolves to main"

check_patch_scope() {
  local variant=$1
  local expected actual
  case "${variant}" in
    truth|combined)
      expected=$(printf '%s\n' \
        plugins/FiberDRCaloSDAction.cpp \
        plugins/FiberDRCaloSDAction.h \
        plugins/Geant4Output2EDM4hep_DRC.cpp | sort)
      ;;
    lookup)
      expected=plugins/Geant4Output2EDM4hep_DRC.cpp
      ;;
    *) die "unknown k4geo patch-scope variant: ${variant}" ;;
  esac
  actual=$(git -C "${repository}" diff --name-only \
    "${variant_sha[main]}...${variant_sha[${variant}]}" | sort) || \
    die "cannot determine k4geo ${variant} patch scope; use a non-shallow clone"
  [[ "${actual}" == "${expected}" ]] || \
    die "unexpected k4geo ${variant} scope; expected '${expected}', observed '${actual}'"
}

check_patch_scope truth
check_patch_scope lookup
check_patch_scope combined

if [[ -n "${reccalo_repository}" ]]; then
  reccalo_repository=$(realpath "${reccalo_repository}")
  git -C "${reccalo_repository}" rev-parse --git-dir >/dev/null 2>&1 || \
    die "not a git repository: ${reccalo_repository}"
  printf 'Using local k4RecCalorimeter repository read-only: %s\n' "${reccalo_repository}"
else
  reccalo_repository="${workdir}/k4RecCalorimeter-repository"
  current_log="${workdir}/reccalo-clone.log"
  printf 'Cloning %s into the disposable workspace...\n' "${reccalo_repository_url}"
  git clone --quiet "${reccalo_repository_url}" "${reccalo_repository}" >"${current_log}" 2>&1 || \
    die "k4RecCalorimeter clone failed"
  current_log=""
fi

declare -A reccalo_variant_sha
declare -A reccalo_source_dir
declare -A reccalo_install_dir
declare -A reccalo_plugin_dir
reccalo_variant_sha[main]=$(resolve_ref_in "${reccalo_repository}" "${reccalo_main_ref}")
reccalo_variant_sha[truth]=$(resolve_ref_in "${reccalo_repository}" "${reccalo_truth_ref}")
reccalo_variant_sha[optical]=$(resolve_ref_in "${reccalo_repository}" "${reccalo_optical_ref}")
reccalo_variant_sha[cluster]=$(resolve_ref_in "${reccalo_repository}" "${reccalo_cluster_ref}")
reccalo_variant_sha[combined]=$(resolve_ref_in "${reccalo_repository}" "${reccalo_combined_ref}")
readonly -a reccalo_variants=(main truth optical cluster combined)

printf '\nk4RecCalorimeter refs:\n'
for variant in "${reccalo_variants[@]}"; do
  printf '%-9s %s\n' "${variant}" "${reccalo_variant_sha[${variant}]}"
done

readonly -a reccalo_truth_files=(
  RecCalorimeter/src/components/CreateTruthLinks.cpp
  RecCalorimeter/src/components/CreateTruthLinks.h
)
readonly -a reccalo_optical_files=(
  RecCalorimeter/src/components/SimulateSiPMwithOpticalPhoton.cpp
  RecCalorimeter/src/components/SimulateSiPMwithOpticalPhoton.h
)
readonly -a reccalo_cluster_files=(
  RecFCCeeCalorimeter/src/components/CaloTopoClusterFCCee.cpp
  RecFCCeeCalorimeter/src/components/CaloTopoClusterFCCee.h
)

check_reccalo_patch_scope() {
  local variant=$1
  shift
  local expected actual
  expected=$(printf '%s\n' "$@" | sort)
  actual=$(git -C "${reccalo_repository}" diff --name-only \
    "${reccalo_variant_sha[main]}...${reccalo_variant_sha[${variant}]}" | sort) || \
    die "cannot determine k4RecCalorimeter ${variant} patch scope; use a non-shallow clone"
  [[ "${actual}" == "${expected}" ]] || \
    die "unexpected k4RecCalorimeter ${variant} scope; expected '${expected}', observed '${actual}'"
}

check_reccalo_patch_scope truth "${reccalo_truth_files[@]}"
check_reccalo_patch_scope optical "${reccalo_optical_files[@]}"
check_reccalo_patch_scope cluster "${reccalo_cluster_files[@]}"
check_reccalo_patch_scope combined \
  "${reccalo_truth_files[@]}" "${reccalo_optical_files[@]}" "${reccalo_cluster_files[@]}"

for path in "${reccalo_truth_files[@]}"; do
  git -C "${reccalo_repository}" diff --quiet \
    "${reccalo_variant_sha[truth]}" "${reccalo_variant_sha[combined]}" -- "${path}" || \
    die "combined k4RecCalorimeter branch does not preserve truth-link file ${path}"
done
for path in "${reccalo_optical_files[@]}"; do
  git -C "${reccalo_repository}" diff --quiet \
    "${reccalo_variant_sha[optical]}" "${reccalo_variant_sha[combined]}" -- "${path}" || \
    die "combined k4RecCalorimeter branch does not preserve optical-link file ${path}"
done
for path in "${reccalo_cluster_files[@]}"; do
  git -C "${reccalo_repository}" diff --quiet \
    "${reccalo_variant_sha[cluster]}" "${reccalo_variant_sha[combined]}" -- "${path}" || \
    die "combined k4RecCalorimeter branch does not preserve cluster file ${path}"
done

component_refs="${workdir}/component_refs.tsv"
printf 'repository\tvariant\tcommit\n' >"${component_refs}"
for variant in "${variants[@]}"; do
  printf 'k4geo\t%s\t%s\n' "${variant}" "${variant_sha[${variant}]}" >>"${component_refs}"
done
for variant in "${reccalo_variants[@]}"; do
  printf 'k4RecCalorimeter\t%s\t%s\n' "${variant}" "${reccalo_variant_sha[${variant}]}" >>"${component_refs}"
done

build_variant() {
  local variant=$1
  local src="${workdir}/sources/${variant}"
  local build="${workdir}/build/${variant}"
  local install="${workdir}/install/${variant}"
  local log="${workdir}/build-${variant}.log"

  mkdir -p "${src}" "${build}" "${install}"
  git -C "${repository}" archive "${variant_sha[${variant}]}" | tar -x -C "${src}"
  if [[ "${variant}" == pre_fix ]]; then
    for path in "${keyed_lookup_files[@]}"; do
      git -C "${repository}" show "${keyed_lookup_parent_sha}:${path}" >"${src}/${path}" || \
        die "failed to restore pre-fix source for ${path}"
    done
  fi

  printf '\nBuilding %-9s with at most %d compile jobs...\n' "${variant}" "${build_jobs}"
  current_log="${log}"
  cmake -S "${src}" -B "${build}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="${install}" \
    -DBUILD_TESTING=OFF \
    -DK4GEO_USE_LCIO=OFF \
    -DINSTALL_COMPACT_FILES=OFF >"${log}" 2>&1 || die "CMake configuration failed for ${variant}"
  cmake --build "${build}" --parallel "${build_jobs}" >>"${log}" 2>&1 || die "build failed for ${variant}"
  cmake --install "${build}" >>"${log}" 2>&1 || die "install failed for ${variant}"
  [[ -s "${install}/lib/libk4geoG4.so" ]] || die "${variant} did not install libk4geoG4.so"

  source_dir[${variant}]="${src}"
  install_dir[${variant}]="${install}"
  current_log=""

  # The installed libraries and exported source geometry are sufficient for
  # validation. Removing this known temporary target controls disk usage.
  [[ "${build}" == "${workdir}"/build/* ]] || die "refusing to remove unexpected build path: ${build}"
  rm -rf -- "${build}"
}

for variant in "${build_variants[@]}"; do
  build_variant "${variant}"
done

apply_reccalo_metadata_compat() {
  local src=$1
  local variant=$2
  local patch_log="${workdir}/reccalo-metadata-compat-${variant}.patch"
  local header="${src}/RecFCCeeCalorimeter/src/components/CaloTopoClusterFCCee.h"
  local implementation="${src}/RecFCCeeCalorimeter/src/components/CaloTopoClusterFCCee.cpp"
  local rec_cmake="${src}/RecCalorimeter/CMakeLists.txt"
  local fccee_cmake="${src}/RecFCCeeCalorimeter/CMakeLists.txt"
  cp "${header}" "${header}.before-compat"
  cp "${implementation}" "${implementation}.before-compat"
  cp "${rec_cmake}" "${rec_cmake}.before-compat"
  cp "${fccee_cmake}" "${fccee_cmake}.before-compat"
  python3 - "${header}" "${implementation}" "${rec_cmake}" "${fccee_cmake}" <<'PY'
from pathlib import Path
import re
import sys

header_path, implementation_path, rec_cmake_path, fccee_cmake_path = map(Path, sys.argv[1:])
header = header_path.read_text()
implementation = implementation_path.read_text()

include_anchor = '#include "k4FWCore/DataHandle.h"\n'
if header.count(include_anchor) != 1:
    raise SystemExit("metadata compatibility shim: DataHandle include anchor mismatch")
header = header.replace(
    include_anchor,
    include_anchor + '#include "k4FWCore/MetaDataHandle.h"\n#include "edm4hep/Constants.h"\n',
)

property_anchor = "  Gaudi::Property<std::vector<int>> m_caloIDs{"
if header.count(property_anchor) != 1:
    raise SystemExit("metadata compatibility shim: calorimeter property anchor mismatch")
handles = """  // Compatibility handles for Key4HEP 2026-04-08, which predates
  // k4FWCore::putCollectionParameter. This changes metadata transport only.
  k4FWCore::MetaDataHandle<std::vector<int>> m_caloIDsMetaData{
      m_clusterCollection, "inputSystemIDs", Gaudi::DataHandle::Writer};
  k4FWCore::MetaDataHandle<std::vector<std::string>> m_cellsMetaData{
      m_clusterCollection, "inputCellCollections", Gaudi::DataHandle::Writer};
  k4FWCore::MetaDataHandle<std::vector<std::string>> m_shapeParametersHandle{
      m_clusterCollection, edm4hep::labels::ShapeParameterNames, Gaudi::DataHandle::Writer};

"""
header = header.replace(property_anchor, handles + property_anchor)

implementation = implementation.replace('#include "k4FWCore/MetadataUtils.h"\n\n', "")
shape_pattern = re.compile(
    r"  k4FWCore::putCollectionParameter\(m_clusterCollection\.objKey\(\), "
    r"edm4hep::labels::ShapeParameterNames,\s*shapeParameterNames, this\);"
)
implementation, shape_count = shape_pattern.subn("  m_shapeParametersHandle.put(shapeParameterNames);", implementation)
if shape_count != 1:
    raise SystemExit(f"metadata compatibility shim: replaced {shape_count} shape metadata calls")

system_pattern = re.compile(
    r"      k4FWCore::putCollectionParameter\(m_clusterCollection\.objKey\(\), "
    r'"inputSystemIDs", IDs, this\);'
)
implementation, system_count = system_pattern.subn("      m_caloIDsMetaData.put(IDs);", implementation)
cell_pattern = re.compile(
    r"      k4FWCore::putCollectionParameter\(m_clusterCollection\.objKey\(\), "
    r'"inputCellCollections", colls, this\);'
)
implementation, cell_count = cell_pattern.subn("      m_cellsMetaData.put(colls);", implementation)
if system_count != 1 or cell_count != 1:
    raise SystemExit(
        f"metadata compatibility shim: replaced system={system_count}, cell={cell_count} metadata calls"
    )

header_path.write_text(header)
implementation_path.write_text(implementation)

# Key4HEP 2026-04-08 cannot compile unrelated current-main components that use
# the newer k4FWCore metadata API. Build exactly the five components exercised
# by the IDEA reconstruction matrix, so the compatibility export remains
# focused and the PR implementation files remain otherwise unchanged.
component_sources = {
    rec_cmake_path: """set(_module_sources
  src/components/ConstNoiseTool.cpp
  src/components/CreateTruthLinks.cpp
  src/components/SimulateSiPMwithEdep.cpp
  src/components/SimulateSiPMwithOpticalPhoton.cpp
)
""",
    fccee_cmake_path: """set(_module_sources
  src/components/CaloTopoClusterFCCee.cpp
)
""",
}
for cmake_path, replacement in component_sources.items():
    cmake = cmake_path.read_text()
    cmake, count = re.subn(
        r"file\(GLOB _module_sources src/components/\*\.cpp ?\)\n",
        replacement,
        cmake,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"metadata compatibility shim: source-list anchor mismatch in {cmake_path}")
    cmake_path.write_text(cmake)
PY
  {
    diff -u "${header}.before-compat" "${header}" || true
    diff -u "${implementation}.before-compat" "${implementation}" || true
    diff -u "${rec_cmake}.before-compat" "${rec_cmake}" || true
    diff -u "${fccee_cmake}.before-compat" "${fccee_cmake}" || true
  } >"${patch_log}"
  rm -f -- \
    "${header}.before-compat" \
    "${implementation}.before-compat" \
    "${rec_cmake}.before-compat" \
    "${fccee_cmake}.before-compat"
  [[ -s "${patch_log}" ]] || die "metadata compatibility shim produced an empty patch for ${variant}"
}

build_reccalo_variant() {
  local variant=$1
  local src="${workdir}/reccalo-sources/${variant}"
  local build="${workdir}/reccalo-build/${variant}"
  local install="${workdir}/reccalo-install/${variant}"
  local log="${workdir}/reccalo-build-${variant}.log"

  mkdir -p "${src}" "${build}" "${install}"
  git -C "${reccalo_repository}" archive "${reccalo_variant_sha[${variant}]}" | tar -x -C "${src}"
  if ((reccalo_metadata_compat)); then
    apply_reccalo_metadata_compat "${src}" "${variant}"
  fi

  printf '\nBuilding k4RecCalorimeter %-9s with at most %d compile jobs...\n' "${variant}" "${build_jobs}"
  current_log="${log}"
  cmake -S "${src}" -B "${build}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_INSTALL_PREFIX="${install}" \
    -DBUILD_TESTING=OFF >"${log}" 2>&1 || die "k4RecCalorimeter CMake configuration failed for ${variant}"
  cmake --build "${build}" --parallel "${build_jobs}" >>"${log}" 2>&1 || \
    die "k4RecCalorimeter build failed for ${variant}"
  cmake --install "${build}" >>"${log}" 2>&1 || die "k4RecCalorimeter install failed for ${variant}"

  local plugin_dir=""
  for candidate in "${install}/lib64" "${install}/lib"; do
    if [[ -s "${candidate}/libk4RecCalorimeterPlugins.so" && \
          -s "${candidate}/libk4RecFCCeeCalorimeterPlugins.so" ]]; then
      plugin_dir="${candidate}"
      break
    fi
  done
  [[ -n "${plugin_dir}" ]] || die "${variant} did not install the required k4RecCalorimeter plugins"
  reccalo_source_dir[${variant}]="${src}"
  reccalo_install_dir[${variant}]="${install}"
  reccalo_plugin_dir[${variant}]="${plugin_dir}"
  current_log=""

  [[ "${build}" == "${workdir}"/reccalo-build/* ]] || die "refusing to remove unexpected build path: ${build}"
  rm -rf -- "${build}"
}

printf '\nk4RecCalorimeter metadata compatibility shim: %s\n' \
  "$(if ((reccalo_metadata_compat)); then printf enabled; else printf disabled; fi)"
for variant in "${reccalo_variants[@]}"; do
  build_reccalo_variant "${variant}"
done

prepare_fast_steering() {
  local variant=$1
  local input="${source_dir[${variant}]}/example/SteeringFile_IDEA_o1_v03.py"
  local output="${workdir}/steering/${variant}/SteeringFile_IDEA_o1_v03_fast.py"
  mkdir -p "$(dirname "${output}")"
  python3 - "${input}" "${output}" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
needle = "# SIM.physics.setupUserPhysics(setupDRCFastSim)"
if source.count(needle) != 1:
    raise SystemExit(f"expected exactly one disabled fast-simulation hook, found {source.count(needle)}")
Path(sys.argv[2]).write_text(source.replace(needle, "SIM.physics.setupUserPhysics(setupDRCFastSim)"))
PY
}

for variant in "${build_variants[@]}"; do
  prepare_fast_steering "${variant}"
done

prepare_reccalo_steering() {
  local variant=$1
  local reverse_inputs=${2:-0}
  local suffix=""
  ((reverse_inputs)) && suffix="-reversed"
  local input="${fcc_idea_dir}/run_digi_reco.py"
  local output="${workdir}/reccalo-steering/${variant}${suffix}/run_digi_reco.py"
  local expect_optical=0 expect_dual=0 expect_min_energy=0
  case "${variant}" in
    main) ;;
    truth) expect_min_energy=1 ;;
    optical) expect_optical=1 ;;
    cluster) expect_dual=1 ;;
    combined) expect_optical=1; expect_dual=1; expect_min_energy=1 ;;
    *) die "unknown k4RecCalorimeter steering variant: ${variant}" ;;
  esac
  mkdir -p "$(dirname "${output}")"
  python3 - "${input}" "${output}" "${expect_optical}" "${expect_dual}" \
    "${expect_min_energy}" "${reverse_inputs}" <<'PY'
from pathlib import Path
import re
import sys

input_path, output_path = map(Path, sys.argv[1:3])
expect_optical, expect_dual, expect_min_energy, reverse_inputs = map(int, sys.argv[3:])
source = input_path.read_text()

required_patterns = {
    "optical": r'(?m)^\s*outputHitLinkCollection\s*=\s*"DRcaloSiPMreadoutDigiHit_cheren_link",\s*$',
    "dual": r'(?m)^\s*inputEnergyNames\s*=\s*\["energy_cherenkov",\s*"energy_scintillation"\],\s*$',
    "minimum": r"(?m)^\s*minCellEnergy\s*=\s*0\.,\s*$",
    "truth_inputs": r'(?m)^\s*cell_hit_links\s*=\s*\["DRcaloSiPMreadoutDigiHit_scint_link",\s*"DRcaloSiPMreadoutDigiHit_cheren_link"\],\s*$',
    "cells": r'(?m)^\s*cells\s*=\s*\["DRcaloSiPMreadoutDigiHit","DRcaloSiPMreadoutDigiHit_scint"\],\s*$',
}
for label, pattern in required_patterns.items():
    if len(re.findall(pattern, source)) != 1:
        raise SystemExit(f"steering generation: expected one {label} property")

if not expect_optical:
    source = re.sub(required_patterns["optical"], "", source)
if not expect_dual:
    source = re.sub(required_patterns["dual"], "", source)
if not expect_min_energy:
    source = re.sub(required_patterns["minimum"], "", source)
if not expect_optical:
    source = re.sub(
        required_patterns["truth_inputs"],
        '    cell_hit_links=["DRcaloSiPMreadoutDigiHit_scint_link"],',
        source,
    )
if reverse_inputs:
    if not expect_dual:
        raise SystemExit("reversed inputs require the dual-readout cluster branch")
    source = re.sub(
        required_patterns["cells"],
        '    cells = ["DRcaloSiPMreadoutDigiHit_scint","DRcaloSiPMreadoutDigiHit"],',
        source,
    )
    source = re.sub(
        required_patterns["dual"],
        '    inputEnergyNames = ["energy_scintillation", "energy_cherenkov"],',
        source,
    )

Path(output_path).write_text(source)
PY
  printf '%s\n' "${output}"
}

declare -A reccalo_steering
declare -A reccalo_reversed_steering
for variant in "${reccalo_variants[@]}"; do
  reccalo_steering[${variant}]=$(prepare_reccalo_steering "${variant}" 0)
done
for variant in cluster combined; do
  reccalo_reversed_steering[${variant}]=$(prepare_reccalo_steering "${variant}" 1)
done

metrics_tsv="${workdir}/metrics.tsv"
printf 'variant\tcase\trepetition\tuser_s\tsystem_s\twall_s\tmax_rss_kb\toutput_bytes\tstartup_s\tprocessing_s\tprocessing_s_per_event\n' >"${metrics_tsv}"

run_measured() {
  local metrics_file=$1
  local log_file=$2
  shift 2
  python3 - "${metrics_file}" "${log_file}" "$@" <<'PY'
import json
import re
import resource
import subprocess
import sys
import time

metrics_path, log_path, *command = sys.argv[1:]
before = resource.getrusage(resource.RUSAGE_CHILDREN)
start = time.monotonic()
with open(log_path, "wb") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    next_report = start + 60.0
    loaded_k4geo_libraries = set()
    loaded_k4rec_libraries = set()
    while process.poll() is None:
        try:
            with open(f"/proc/{process.pid}/maps") as maps:
                for line in maps:
                    path = line.split()[-1] if "/" in line else ""
                    if "libk4geo" in path:
                        loaded_k4geo_libraries.add(path)
                    if "libk4Rec" in path:
                        loaded_k4rec_libraries.add(path)
        except FileNotFoundError:
            pass
        time.sleep(1.0)
        now = time.monotonic()
        if now >= next_report:
            print(f"  still running: pid={process.pid} elapsed={now - start:.0f}s", flush=True)
            next_report = now + 60.0
status = process.returncode
end = time.monotonic()
after = resource.getrusage(resource.RUSAGE_CHILDREN)
with open(log_path, errors="replace") as stream:
    log_text = stream.read()
summary_match = re.search(
    r"StartUp Time:\s*([0-9.]+) s, Processing and Init:\s*([0-9.]+) s "
    r"\(~([0-9.]+) s/Event\)",
    log_text,
)
startup_s = float(summary_match.group(1)) if summary_match else None
processing_s = float(summary_match.group(2)) if summary_match else None
processing_s_per_event = float(summary_match.group(3)) if summary_match else None
metrics = {
    "status": status,
    "user_s": after.ru_utime - before.ru_utime,
    "system_s": after.ru_stime - before.ru_stime,
    "wall_s": end - start,
    "max_rss_kb": after.ru_maxrss,
    "startup_s": startup_s,
    "processing_s": processing_s,
    "processing_s_per_event": processing_s_per_event,
    "loaded_k4geo_libraries": sorted(loaded_k4geo_libraries),
    "loaded_k4rec_libraries": sorted(loaded_k4rec_libraries),
    "command": command,
}
with open(metrics_path, "w") as stream:
    json.dump(metrics, stream, indent=2)
raise SystemExit(status)
PY
}

append_ddsim_metrics() {
  local metrics=$1
  local output=$2
  local variant=$3
  local case_name=$4
  local repetition=$5
  local expected_library_dir=$6
  python3 - "${metrics}" "${output}" "${variant}" "${case_name}" "${repetition}" \
    "${expected_library_dir}" >>"${metrics_tsv}" <<'PY'
import json
from pathlib import Path
import sys

metrics_path, output_path, variant, case_name, repetition, expected_library_dir = sys.argv[1:]
metrics = json.loads(Path(metrics_path).read_text())
loaded = metrics.get("loaded_k4geo_libraries", [])
if not any(str(Path(path)).startswith(str(Path(expected_library_dir))) for path in loaded):
    raise SystemExit(
        f"DDSim did not map a k4geo library from {expected_library_dir}; observed={loaded}"
    )
timing_fields = ("startup_s", "processing_s", "processing_s_per_event")
if any(metrics.get(field) is None for field in timing_fields):
    raise SystemExit(f"DDSim timing summary missing from {metrics_path}")
print(
    f"{variant}\t{case_name}\t{repetition}\t"
    f"{metrics['user_s']:.6f}\t{metrics['system_s']:.6f}\t{metrics['wall_s']:.6f}\t"
    f"{metrics['max_rss_kb']}\t{Path(output_path).stat().st_size}\t"
    f"{metrics['startup_s']:.6f}\t{metrics['processing_s']:.6f}\t"
    f"{metrics['processing_s_per_event']:.6f}"
)
PY
}

case_detector() {
  case "$1" in
    clic_*) printf 'clic\n' ;;
    cld_*) printf 'cld\n' ;;
    idea_keyed_lookup) printf 'idea_fast\n' ;;
    idea_standard_*) printf 'idea_standard\n' ;;
    idea_fast_*) printf 'idea_fast\n' ;;
    *) die "unknown validation case: $1" ;;
  esac
}

case_particle() {
  case "$1" in
    idea_keyed_lookup) printf 'pi-\n' ;;
    *_electron) printf 'e-\n' ;;
    *_pion) printf 'pi-\n' ;;
    *_empty) printf 'geantino\n' ;;
    *) die "unknown particle case: $1" ;;
  esac
}

case_repetitions() {
  case "$1" in
    clic_electron|cld_electron|idea_fast_electron) printf '%s\n' "${performance_repetitions}" ;;
    *) printf '1\n' ;;
  esac
}

run_case() {
  local variant=$1
  local case_name=$2
  local repetition=$3
  local detector particle number_of_events gun_energy geometry steering=""
  local run_dir output log metrics

  detector=$(case_detector "${case_name}")
  particle=$(case_particle "${case_name}")
  number_of_events=${4:-${events}}
  gun_energy=${5:-10*GeV}
  run_dir="${workdir}/runs/${variant}/${case_name}/rep-${repetition}"
  output="${run_dir}/events.root"
  log="${run_dir}/ddsim.log"
  metrics="${run_dir}/metrics.json"
  mkdir -p "${run_dir}"

  case "${detector}" in
    clic)
      geometry="${source_dir[${variant}]}/CLIC/compact/CLIC_o3_v15/CLIC_o3_v15.xml"
      ;;
    cld)
      geometry="${source_dir[${variant}]}/FCCee/CLD/compact/CLD_o2_v08/CLD_o2_v08.xml"
      ;;
    idea_standard)
      geometry="${source_dir[${variant}]}/test/compact/IDEA_withDRC_o1_v03.xml"
      steering="${source_dir[${variant}]}/example/SteeringFile_IDEA_o1_v03.py"
      ;;
    idea_fast)
      geometry="${source_dir[${variant}]}/test/compact/IDEA_withDRC_o1_v03.xml"
      steering="${workdir}/steering/${variant}/SteeringFile_IDEA_o1_v03_fast.py"
      ;;
  esac
  [[ -r "${geometry}" ]] || die "missing geometry: ${geometry}"
  [[ -z "${steering}" || -r "${steering}" ]] || die "missing steering file: ${steering}"

  printf 'Running %-9s %-23s repetition %d (%d event(s))...\n' \
    "${variant}" "${case_name}" "${repetition}" "${number_of_events}"
  current_log="${log}"
  (
    # The Key4HEP stack is already active. Prepending only this disposable
    # install selects its plugins without re-sourcing ROOT/DD4hep setup code.
    export LD_LIBRARY_PATH="${install_dir[${variant}]}/lib:${LD_LIBRARY_PATH:-}"
    export ROOT_LIBRARY_PATH="${install_dir[${variant}]}/lib:${ROOT_LIBRARY_PATH:-}"
    export K4GEO="${source_dir[${variant}]}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1
    cd "${run_dir}"

    command=(
      ddsim
      --compactFile "${geometry}"
      --runType batch
      --enableGun
      --outputFile "$(basename "${output}")"
      --numberOfEvents "${number_of_events}"
      --random.seed 1988301045
      --random.enableEventSeed
      --gun.particle "${particle}"
      --gun.energy "${gun_energy}"
      --gun.direction '1,0,0'
      --crossingAngleBoost 0.0
    )
    if [[ -n "${steering}" ]]; then
      command+=(--steeringFile "${steering}")
    fi
    run_measured "${metrics}" "${log}" "${command[@]}"
  ) || die "DDSim failed for ${variant}/${case_name}/rep-${repetition}"
  [[ -s "${output}" ]] || die "DDSim produced no output for ${variant}/${case_name}/rep-${repetition}"
  current_log=""

  append_ddsim_metrics "${metrics}" "${output}" "${variant}" "${case_name}" \
    "${repetition}" "${install_dir[${variant}]}/lib"
}

generate_qq_events() {
  local generation_dir="${workdir}/qq_generation"
  local card="${generation_dir}/card.cmd"
  local output="${generation_dir}/events.hepmc"
  local log="${generation_dir}/generation.log"
  local metrics="${generation_dir}/metrics.json"
  mkdir -p "${generation_dir}"
  cp "${pythia_qq_card}" "${card}"
  printf '\nRandom:seed=%s\n' "${qq_seed}" >>"${card}"

  printf '\nGenerating %d fixed-seed p8_ee_qq_ecm365 event(s)...\n' "${qq_events}"
  current_log="${log}"
  (
    cd "${generation_dir}"
    run_measured "${metrics}" "${log}" \
      k4run "${cld_config_dir}/pythia.py" \
      -n "${qq_events}" \
      --Dumper.Filename "$(basename "${output}")" \
      --Pythia8.PythiaInterface.pythiacard "$(basename "${card}")"
  ) || die "qq event generation failed"
  [[ -s "${output}" ]] || die "qq generation produced no HepMC output"
  current_log=""
}

run_qq_simulation() {
  local variant=$1
  local run_dir="${workdir}/runs/${variant}/idea_qq/rep-1"
  local input="${workdir}/qq_generation/events.hepmc"
  local output="${run_dir}/events.root"
  local log="${run_dir}/ddsim.log"
  local metrics="${run_dir}/metrics.json"
  local geometry="${source_dir[${variant}]}/FCCee/IDEA/compact/IDEA_o1_v03/IDEA_o1_v03.xml"
  local steering="${workdir}/steering/${variant}/SteeringFile_IDEA_o1_v03_fast.py"
  mkdir -p "${run_dir}"
  [[ -r "${input}" ]] || die "missing generated qq input: ${input}"
  [[ -r "${geometry}" ]] || die "missing IDEA production geometry: ${geometry}"
  [[ -r "${steering}" ]] || die "missing IDEA fast-simulation steering: ${steering}"

  printf 'Running %-9s %-23s (%d event(s))...\n' "${variant}" idea_qq "${qq_events}"
  current_log="${log}"
  (
    export LD_LIBRARY_PATH="${install_dir[${variant}]}/lib:${LD_LIBRARY_PATH:-}"
    export ROOT_LIBRARY_PATH="${install_dir[${variant}]}/lib:${ROOT_LIBRARY_PATH:-}"
    export K4GEO="${source_dir[${variant}]}"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1
    cd "${run_dir}"
    run_measured "${metrics}" "${log}" \
      ddsim \
      --compactFile "${geometry}" \
      --steeringFile "${steering}" \
      --runType batch \
      --inputFiles "${input}" \
      --outputFile "$(basename "${output}")" \
      --numberOfEvents -1 \
      --random.seed "${qq_seed}" \
      --random.enableEventSeed
  ) || die "DDSim failed for ${variant}/idea_qq"
  [[ -s "${output}" ]] || die "DDSim produced no output for ${variant}/idea_qq"
  current_log=""
  python3 - "${output}" "${qq_events}" "${variant}" <<'PY'
import sys
import uproot

path, expected, variant = sys.argv[1:]
with uproot.open(path) as root_file:
    observed = root_file["events"].num_entries
if observed != int(expected):
    raise SystemExit(f"{variant} qq event-count mismatch: {observed} != {expected}")
print(f"PASS {variant} qq event count: {observed}")
PY
  append_ddsim_metrics "${metrics}" "${output}" "${variant}" idea_qq 1 \
    "${install_dir[${variant}]}/lib"
}

prepare_reconstruction_inputs() {
  local input_dir="${workdir}/reconstruction-inputs"
  local data_file="${input_dir}/DataAlgFORGEANT.root"
  mkdir -p "${input_dir}"
  if [[ ! -s "${data_file}" ]]; then
    printf '\nDownloading the IDEA DCH digitization data file...\n'
    current_log="${input_dir}/download.log"
    wget -nv -O "${data_file}" \
      https://fccsw.web.cern.ch/fccsw/filesForSimDigiReco/IDEA/DataAlgFORGEANT.root \
      >"${current_log}" 2>&1 || die "failed to download IDEA DCH digitization data"
    [[ -s "${data_file}" ]] || die "downloaded IDEA DCH digitization data is empty"
    current_log=""
  fi
}

reconstruction_metrics_tsv="${workdir}/reconstruction_metrics.tsv"
printf 'variant\torder\tuser_s\tsystem_s\twall_s\tmax_rss_kb\toutput_bytes\n' >"${reconstruction_metrics_tsv}"

run_qq_reconstruction() {
  local variant=$1
  local reverse_inputs=${2:-0}
  local order=normal suffix=""
  if ((reverse_inputs)); then
    order=reversed
    suffix="-reversed"
  fi
  local run_dir="${workdir}/reconstruction/${variant}${suffix}"
  local input="${workdir}/runs/combined/idea_qq/rep-1/events.root"
  local output="${run_dir}/events_digi_reco.root"
  local log="${run_dir}/reconstruction.log"
  local metrics="${run_dir}/metrics.json"
  local summary="${run_dir}/validation_summary.json"
  local steering="${reccalo_steering[${variant}]}"
  ((reverse_inputs)) && steering="${reccalo_reversed_steering[${variant}]}"
  mkdir -p "${run_dir}"
  [[ -r "${input}" ]] || die "missing combined-variant qq simulation for reconstruction: ${input}"
  [[ -r "${steering}" ]] || die "missing k4RecCalorimeter steering for ${variant}/${order}"
  ln -s "${workdir}/reconstruction-inputs/DataAlgFORGEANT.root" "${run_dir}/DataAlgFORGEANT.root"

  printf 'Reconstructing %d qq event(s) with k4RecCalorimeter %-9s (%s order, GGTF disabled)...\n' \
    "${qq_events}" "${variant}" "${order}"
  current_log="${log}"
  (
    export LD_LIBRARY_PATH="${install_dir[combined]}/lib:${reccalo_plugin_dir[${variant}]}:${LD_LIBRARY_PATH:-}"
    export ROOT_LIBRARY_PATH="${install_dir[combined]}/lib:${reccalo_plugin_dir[${variant}]}:${ROOT_LIBRARY_PATH:-}"
    export PYTHONPATH="${reccalo_install_dir[${variant}]}/python/RecCalorimeter:${reccalo_install_dir[${variant}]}/python/RecFCCeeCalorimeter:${PYTHONPATH:-}"
    export K4GEO="${source_dir[combined]}"
    export ENABLE_GGTF=0
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TBB_NUM_THREADS=1
    cd "${run_dir}"
    run_measured "${metrics}" "${log}" \
      k4run "${steering}" \
      --IOSvc.Input "${input}" \
      --IOSvc.Output "${output}"
  ) || die "IDEA qq reconstruction failed for ${variant}/${order}"
  [[ -s "${output}" ]] || die "IDEA qq reconstruction produced no output for ${variant}/${order}"
  current_log=""

  python3 - "${log}" "${metrics}" "${output}" "${install_dir[combined]}/lib" \
    "${reccalo_plugin_dir[${variant}]}" "${variant}" "${order}" >>"${reconstruction_metrics_tsv}" <<'PY'
import json
from pathlib import Path
import re
import sys

log_path, metrics_path, output_path, expected_k4geo_dir, expected_k4rec_dir, variant, order = sys.argv[1:]
log_text = Path(log_path).read_text(errors="replace")
bad_lines = re.findall(r"(?m)^\S+\s+(?:ERROR|FATAL)\b.*$", log_text)
if bad_lines:
    raise SystemExit("reconstruction logged errors: " + " | ".join(bad_lines[:10]))
metrics = json.loads(Path(metrics_path).read_text())
loaded_k4geo = metrics.get("loaded_k4geo_libraries", [])
if not any(str(Path(path)).startswith(str(Path(expected_k4geo_dir))) for path in loaded_k4geo):
    raise SystemExit(f"reconstruction did not map k4geo from {expected_k4geo_dir}; observed={loaded_k4geo}")
loaded_k4rec = metrics.get("loaded_k4rec_libraries", [])
if not any(str(Path(path)).startswith(str(Path(expected_k4rec_dir))) for path in loaded_k4rec):
    raise SystemExit(f"reconstruction did not map k4RecCalorimeter from {expected_k4rec_dir}; observed={loaded_k4rec}")
print(
    f"{variant}\t{order}\t{metrics['user_s']:.6f}\t{metrics['system_s']:.6f}\t"
    f"{metrics['wall_s']:.6f}\t{metrics['max_rss_kb']}\t{Path(output_path).stat().st_size}"
)
PY

  validator_args=(
    validate "${output}"
    --variant "${variant}-${order}"
    --expected-events "${qq_events}"
    --summary "${summary}"
  )
  case "${variant}" in
    optical|combined) validator_args+=(--expect-optical-links) ;;
  esac
  case "${variant}" in
    cluster|combined) validator_args+=(--expect-dual-readout) ;;
  esac
  case "${variant}" in
    truth|combined) validator_args+=(--expect-positive-linked-cells) ;;
  esac
  case "${variant}" in
    truth|combined) validator_args+=(--expect-cellid-cluster-truth) ;;
  esac
  ((reverse_inputs)) && validator_args+=(--reversed-inputs)
  python3 "${validation_script_dir}/validate_k4reccalorimeter_output.py" "${validator_args[@]}" || \
    die "k4RecCalorimeter output validation failed for ${variant}/${order}"
}

declare -a cases=(clic_electron cld_electron idea_standard_electron idea_fast_electron idea_fast_empty)
if ((include_pions)); then
  cases+=(clic_pion cld_pion idea_fast_pion)
fi

# Case-major ordering keeps paired measurements close in time. Reverse the
# variant order on even repetitions to reduce systematic warm-cache bias.
for case_name in "${cases[@]}"; do
  repetitions=$(case_repetitions "${case_name}")
  for ((repetition = 1; repetition <= repetitions; ++repetition)); do
    if ((repetition % 2)); then
      run_order=(main truth lookup combined)
    else
      run_order=(combined lookup truth main)
    fi
    for variant in "${run_order[@]}"; do
      run_case "${variant}" "${case_name}" "${repetition}"
    done
  done
done

printf '\nRunning the isolated pre-fix/current keyed-hit lookup comparison...\n'
run_case pre_fix idea_keyed_lookup 1 "${speedup_events}" "${speedup_energy_gev}*GeV"
run_case main idea_keyed_lookup 1 "${speedup_events}" "${speedup_energy_gev}*GeV"

printf '\nRunning the representative pre-fix/current qq comparison...\n'
generate_qq_events
run_qq_simulation pre_fix
run_qq_simulation main
run_qq_simulation combined

compare_edm4hep() {
  local reference=$1
  local candidate=$2
  local mode=$3
  python3 - "${reference}" "${candidate}" "${mode}" <<'PY'
import sys
import awkward as ak
import uproot

reference_path, candidate_path, mode = sys.argv[1:]
if mode not in {"exact", "ignore_drc_truth"}:
    raise SystemExit(f"unknown comparison mode: {mode}")

def ignored(name):
    # Wall-clock metadata is not physics content.
    if "EventHeader.timeStamp" in name:
        return True
    if mode == "ignore_drc_truth":
        truth_names = (
            "DRcaloSiPMreadoutSimHitContributions",
            "_DRcaloSiPMreadoutSimHit_contributions",
            "DRcaloSiPMreadoutSimHit.contributions_begin",
            "DRcaloSiPMreadoutSimHit.contributions_end",
        )
        return any(token in name for token in truth_names)
    return False

def leaves(tree):
    result = {}
    for name in tree.keys(recursive=True):
        branch = tree[name]
        if len(branch.branches) == 0 and not ignored(name):
            result[name] = branch
    return result

with uproot.open(reference_path) as reference_file, uproot.open(candidate_path) as candidate_file:
    reference_tree = reference_file["events"]
    candidate_tree = candidate_file["events"]
    if reference_tree.num_entries != candidate_tree.num_entries:
        raise SystemExit(
            f"event-count mismatch: {reference_tree.num_entries} != {candidate_tree.num_entries}"
        )
    reference_leaves = leaves(reference_tree)
    candidate_leaves = leaves(candidate_tree)
    missing = sorted(set(reference_leaves) - set(candidate_leaves))
    extra = sorted(set(candidate_leaves) - set(reference_leaves))
    if missing or extra:
        raise SystemExit(f"branch mismatch: missing={missing[:20]} extra={extra[:20]}")

    mismatches = []
    for name in sorted(reference_leaves):
        left = reference_leaves[name].array(library="ak")
        right = candidate_leaves[name].array(library="ak")
        if not ak.almost_equal(left, right, rtol=0.0, atol=0.0, dtype_exact=True):
            mismatches.append(name)
            if len(mismatches) == 20:
                break
    if mismatches:
        raise SystemExit("payload mismatch in branches: " + ", ".join(mismatches))

print(f"PASS {mode}: {reference_path} == {candidate_path}")
PY
}

validate_truth_closure() {
  local input_file=$1
  python3 - "${input_file}" <<'PY'
import math
import sys
from podio import root_io

path = sys.argv[1]
reader = root_io.Reader(path)
events = 0
hits_seen = 0
contributions_seen = 0
for frame in reader.get("events"):
    events += 1
    names = set(frame.getAvailableCollections())
    hit_name = "DRcaloSiPMreadoutSimHit"
    contribution_name = "DRcaloSiPMreadoutSimHitContributions"
    if hit_name not in names or contribution_name not in names:
        raise SystemExit(f"missing IDEA DRC truth collection in event {events - 1}: {sorted(names)}")
    hits = frame.get(hit_name)
    contribution_collection = frame.get(contribution_name)
    mcparticles = frame.get("MCParticles")
    mcparticle_ids = {
        (int(particle.getObjectID().collectionID), int(particle.getObjectID().index))
        for particle in mcparticles
    }
    event_relation_count = 0
    event_hit_energy = 0.0
    event_contribution_energy = 0.0
    for hit in hits:
        hits_seen += 1
        relations = hit.getContributions()
        relation_energy = 0.0
        for contribution in relations:
            contributions_seen += 1
            event_relation_count += 1
            energy = float(contribution.getEnergy())
            if energy <= 0.0 or not energy.is_integer():
                raise SystemExit(f"non-positive or non-integral photon count: {energy}")
            relation_energy += energy
            particle = contribution.getParticle()
            particle_id = (
                int(particle.getObjectID().collectionID),
                int(particle.getObjectID().index),
            )
            if particle_id not in mcparticle_ids:
                raise SystemExit(f"Cherenkov contribution has an invalid MCParticle relation: {particle_id}")
            if not math.isfinite(float(particle.getCharge())) or abs(float(particle.getCharge())) <= 0.0:
                raise SystemExit(f"Cherenkov contribution is not assigned to a charged ancestor: {particle_id}")
        hit_energy = float(hit.getEnergy())
        if not math.isclose(relation_energy, hit_energy, rel_tol=0.0, abs_tol=0.0):
            raise SystemExit(
                f"per-hit photon closure failed: contributions={relation_energy} hit={hit_energy}"
            )
        event_hit_energy += hit_energy
        event_contribution_energy += relation_energy
    if event_relation_count != len(contribution_collection):
        raise SystemExit(
            f"unreferenced or multiply referenced contribution: relations={event_relation_count} "
            f"collection={len(contribution_collection)}"
        )
    if event_hit_energy != event_contribution_energy:
        raise SystemExit(
            f"event photon closure failed: contributions={event_contribution_energy} hits={event_hit_energy}"
        )

print(
    f"PASS truth closure: events={events} hits={hits_seen} "
    f"contributions={contributions_seen} file={path}"
)
PY
}

printf '\nComparing deterministic EDM4hep payloads...\n'
for case_name in "${cases[@]}"; do
  main_output="${workdir}/runs/main/${case_name}/rep-1/events.root"
  truth_output="${workdir}/runs/truth/${case_name}/rep-1/events.root"
  lookup_output="${workdir}/runs/lookup/${case_name}/rep-1/events.root"
  combined_output="${workdir}/runs/combined/${case_name}/rep-1/events.root"

  compare_edm4hep "${main_output}" "${lookup_output}" exact
  compare_edm4hep "${truth_output}" "${combined_output}" exact
  case "${case_name}" in
    idea_*)
      compare_edm4hep "${main_output}" "${truth_output}" ignore_drc_truth
      validate_truth_closure "${truth_output}"
      validate_truth_closure "${combined_output}"
      ;;
    *)
      compare_edm4hep "${main_output}" "${truth_output}" exact
      ;;
  esac
done

keyed_current_output="${workdir}/runs/main/idea_keyed_lookup/rep-1/events.root"
keyed_pre_fix_output="${workdir}/runs/pre_fix/idea_keyed_lookup/rep-1/events.root"
compare_edm4hep "${keyed_pre_fix_output}" "${keyed_current_output}" exact

qq_current_output="${workdir}/runs/main/idea_qq/rep-1/events.root"
qq_pre_fix_output="${workdir}/runs/pre_fix/idea_qq/rep-1/events.root"
qq_combined_output="${workdir}/runs/combined/idea_qq/rep-1/events.root"
compare_edm4hep "${qq_pre_fix_output}" "${qq_current_output}" exact
compare_edm4hep "${qq_current_output}" "${qq_combined_output}" ignore_drc_truth
validate_truth_closure "${qq_combined_output}"

printf '\nRunning the k4RecCalorimeter reconstruction matrix...\n'
prepare_reconstruction_inputs
for variant in "${reccalo_variants[@]}"; do
  run_qq_reconstruction "${variant}" 0
done
for variant in cluster combined; do
  run_qq_reconstruction "${variant}" 1
  python3 "${validation_script_dir}/validate_k4reccalorimeter_output.py" compare-order \
    "${workdir}/reconstruction/${variant}/validation_summary.json" \
    "${workdir}/reconstruction/${variant}-reversed/validation_summary.json" || \
    die "dual-readout input-order validation failed for ${variant}"
done

printf '\nk4RecCalorimeter reconstruction performance summary...\n'
python3 - "${reconstruction_metrics_tsv}" "${performance_tolerance_percent}" "${qq_events}" <<'PY'
import csv
import sys

path, tolerance_percent, events = sys.argv[1:]
limit = 1.0 + int(tolerance_percent) / 100.0
events = int(events)
rows = {}
with open(path, newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        row["cpu_s"] = float(row["user_s"]) + float(row["system_s"])
        row["wall_s"] = float(row["wall_s"])
        row["max_rss_kb"] = int(row["max_rss_kb"])
        row["output_bytes"] = int(row["output_bytes"])
        rows[(row["variant"], row["order"])] = row

print(f"{'variant':9} {'order':8} {'cpu_s':>11} {'wall_s':>11} {'rss_mib':>10} {'file_mib':>10}")
for key in sorted(rows):
    row = rows[key]
    print(
        f"{key[0]:9} {key[1]:8} {row['cpu_s']:11.2f} {row['wall_s']:11.2f} "
        f"{row['max_rss_kb']/1024:10.1f} {row['output_bytes']/1048576:10.1f}"
    )

if events < 3:
    print("Reconstruction performance gates skipped: fewer than three qq events.")
    raise SystemExit(0)

failures = []
baseline = rows[("main", "normal")]
for variant in ("truth", "optical"):
    candidate = rows[(variant, "normal")]
    for metric in ("cpu_s", "max_rss_kb"):
        ratio = candidate[metric] / baseline[metric]
        status = "PASS" if ratio <= limit else "FAIL"
        print(f"{status} reconstruction {variant}/main {metric} ratio={ratio:.3f} limit={limit:.3f}")
        if status == "FAIL":
            failures.append((variant, metric, ratio, limit))

for variant in ("cluster", "combined"):
    normal = rows[(variant, "normal")]
    reversed_row = rows[(variant, "reversed")]
    for metric in ("cpu_s", "max_rss_kb"):
        ratio = max(normal[metric], reversed_row[metric]) / min(normal[metric], reversed_row[metric])
        status = "PASS" if ratio <= limit else "FAIL"
        print(f"{status} reconstruction {variant} order stability {metric} ratio={ratio:.3f} limit={limit:.3f}")
        if status == "FAIL":
            failures.append((variant, "order", metric, ratio, limit))

if failures:
    raise SystemExit(f"{len(failures)} reconstruction performance gate(s) failed")
PY

run_lookup_microbenchmark() {
  local source="${workdir}/lookup_benchmark.cpp"
  local binary="${workdir}/lookup_benchmark"
  cat >"${source}" <<'CPP'
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
volatile std::uint64_t sink = 0;

struct CountingLess {
  static std::uint64_t comparisons;
  bool operator()(int left, int right) const {
    ++comparisons;
    return left < right;
  }
};
std::uint64_t CountingLess::comparisons = 0;
using value_map = std::map<int, int, CountingLess>;

__attribute__((noinline)) std::uint64_t old_lookup(const value_map& values, int bins, int rounds) {
  std::uint64_t total = 0;
  for (int round = 0; round < rounds; ++round) {
    for (int bin = 1; bin <= bins; ++bin) {
      int count = 0;
      if (values.find(bin) != values.end())
        count = values.at(bin);
      total += static_cast<std::uint64_t>(count);
    }
  }
  return total;
}

__attribute__((noinline)) std::uint64_t new_lookup(const value_map& values, int bins, int rounds) {
  std::uint64_t total = 0;
  for (int round = 0; round < rounds; ++round) {
    for (int bin = 1; bin <= bins; ++bin) {
      int count = 0;
      const auto entry = values.find(bin);
      if (entry != values.end())
        count = entry->second;
      total += static_cast<std::uint64_t>(count);
    }
  }
  return total;
}

double measure(bool updated, const value_map& values, int bins, int rounds) {
  const auto start = clock_type::now();
  sink += updated ? new_lookup(values, bins, rounds) : old_lookup(values, bins, rounds);
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

void benchmark(const std::string& name, int bins, int stride) {
  value_map values;
  for (int bin = 1; bin <= bins; bin += stride)
    values.emplace(bin, 1 + bin % 7);

  CountingLess::comparisons = 0;
  sink += old_lookup(values, bins, 1);
  const auto old_comparisons = CountingLess::comparisons;
  CountingLess::comparisons = 0;
  sink += new_lookup(values, bins, 1);
  const auto new_comparisons = CountingLess::comparisons;
  if (!(new_comparisons < old_comparisons))
    throw std::runtime_error(name + " did not reduce ordered-map comparisons");

  constexpr int samples = 5;
  const int rounds = std::max(25, 100000 / bins);
  std::vector<double> old_times;
  std::vector<double> new_times;
  for (int sample = 0; sample < samples; ++sample) {
    if (sample % 2 == 0) {
      old_times.push_back(measure(false, values, bins, rounds));
      new_times.push_back(measure(true, values, bins, rounds));
    } else {
      new_times.push_back(measure(true, values, bins, rounds));
      old_times.push_back(measure(false, values, bins, rounds));
    }
  }
  std::sort(old_times.begin(), old_times.end());
  std::sort(new_times.begin(), new_times.end());
  const double old_median = old_times[samples / 2];
  const double new_median = new_times[samples / 2];
  const double ratio = new_median / old_median;
  std::cout << name << " old_s=" << std::setprecision(9) << old_median
            << " new_s=" << new_median << " ratio=" << ratio
            << " old_comparisons=" << old_comparisons
            << " new_comparisons=" << new_comparisons << '\n';
}

int main() {
  // FiberDRCaloSDAction has 2,000 time bins and 120 wavelength bins. These
  // sparse, typical, and dense occupancies cover the maps serialized by IDEA.
  benchmark("time_sparse", 2000, 100);
  benchmark("time_typical", 2000, 10);
  benchmark("time_dense", 2000, 1);
  benchmark("wavelength_typical", 120, 5);
  std::cout << "sink=" << sink << '\n';
}
CPP
  c++ -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Werror "${source}" -o "${binary}"
  "${binary}"
}

printf '\nRunning the standalone dense-map lookup microbenchmark...\n'
run_lookup_microbenchmark

printf '\nPerformance summary and regression gates...\n'
python3 - "${metrics_tsv}" "${performance_repetitions}" \
  "${performance_tolerance_percent}" "${truth_tolerance_percent}" \
  "${minimum_speedup_factor}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

path, repetitions, general_tolerance, truth_tolerance, minimum_speedup = sys.argv[1:]
repetitions = int(repetitions)
general_limit = 1.0 + int(general_tolerance) / 100.0
truth_limit = 1.0 + int(truth_tolerance) / 100.0
minimum_speedup = float(minimum_speedup)
groups = defaultdict(list)
with open(path, newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        row["cpu_s"] = float(row["user_s"]) + float(row["system_s"])
        row["wall_s"] = float(row["wall_s"])
        row["max_rss_kb"] = int(row["max_rss_kb"])
        row["output_bytes"] = int(row["output_bytes"])
        row["processing_s_per_event"] = float(row["processing_s_per_event"])
        groups[(row["case"], row["variant"])].append(row)

summary = {}
print(
    f"{'case':23} {'variant':9} {'n':>2} {'cpu_s':>11} {'wall_s':>11} "
    f"{'proc_s/evt':>11} {'rss_mib':>10} {'file_mib':>10}"
)
for key in sorted(groups):
    rows = groups[key]
    values = {
        "n": len(rows),
        "cpu": statistics.median(row["cpu_s"] for row in rows),
        "wall": statistics.median(row["wall_s"] for row in rows),
        "rss": statistics.median(row["max_rss_kb"] for row in rows),
        "size": statistics.median(row["output_bytes"] for row in rows),
        "processing_per_event": statistics.median(
            row["processing_s_per_event"] for row in rows
        ),
    }
    summary[key] = values
    print(
        f"{key[0]:23} {key[1]:9} {values['n']:2d} {values['cpu']:11.2f} "
        f"{values['wall']:11.2f} {values['processing_per_event']:11.2f} "
        f"{values['rss']/1024:10.1f} {values['size']/1048576:10.1f}"
    )

failures = []
pre_fix = summary[("idea_qq", "pre_fix")]
current = summary[("idea_qq", "main")]
speedup = pre_fix["processing_per_event"] / current["processing_per_event"]
speedup_status = "PASS" if speedup >= minimum_speedup else "FAIL"
print(
    f"{speedup_status} idea_qq current/pre_fix processing speedup="
    f"{speedup:.2f}x minimum={minimum_speedup:.2f}x"
)
if speedup_status == "FAIL":
    failures.append(("idea_qq", "processing_speedup", speedup, minimum_speedup))

gun_pre_fix = summary[("idea_keyed_lookup", "pre_fix")]
gun_current = summary[("idea_keyed_lookup", "main")]
gun_speedup = gun_pre_fix["processing_per_event"] / gun_current["processing_per_event"]
print(
    f"INFO idea_keyed_lookup 50 GeV gun current/pre_fix processing speedup="
    f"{gun_speedup:.2f}x (equivalence check only)"
)

if repetitions < 2:
    print("Performance gates skipped: fewer than two repetitions were requested.")
    if failures:
        raise SystemExit(f"{len(failures)} computational regression gate(s) failed")
    raise SystemExit(0)

for case_name in ("clic_electron", "cld_electron", "idea_fast_electron"):
    comparisons = []
    if case_name.startswith(("clic_", "cld_")):
        comparisons = [("truth", "main", general_limit), ("lookup", "main", general_limit),
                       ("combined", "main", general_limit)]
    else:
        comparisons = [("truth", "main", truth_limit), ("lookup", "main", general_limit),
                       ("combined", "truth", general_limit)]
    for candidate, baseline, limit in comparisons:
        candidate_values = summary[(case_name, candidate)]
        baseline_values = summary[(case_name, baseline)]
        for metric in ("cpu", "rss"):
            ratio = candidate_values[metric] / baseline_values[metric]
            status = "PASS" if ratio <= limit else "FAIL"
            print(
                f"{status} {case_name} {candidate}/{baseline} {metric} "
                f"ratio={ratio:.3f} limit={limit:.3f}"
            )
            if status == "FAIL":
                failures.append((case_name, candidate, baseline, metric, ratio, limit))

if failures:
    raise SystemExit(f"{len(failures)} computational regression gate(s) failed")
PY

final_validation_dir="${workdir}/final_validation"
plot_script="${validation_script_dir}/plot_k4geo_keyed_lookup_validation.py"
[[ -r "${plot_script}" ]] || die "missing final-validation plotter: ${plot_script}"
printf '\nCreating final keyed-lookup physics and computing plots...\n'
python3 "${plot_script}" "${workdir}" "${final_validation_dir}" \
  --minimum-speedup "${minimum_speedup_factor}" || \
  die "final keyed-lookup plot validation failed"

printf '\nAll k4geo and k4RecCalorimeter validation gates passed.\n'
printf 'The map-lookup improvement is IDEA-specific; CLIC and CLD results are non-regression checks.\n'
printf 'Component refs:       %s\n' "${component_refs}"
printf 'Workflow refs:        %s\n' "${workflow_refs}"
printf 'Reconstruction table: %s\n' "${reconstruction_metrics_tsv}"
printf 'Final physics plot:   %s\n' "${final_validation_dir}/keyed_lookup_physics.png"
printf 'Final computing plot: %s\n' "${final_validation_dir}/keyed_lookup_computing.png"
