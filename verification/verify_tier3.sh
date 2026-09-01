#!/usr/bin/env bash
# Tier 3 -- cold start. Re-run this after fixing the gaps documented in
# COLD_START.md to check whether they're actually closed.
#
# This script deliberately does NOT patch around Findings 1/2/3 from
# COLD_START.md -- it runs the documented recipe as written, so a failure
# here means the documentation is still wrong, not that this script forgot
# a step. If you need it to pass today, see COLD_START.md for the manual
# workarounds and their exact diffs.
set -euo pipefail

REPO_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$(mktemp -d)}"
ENV_NAME="sqd-coldstart-$(date +%s)"
CLONE="$WORK/sqd-project"

echo "=== Tier 3 cold start -- $(date) ==="
echo "clone target: $CLONE"
echo "conda env:    $ENV_NAME"

echo
echo "--- 1. git clone ---"
time git clone "$REPO_SRC" "$CLONE"

echo
echo "--- 2. conda env create -f environment.yml ---"
source "$(conda info --base)/etc/profile.d/conda.sh"
time conda env create -f "$CLONE/environment.yml" -n "$ENV_NAME"

echo
echo "--- 2b. check for mpirun/mpicxx (Finding 1: expected ABSENT) ---"
if env -i PATH="$(conda info --base)/envs/$ENV_NAME/bin:/usr/bin:/bin" which mpirun mpicxx 2>/dev/null; then
  echo "mpirun/mpicxx present -- Finding 1 appears FIXED, environment.yml now provides them"
else
  echo "mpirun/mpicxx absent, as documented in COLD_START.md Finding 1 -- NOT fixed"
  echo "manual workaround: conda install -n $ENV_NAME -y -c conda-forge openmpi llvm-openmp clangxx_osx-arm64"
fi

echo
echo "--- 3. sbd source: NOT obtainable from this repository (undocumented) ---"
echo "sbd/ is gitignored and its acquisition is not documented in the README."
echo "This step cannot be automated here -- see COLD_START.md."
echo "If you have a local sbd/ checkout, copy it into $CLONE/sbd to continue manually."

echo
echo "--- 4. WITHOUT cached data: hide cache/, run the three checks ---"
if [ -d "$CLONE/sbd/apps/chemistry_tpb_selected_basis_diagonalization" ]; then
  mv "$CLONE/cache" "$CLONE/cache.hidden"

  echo "4a. README benchmark command"
  export SBD_BIN="$CLONE/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
  time mpirun -n 1 "$SBD_BIN" \
    --fcidump "$CLONE/sbd/data/n2/fcidump.txt" \
    --adetfile "$CLONE/sbd/data/n2/1em4-alpha.txt" \
    --bdetfile "$CLONE/sbd/data/n2/1em4-alpha.txt" \
    --bit_length 20 \
    --method 0 --iteration 200 --tolerance 1e-10 \
    --carryover_type 0 --shuffle 0 --init 0 \
    --adet_comm_size 1 --bdet_comm_size 1 --task_comm_size 1 \
    | tee "$WORK/4a_benchmark.log"
  grep -q "Energy = -109.0483526946501" "$WORK/4a_benchmark.log" \
    && echo "4a: PASS" || echo "4a: FAIL -- see $WORK/4a_benchmark.log"

  echo
  echo "4b. H10 R=1.6 reference construction from scratch"
  conda run -n "$ENV_NAME" python3 -c "
import sys
sys.path.insert(0, '$CLONE/scripts'); sys.path.insert(0, '$CLONE/src')
import run_ordering_pipeline as R
ref = R.build_or_load_h10_reference(1.6, 10, 'sto-6g', cachedir='$CLONE/cache/h10_R1.6')
print('E_CASCI fresh:', repr(float(ref['E_CASCI'])))
" | tee "$WORK/4b_reference.log"

  echo
  echo "4c. WARNING (Finding 4): a full evaluation against the freshly-rebuilt"
  echo "reference above is NOT expected to match the shipped per-layout CSVs"
  echo "-- see COLD_START.md Finding 4. This is a known, explained gap, not a"
  echo "bug in this script."

  mv "$CLONE/cache.hidden" "$CLONE/cache"
else
  echo "sbd/ not present in the clone -- step 4 skipped. See step 3 above."
fi

echo
echo "=== done. Working directory: $WORK (not cleaned up automatically) ==="
