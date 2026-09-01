# Tier 3 — Cold start

Run 2026-09-01. This is the check the audit spec calls "the one that matters,"
so it was run first, before any of the claims-manifest or Tier 0-2 work.

**Method.** A fresh `git clone` of the local repository (equivalent to
cloning from GitHub — only committed content is present) into a scratch
directory, a **new** conda environment built only from `environment.yml`
(not the pre-existing `sqd` env this whole project has been developed in),
and `sbd` built fresh from the documented recipe. `cache/` was renamed aside
before steps 4b/4c so reference construction could not silently reuse it.

Two genuine failures were found. Both are reported here, not worked around.

---

## Finding 1 (blocking): `environment.yml` does not provide `sbd`'s build toolchain

The README's benchmark command is the very first thing a reader runs, and it
requires `mpirun`. Building `sbd` requires `mpicxx` and OpenMP. Neither
`mpirun`/`mpicxx` (from `openmpi`) nor `llvm-openmp` nor a C++ compiler
(`clangxx_osx-arm64`) is listed in `environment.yml`.

Verified directly, with a fully isolated `PATH` so no pre-existing
environment could mask the gap:

```
$ env -i PATH="/Users/.../envs/sqd-coldstart-verify/bin:/usr/bin:/bin" bash -c \
    'which mpirun; echo $?; which mpicxx; echo $?'
mpirun exit:1
mpicxx exit:1
```

confirmed by listing the fresh env's `bin/` directly: no `mpirun`, `mpicxx`,
or `clang++`, only an unrelated `compile_et`. `conda activate sqd` "worked"
for me earlier in this project only because my personal shell already had
the original, separately-built `sqd` environment on `PATH` — that is not
available to a genuinely new reader.

**Fix required:** add `openmpi`, `llvm-openmp`, and a C++ compiler
(`clangxx_osx-arm64` on macOS/Apple Silicon; the Linux equivalent for other
platforms) to `environment.yml`, or document them as a separate manual
prerequisite in the README's Install section. Not fixed here — Step 2 of the
task says report, not silently patch.

## Finding 2 (blocking): `sbd-build-notes/Configuration.macos-arm64` hardcodes this machine's path

```
SYSLIB = -L/Users/maxim/miniforge3/envs/sqd/lib -lomp -framework Accelerate
```

This is an absolute path to one specific user's home directory and one
specific, particularly-named conda environment. It happened to link
successfully in this test only because that exact path coincidentally still
exists on this machine (a leftover, unrelated environment from earlier
project history) — not because it is correct for the environment actually
being tested (`sqd-coldstart-verify`, a different name, same machine).

Confirmed by inspecting the built binary:

```
$ otool -l diag | grep -A2 LC_RPATH
    path /Users/maxim/miniforge3/envs/sqd-coldstart-verify/lib
```

The binary's *actual* runtime rpath comes from `mpicxx`'s own automatic
environment-relative injection, not from the `Configuration` file's `-L`
flag at all — the shipped `-L` line is dead weight when it happens to
resolve, and a hard link failure when it doesn't (any machine without a
coincidentally-matching `/Users/maxim/miniforge3/envs/sqd` path, i.e. any
machine other than this one). Rebuilding with the `-L` line removed
entirely (`SYSLIB = -lomp -framework Accelerate`) links and runs correctly,
confirming the hardcoded path is unneeded, not merely wrong:

```
$ env -i HOME=$HOME PATH=".../sqd-coldstart-verify/bin:/usr/bin:/bin" make
mpicxx -o diag main.o   -lomp -framework Accelerate      # succeeds
```

**Fix required:** drop the `-L` line from
`sbd-build-notes/Configuration.macos-arm64` (or replace it with
`-L$(CONDA_PREFIX)/lib`, portable). Not fixed here.

## Finding 3 (also worth noting): `environment.yml`'s env name collides with any pre-existing "sqd" environment

`environment.yml` specifies `name: sqd`. `conda env create -f environment.yml`
on any machine that already has an environment named `sqd` (for instance,
this one) will refuse to overwrite it silently but gives no indication in
the README that this could happen or what to do about it. This audit worked
around it by creating `sqd-coldstart-verify` instead; a reader following the
README literally on a machine with an unrelated pre-existing `sqd` env would
hit an unexplained `CondaValueError`.

---

## Step-by-step, with the above gaps bridged manually and documented

Since `sbd/` itself is gitignored (correctly — it's vendored, not part of
this project's history) and the README does not state where to obtain its
source (only `sbd/README.md`'s own license link points at
`github.com/r-ccs-cms/sbd`, and only in passing), this test copied the
already-present local `sbd/` source into the clone to test the *build* step.
**The acquisition step itself was not and could not be tested** — this is a
fourth, separate documentation gap.

| Step | Result | Wall time |
|---|---|---|
| `git clone` | OK | 1.7s (local; a real GitHub clone would add network time) |
| `conda env create -f environment.yml` | OK, but see Finding 1 | ~1 min *(warm local package cache from this machine's history — a genuinely first-time run downloads the full package set and will take longer)* |
| `conda install openmpi llvm-openmp clangxx_osx-arm64` (undocumented, required) | OK | ~5s (same caching caveat) |
| build `sbd` per documented recipe, path bug fixed | OK | 6.3s |
| **4a. README benchmark command** | **PASS** — `Energy = -109.0483526946501`, exact match | 2m31s |
| **4b. H10 R=1.6 reference construction from scratch** (`cache/` hidden) | **PASS** — fresh `E_CASCI = -4.966071088325831` vs. cached `-4.966071088325821`, agree to 1e-14 Ha | 24.6s |
| **4c. One complete small experiment end to end** | **See Finding 4 below** | 35s |

## Finding 4: a freshly-rebuilt H10 reference gives a different SQD answer

**Reclassified after investigation (see `TASK_A_FINDING4.md` for the full
characterisation): this is a genuine scientific result about the
near-degenerate H10 identity/default-anchor/budget-15 configuration, not a
defect in the repository or its documentation.** It does not block cold
start in the way Findings 1-3 did — the pipeline runs correctly end to end;
it is the *specific numeric outcome* at this one fragile configuration that
is not the kind of thing "cached vs. rebuilt" reproducibility can be
expected to guarantee. Summary below; full five-part characterisation
(which determinants swap, whether it's the distribution or just the cut,
how sensitive it is to perturbation size, whether it's specific to H10's
near-degenerate boundary, and whether more shots help) is in
`TASK_A_FINDING4.md`.

Step 4c ran one full evaluation — H10 identity chain, default anchor triple,
seed 2026, 2,000,000 shots, budget 15 — end to end (mask → LUCJ circuit →
Aer sampling → `sbd` diagonalization), using the reference freshly built in
4b. The stored value for this exact configuration
(`experiments/outputs/h10_baseline_R1.6/h10_baseline_results.csv`, and the
report's own Table in `sec:results`) is **300.32 mHa**. The cold-start run
gave **373.63 mHa** — a 73 mHa difference, reproduced exactly on a repeat
run (373.62816291596255 both times, so not shot noise), and reproduced again
(373.61 mHa) running the *original* repo's environment and `sbd` binary
against the *cold-start clone's* freshly-built reference data alone. Running
the original repo's own cached reference through the same minimal script
reproduces the stored 300.31919403956664 exactly.

This isolates the cause precisely: it is not the new conda environment, not
a different `sbd` binary, not a scripting bug (the minimal replication
script matches the stored value exactly when pointed at the original cached
reference) — it is that the **freshly-rebuilt reference data**, though it
agrees with the cached reference to 1e-14 Ha in every stored number, sits on
the *other side* of a near-exact tie in the determinant-selection boundary.
The report's own reproducibility section already documents why this
particular case is fragile: H10's marginal weight ratio at the selection
boundary is $w_{16}/w_{15}=0.989$, i.e. the 15th and 16th most-probable
determinants are almost equally likely. A perturbation as small as 1e-14 in
the underlying CCSD amplitudes is enough to swap which one clears the
budget-15 cutoff, and because these are near-degenerate boundary
determinants (not noise), swapping one changes the diagonalized energy by
tens of mHa, not a rounding amount.

**What this means for the audit's tolerances:** the manifest's declared
"re-sampled energies at fixed seed and shot count: exact to 1e-9 mHa"
tolerance is correct *only when reusing the exact cached reference file*.
It does not hold across independent reference reconstructions, even numerically
near-identical ones, specifically at configurations close to a selection-boundary
tie. The report's own text already scopes its bit-identical reproducibility
claim to repeated *sampling* at fixed reference data ("at $2\times10^{6}$
shots all five seeds return bit-identical energies and determinant sets") —
it does not claim, and should not be read as claiming, reproducibility
across independent reference builds. This finding does not contradict
anything the report states; it does mean a reader who deletes `cache/` and
rebuilds everything from scratch will not, in general, reproduce every
per-layout number in the paper exactly, even though the *reference itself*
reconstructs correctly to 14 decimal places. Tier 1 (Step 3) and Tier 2
(Step 4) below verify against the *shipped* cached references, as the report
itself does, and this finding is flagged rather than treated as a Tier 0-2
failure.

A secondary, much smaller effect was also observed: the two cold-start runs
against the clone's own fresh reference (373.62816291596255 both times) and
the original-environment run against that same fresh reference file
(373.6114320877784) differ by ~0.017 mHa despite reading the identical
reference file and using the identical seed — smaller by four orders of
magnitude than Finding 4's headline effect, but still outside the declared
1e-9 mHa tolerance, and consistent with environment-dependent floating-point
summation order inside Aer's statevector simulator. Not investigated further
here; noted for completeness.

---

## Prerequisites actually needed, vs. documented

| Prerequisite | Documented? |
|---|---|
| conda / environment.yml packages | Yes |
| `mpirun`/`mpicxx` (openmpi), `llvm-openmp`, a C++ compiler | **No** — Finding 1 |
| Portable link path for `libomp` | **No** — shipped recipe is machine-specific — Finding 2 |
| Avoiding an env-name collision with a pre-existing `sqd` env | **No** — Finding 3 |
| Where to obtain `sbd`'s source | **No** — Finding described above, untested here |
| macOS Accelerate framework for BLAS/LAPACK | Yes (implicitly, via the Configuration file comment) |

## Plain statement of what a reader can reproduce, and at what cost

A reader who clones the repository and follows the README exactly will
**not** get past the first command (`mpirun: command not found`). A reader
who separately discovers and installs `openmpi`/`llvm-openmp`/a C++
compiler, and fixes the hardcoded library path themselves, can build `sbd`
and reproduce the group benchmark exactly (twelve significant figures) in
about 2.5 minutes, and can reconstruct the H10 R=1.6 reference from scratch
in under 30 seconds, matching the cached reference to 1e-14 Ha. That same
reader will **not**, in general, reproduce the paper's per-layout SQD energy
numbers by rebuilding the reference from scratch, even though every
intermediate artefact matches to numerical precision — only by using the
cached reference files this repository ships, exactly as the report's own
methodology does.
