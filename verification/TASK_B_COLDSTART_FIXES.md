# Task B — cold-start defect fixes

All three defects from `COLD_START.md` fixed and re-verified end to end from
a genuinely isolated environment (fresh clone, fresh conda env built only
from the fixed `environment.yml`, fully isolated `PATH`).

## Fix 1 — `environment.yml` now provides the `sbd` build toolchain

Added `openmpi`, `llvm-openmp`, and conda-forge's `compilers` metapackage
(portable across platforms — resolves to `clangxx_osx-arm64` on Apple
Silicon, the Linux equivalent elsewhere, unlike hardcoding one compiler
package name).

Verified with the same isolated-`PATH` test that originally caught the gap:

```
$ env -i HOME=$HOME PATH=".../envs/sqd-orbital-ordering/bin:/usr/bin:/bin" bash -c \
    'which mpirun; echo $?; which mpicxx; echo $?'
/Users/.../envs/sqd-orbital-ordering/bin/mpirun
mpirun exit:0
/Users/.../envs/sqd-orbital-ordering/bin/mpicxx
mpicxx exit:0
```

## Fix 2 — `environment.yml` renamed to avoid collision

`name: sqd` → `name: sqd-orbital-ordering`. Updated the one place the old
name was referenced: `README.md`'s benchmark command
(`conda activate sqd` → `conda activate sqd-orbital-ordering`). Grepped the
whole repo (README, environment.yml, the `.tex` report, the presentation
text extracted via `python-pptx`) for other references to the old name or
to `envs/sqd` paths — none found.

## Fix 3 — `Configuration.macos-arm64` no longer hardcodes a path

```diff
-SYSLIB = -L/Users/maxim/miniforge3/envs/sqd/lib -lomp -framework Accelerate
+SYSLIB = -lomp -framework Accelerate
```

matching the finding from `COLD_START.md` that this line was inert at
runtime (the actual `libomp` rpath comes from `mpicxx`'s own
environment-relative injection) and unnecessary at link time (`mpicxx`
already searches its own environment's `lib/`).

## Full re-verification, isolated environment

1. Fresh `git clone` (uncommitted working-tree fixes copied in manually for
   this check, since they aren't committed yet — see below).
2. `conda env create -f environment.yml` from the fixed file — succeeds,
   ~24s (warm local package cache; a first-time download will take longer).
3. `sbd` built with the fixed `Configuration.macos-arm64`, fully isolated
   `PATH` (only the new env's `bin/`, plus `/usr/bin:/bin`) — succeeds,
   6.0s.
4. README benchmark command, same isolated environment:

   ```
   Energy = -109.0483526946501
   Sample-based diagonalization: Energy = -109.0483526946501
   ```

   **Exact match**, in a build that used none of this machine's pre-existing
   `sqd` environment or its `sbd` binary anywhere in the chain.

All three fixes are currently uncommitted in the working tree
(`environment.yml`, `README.md`, `notes/sbd-build-notes/Configuration.macos-arm64`)
pending your review of Tasks A/B/C together before committing, per your
instructions.
