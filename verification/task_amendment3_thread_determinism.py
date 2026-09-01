"""Amendment 3: is the H10 R=1.6 reference's CCSD amplitude noise
thread-related (and thus fixed by the existing OMP/MKL/OPENBLAS=1 pinning
protocol) or irreducible across independent process invocations?

Single-run worker: build the H10 R=1.6 reference from scratch (unique,
throwaway cachedir every invocation, so this is a genuine independent
build, never a cache hit) and dump t1L/t2L to the path given as argv[1].
Invoked as a fresh `python3` subprocess once per run by the driver shell
script, exactly mirroring how the original Finding 4 comparison was made
(separate process invocations), not repeated calls within one process.
"""
import sys, pickle, uuid
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
import run_ordering_pipeline as R

out_path = sys.argv[1]
cachedir = f"/tmp/h10_ref_amendment3_{uuid.uuid4().hex}"
ref = R.build_or_load_h10_reference(1.6, 10, "sto-6g", cachedir=cachedir)
with open(out_path, "wb") as f:
    pickle.dump(dict(t1L=ref["t1L"], t2L=ref["t2L"], E_CASCI=ref["E_CASCI"]), f)
print(f"saved {out_path}  E_CASCI={ref['E_CASCI']!r}")
