"""src/sqd_ordering/sampling.py

Aer sampling and top-k determinant selection. Extracted from
run_ordering_pipeline.py; the transpiler-related settings (seed, PRE_INIT)
are explicit parameters here instead of read from the pipeline's CFG dict,
so this module has no dependency on run_ordering_pipeline.py.
run_ordering_pipeline.py re-exports these names via thin wrappers that
supply CFG's values, so existing call sites are unaffected.
"""
from __future__ import annotations


def hf_bitstring(norb, nocc):
    return "0" * (norb - nocc) + "1" * nocc      # rightmost bit = orbital 1


def build_circuit(op, norb, nelec):
    """HF state prep + masked LUCJ operator + measure_all, untranspiled.
    The single circuit-construction path used everywhere in this project --
    by sample_bitstrings below, and (imported directly) by
    experiments/transpilation_audit.py, so an audit of transpiled resource
    cost is guaranteed to be transpiling the exact circuit the sampling
    pipeline runs, not a separately-built lookalike.
    """
    import ffsim.qiskit as fq
    from qiskit import QuantumCircuit, QuantumRegister

    qr = QuantumRegister(2 * norb, "q")
    qc = QuantumCircuit(qr)
    qc.append(fq.PrepareHartreeFockJW(norb, nelec), qr)
    qc.append(fq.UCJOpSpinBalancedJW(op), qr)
    qc.measure_all()
    return qc


def sample_bitstrings(op, norb, nelec, shots, seed, *, seed_transpiler=1234, use_pre_init=True):
    """Fresh AerSimulator per call. Returns (alpha_counts, beta_counts)."""
    import ffsim.qiskit as fq
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = build_circuit(op, norb, nelec)
    sim = AerSimulator(seed_simulator=seed)
    tkw = dict(seed_transpiler=seed_transpiler, optimization_level=1)
    if use_pre_init:
        tkw["pre_init"] = fq.PRE_INIT
    try:
        tqc = transpile(qc, sim, **tkw)
    except TypeError:
        tkw.pop("pre_init", None)
        tqc = transpile(qc, sim, **tkw)
    counts = sim.run(tqc, shots=shots).result().get_counts()

    a, b = {}, {}
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        # validated 'split' transform: alpha = rightmost norb, beta = leftmost
        a[bits[norb:]] = a.get(bits[norb:], 0) + n
        b[bits[:norb]] = b.get(bits[:norb], 0) + n
    return a, b, int(tqc.depth())


def top_dets(counts, k, hf):
    """Top-k by marginal count, HF forced in (sbd needs it present)."""
    ranked = [s for s, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    if hf in ranked:
        ranked.remove(hf)
    sel = [hf] + ranked[: k - 1]
    return sel, len(counts)
