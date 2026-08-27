"""src/sqd_ordering/sbd.py

Thin wrapper around the external sbd binary (not modified by this
project -- see https://github.com/r-ccs-cms/sbd). Extracted from
run_ordering_pipeline.py; sbd invocation settings (binary path, mpirun,
method/extra args, timeout) are explicit parameters here instead of read
from the pipeline's CFG dict, so this module has no dependency on
run_ordering_pipeline.py. run_ordering_pipeline.py re-exports these names
via a thin wrapper that supplies CFG's values, so existing call sites are
unaffected.
"""
from __future__ import annotations

import re
import subprocess
import sys


def parse_sbd_energy(text):
    """Extract the final total energy from sbd stdout.

    ADAPT HERE if your build prints a different final line. Returns None on
    failure so the caller can dump the tail rather than guess.
    """
    pats = [
        r"diagonalization:\s*Energy\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        r"[Ff]inal\s+[Ee]nergy\s*[:=]\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
            r"[Tt]otal\s+[Ee]nergy\s*[:=]\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
            r"\bE\s*=\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
            r"[Ee]nergy\s*[:=]?\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)"]
    for p in pats:
        m = re.findall(p, text)
        if m:
            return float(m[-1])
    return None


def run_sbd(fcidump, adet, bdet, norb, *, sbd_bin, mpirun="mpirun",
            method_args=(), extra=(), timeout=900):
    cmd = [mpirun, "-n", "1", sbd_bin,
           "--fcidump", fcidump, "--adetfile", adet, "--bdetfile", bdet,
           "--bit_length", str(max(20, norb))]
    cmd += list(method_args) + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    e = parse_sbd_energy(p.stdout + "\n" + p.stderr)
    if e is None:
        print("---- sbd stdout tail ----")
        print("\n".join((p.stdout + p.stderr).splitlines()[-25:]))
        sys.exit("FATAL: could not parse an energy from sbd output. "
                 "Adapt parse_sbd_energy().")
    return e
