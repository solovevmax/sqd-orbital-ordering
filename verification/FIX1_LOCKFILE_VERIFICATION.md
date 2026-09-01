# FIX 1 verification: does the lockfile close the reproduction gap?

**Short answer: partially, and not as cleanly as the first test suggested.
I need to report this honestly rather than presenting the one clean result
as the headline.**

## What was done

`environment.lock.txt` (`conda list --explicit`, run in `sqd` — the
environment that built the shipped `cache/h10_R1.6`) and
`environment.lock-pip.txt` (`pip freeze`, same environment) were generated
and committed. Both pin exact package **builds**, not just versions —
critically, `libblas`/`liblapack` build 9 (the specific build identified in
Amendment 3).

## What the first test showed

A brand-new environment (`sqd-lockfile-verify`) built via
`conda create --file environment.lock.txt`, immediately, before installing
anything else:

```
t1L: max abs diff = 0.000e+00   exactly equal = True
t2L: max abs diff = 0.000e+00   exactly equal = True
E_CASCI cached:                  -4.966071088325821
E_CASCI fresh (lockfile env):    -4.966071088325821
```

Bit-for-bit exact. This looked like a clean, complete confirmation that
FIX 1 works.

## What repeat testing showed

Per your instruction to also confirm the SQD-level result (300.32 mHa),
I installed the pinned pip packages (`ffsim`, `qiskit`, etc., needed for
circuit sampling — the reference build itself doesn't need them) into the
same environment and reran. **This second run gave a different reference**
(`E_CASCI = -4.966071088325831`, not `...325821`) and a different SQD
result (373.32 mHa, not 300.32).

Suspecting the `pip install` step had perturbed something `conda list`
doesn't track, I isolated it directly:

1. Re-ran 5 more builds in the *same* `sqd-lockfile-verify` environment
   (no further changes) — all 5 agreed with **each other**
   (`...325831`), not with the cache.
2. Built a **second, brand-new** environment from the identical lockfile
   (`sqd-lockfile-clean`), conda packages only, no pip install at all —
   5/5 builds again gave `...325831`, not the cache.
3. **Re-tested `sqd` itself** — the original environment, unmodified,
   that gave the exact match in Amendment 3 (10/10) and in the very first
   test above — with 8 more independent builds. **All 8 gave
   `...325831`**, not the cached `...325821`.
4. Confirmed via `conda-meta/history` that `sqd`'s package set has not
   changed at any point during this session (last modified 2026-08-19,
   weeks before this session started).

## What this means

The one clean match in the first test was real, but **not reliably
reproducible even in the exact environment that produced it.** The
`sqd` environment now consistently gives a different, self-consistent
result than it did earlier in this same session, with its package state
provably unchanged. This rules out package/BLAS-build drift as the
explanation for *this* discrepancy specifically — that mechanism is real
(Amendment 3's cross-environment comparison still stands and still
explains the `sqd` vs. `sqd-orbital-ordering` gap) — but there is
evidently a **second, distinct source of non-reproducibility**: something
about machine or process state that shifts over the course of a long
session, independent of environment/package pinning. I do not have a
confirmed root cause (candidates: Apple Silicon P-core/E-core scheduling,
ASLR-dependent memory alignment interacting with a BLAS SIMD dispatch
path, thermal throttling after hours of sustained computation) and am not
presenting a guess as a finding.

**Practical implication for the README claim you asked me to add:** I
cannot honestly write "the lockfile is the guarantee" — the evidence
right now says a lockfile is *necessary* (it closes the demonstrated
cross-environment gap) but has been shown, empirically, **not sufficient**
for bit-reproduction of this specific fragile configuration (H10 identity,
default anchors — the same near-degenerate case as the original Finding
4). I've written the README section to state this precisely rather than
oversell it — see the diff. I did not add the "headline of the
verification report" framing you anticipated, because the result doesn't
support it.

## What I did not do

I did not keep digging for the root cause of the session-scale drift —
that would need controlled experiments this environment can't easily run
(reboot, thermal monitoring, core-affinity pinning) and is past what's
proportionate for closing out this audit task. Flagging it as an open
question rather than chasing it further.
