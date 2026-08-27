"""src/sqd_ordering/scores.py

Amplitude-derived, non-oracle scoring functions (score1/score2) and the
Amplitudes dataclass they are computed from. Extracted from
run_ordering_pipeline.py so this logic has a single home (mirroring
mask.py) instead of living inline in the pipeline script alongside
everything else. run_ordering_pipeline.py re-exports these names and
supplies its CFG["anchor_mod"] value via thin wrapper functions, so every
existing call site (internal and in experiments/*.py) is unaffected.

Part A ("score audit", experiments/score_audit.py) found none of the 11
score1/score2 variants predictive of H10 subspace error -- current work
uses retained_J_oppspin (mask.py) instead. Kept here because
build_or_load_h10_reference() still uses these for its hill-climbing
reference-ordering search, and several experiments still import them for
comparison/audit purposes.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from sqd_ordering import mask

K_CHANNELS = 20
L_SPAN_SS = 5
D_ANCHOR_OS = 1


@dataclass
class Amplitudes:
    t1: np.ndarray
    t2: np.ndarray
    nocc: int
    norb: int
    A_ss: np.ndarray = field(init=False)
    A_os_site: np.ndarray = field(init=False)
    channels_ss: list = field(init=False)
    channels_os: list = field(init=False)

    def __post_init__(self):
        nocc, norb = self.nocc, self.norb
        nvir = norb - nocc
        t2 = np.asarray(self.t2)
        assert t2.shape == (nocc, nocc, nvir, nvir), t2.shape
        A_ss = np.zeros((norb, norb))
        A_os_site = np.zeros(norb)
        css, cos = [], []
        for i, j, a, b in itertools.product(range(nocc), range(nocc),
                                            range(nvir), range(nvir)):
            ga, gb = nocc + a, nocc + b
            w_os = abs(float(t2[i, j, a, b]))
            if w_os > 0:
                uniq = sorted({i, j, ga, gb})
                for p in uniq:
                    A_os_site[p] += w_os
                cos.append((w_os, tuple(uniq)))
            if i < j and a < b:
                w_ss = abs(float(t2[i, j, a, b] - t2[i, j, b, a]))
                if w_ss > 0:
                    uniq = sorted({i, j, ga, gb})
                    for p, q in itertools.combinations(uniq, 2):
                        A_ss[p, q] += w_ss
                        A_ss[q, p] += w_ss
                    css.append((w_ss, tuple(uniq)))
        for i in range(nocc):
            for a in range(nvir):
                w = abs(float(self.t1[i, a]))
                A_ss[i, nocc + a] += w
                A_ss[nocc + a, i] += w
        self.A_ss, self.A_os_site = A_ss, A_os_site
        self.channels_ss = sorted(css, key=lambda c: -c[0])[:K_CHANNELS]
        self.channels_os = sorted(cos, key=lambda c: -c[0])[:K_CHANNELS]


def score1(pos, amp, J_aa, J_ab, w_ss, anchor_orbitals=None, anchor_mod=4):
    ssp = mask.same_spin_pairs(pos, amp.norb)
    oss = sorted({p for p, _ in mask.opp_spin_pairs(
        pos, amp.norb, anchor_mod=anchor_mod, anchor_offset=0, anchor_orbitals=anchor_orbitals)})
    iu = np.triu_indices(amp.norb, k=1)

    tot = amp.A_ss[iu].sum()
    s_ss = (sum(amp.A_ss[p, q] for p, q in ssp) / tot) if tot > 0 else 0.0
    tot = amp.A_os_site.sum()
    s_os = (sum(amp.A_os_site[p] for p in oss) / tot) if tot > 0 else 0.0

    M_ss = np.abs(J_aa).sum(axis=0) * amp.A_ss
    M_os = np.abs(J_ab).sum(axis=0).diagonal() * amp.A_os_site
    tot2 = M_ss[iu].sum()
    s_ss2 = (sum(M_ss[p, q] for p, q in ssp) / tot2) if tot2 > 0 else 0.0
    tot2 = M_os.sum()
    s_os2 = (sum(M_os[p] for p in oss) / tot2) if tot2 > 0 else 0.0

    return dict(s1_amp=w_ss * s_ss + (1 - w_ss) * s_os,
                s1_amp_ss=s_ss, s1_amp_os=s_os,
                s1_ampJ=w_ss * s_ss2 + (1 - w_ss) * s_os2,
                s1_ampJ_ss=s_ss2, s1_ampJ_os=s_os2)


def _span(pos, orbs):
    ps = [pos[o] for o in orbs]
    return int(max(ps) - min(ps))


def _anchor_dist(pos, orbs, anchor_mod=4):
    m = anchor_mod
    return min(min(abs(int(pos[o]) - a) for a in range(0, len(pos), m))
               for o in orbs)


def score2(pos, amp, w_ss, L=L_SPAN_SS, D=D_ANCHOR_OS, anchor_mod=4):
    def frac(ch, test):
        tot = sum(w for w, _ in ch)
        return (sum(w for w, o in ch if test(o)) / tot) if tot > 0 else 0.0
    r_ss = frac(amp.channels_ss, lambda o: _span(pos, o) <= L)
    r_os = frac(amp.channels_os, lambda o: _anchor_dist(pos, o, anchor_mod=anchor_mod) <= D)
    tot = sum(w for w, _ in amp.channels_ss) or 1.0
    soft = sum(w * np.exp(-max(0, _span(pos, o) - 3) / 2.0)
               for w, o in amp.channels_ss) / tot
    return dict(s2=w_ss * r_ss + (1 - w_ss) * r_os,
                s2_ss=r_ss, s2_os=r_os, s2_soft_ss=soft)
