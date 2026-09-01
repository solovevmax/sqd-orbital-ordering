.PHONY: verify verify-tier0 verify-tier1 verify-tier2 verify-tier3 test

# Tiers 0 and 1: independent statistics + cached-reference re-derivation.
# No new sampling, no sbd calls beyond what Tier 1's captured-weight check
# needs indirectly via cached data -- a few seconds total.
verify: verify-tier0 verify-tier1

verify-tier0:
	python3 verification/verify_tier0.py

verify-tier1:
	python3 verification/verify_tier1.py

# Tier 2 re-runs the real sampling + sbd pipeline for a declared sample of
# evaluations (~25 minutes, dominated by one Cr2 evaluation at ~16 min).
# Not part of the default `make verify` target -- run explicitly.
verify-tier2:
	python3 verification/verify_tier2.py

# Tier 3 (cold start) needs a git clone, a fresh conda env, and a from-scratch
# sbd build -- see verification/COLD_START.md for what it found and why it
# isn't scripted as a single unattended target.
verify-tier3:
	bash verification/verify_tier3.sh

test:
	pytest tests/
