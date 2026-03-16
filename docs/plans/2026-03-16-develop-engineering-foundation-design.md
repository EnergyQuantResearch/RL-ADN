# RL-ADN Develop Engineering Foundation Design

## Context

`codex/develop` should become the long-lived engineering branch for `RL-ADN`, separate from paper-specific or experimental research branches. The current branch state is closer to a paper artifact than a maintainable Python library:

- repository artifacts are tracked (`build/`, `dist/`, `RL_ADN.egg-info/`, `docs/_build/`, `__pycache__/`, `.DS_Store`)
- packaging is inconsistent because [`setup.py`](/E:/科研项目/RL-ADN-develop/setup.py) reads `requirements.txt` but does not pass the parsed dependencies to `install_requires`
- the public package surface is not defined in [`rl_adn/__init__.py`](/E:/科研项目/RL-ADN-develop/rl_adn/__init__.py)
- tests depend on optional heavy packages without graceful skipping and currently fail at collection when `pandapower` is missing
- README and docs describe the paper but do not provide a clear install-test-quickstart path for library users

## Goal

Stabilize `codex/develop` as an installable, testable, clean Python library branch before adding new usability features or research workflows.

## Scope For Phase 1

Phase 1 focuses on engineering foundation only:

1. repository hygiene and ignore policy
2. packaging and dependency correctness
3. test collection stability and lightweight smoke coverage
4. minimal README/quickstart improvements

This phase does not redesign algorithms, add new research features, or restructure the entire API surface.

## Approaches Considered

### Approach 1: Minimal hardening

Only fix `.gitignore`, `setup.py`, and a few brittle tests.

Pros:
- fastest path
- low risk

Cons:
- leaves unclear package surface and weak onboarding
- likely causes repeated cleanup work in later phases

### Approach 2: Foundation hardening with testable install path

Clean tracked artifacts, repair packaging, define a minimal package entry surface, and make tests behave predictably with optional dependencies.

Pros:
- establishes a stable base for future improvements
- keeps scope contained to engineering essentials
- matches the branch purpose best

Cons:
- slightly more work up front

### Approach 3: Full library restructure now

Reorganize modules, rename packages, rewrite docs, and redesign CLI/examples immediately.

Pros:
- ambitious cleanup

Cons:
- too much scope for a first develop-branch pass
- high regression risk
- mixes strategic refactor with basic hygiene

## Decision

Use **Approach 2**.

This branch first needs a credible library foundation: clean repository state, correct packaging metadata, deterministic test behavior, and a short install/verification path. Once that exists, later phases can expand examples, training/evaluation runners, and public APIs without carrying obvious hygiene debt.

## Design

### Repository hygiene

Ignore generated build artifacts and local machine files. Remove tracked generated files from the index where appropriate on `codex/develop`, while preserving source data and docs sources.

### Packaging

Keep `setuptools` packaging for now, but make it internally consistent:

- parse `requirements.txt` once and feed the result to `install_requires`
- keep package data for bundled network/time-series CSV files
- expose package metadata cleanly

No migration to `pyproject.toml` in this phase; that can be a later improvement once the current package is stable.

### Test strategy

Split expectations into two tiers:

- lightweight smoke tests that always run without heavy optional power-system dependencies
- power-flow tests that skip cleanly when `pandapower` is unavailable

This keeps `pytest` collection stable on a fresh environment while preserving domain-specific validation when optional dependencies are installed.

### Documentation

Keep README concise but practical:

- what the library is
- how to install dependencies
- how to run tests
- where to look for examples and docs

This is a develop-branch foundation pass, not a full documentation rewrite.

## Success Criteria

Phase 1 is successful when:

- `git status` is clean except for intended source changes
- generated artifacts are ignored instead of tracked
- package installation metadata is internally consistent
- `pytest` can collect and run without hard import failures in a fresh environment
- README gives a minimal install and verification path

## Deferred Work

These are intentionally deferred to later phases:

- broader API redesign
- notebook/tutorial overhaul
- experiment runners and benchmark CLI
- CI workflow addition
- research-specific meta-agent or agentic routing work
