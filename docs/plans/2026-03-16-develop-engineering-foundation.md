# RL-ADN Develop Engineering Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn `codex/develop` into a clean, installable, testable engineering branch for `RL-ADN`.

**Architecture:** Keep the current package layout, but harden the repository around it: remove tracked generated artifacts, fix packaging metadata, define a minimal package surface, and make tests robust to missing optional dependencies. This is a foundation pass, not a research or API redesign pass.

**Tech Stack:** Python, setuptools, pytest, pandas, numpy, optional pandapower

---

### Task 1: Baseline Audit

**Files:**
- Create: `docs/plans/2026-03-16-develop-engineering-foundation-design.md`
- Create: `docs/plans/2026-03-16-develop-engineering-foundation.md`
- Modify: none
- Test: repository baseline commands

**Step 1: Record the current repo state**

Run: `git -C E:/科研项目/RL-ADN-develop status --short --branch`
Expected: clean branch state on `codex/develop`

**Step 2: Record the current test failure**

Run: `py -3 -m pytest E:/科研项目/RL-ADN-develop/tests -q`
Expected: collection errors due to missing `pandapower`

**Step 3: Save the design and plan documents**

Expected: both files exist under `docs/plans/`

**Step 4: Commit the planning documents**

```bash
git -C E:/科研项目/RL-ADN-develop add docs/plans/2026-03-16-develop-engineering-foundation-design.md docs/plans/2026-03-16-develop-engineering-foundation.md
git -C E:/科研项目/RL-ADN-develop commit -m "docs: add develop engineering foundation plan"
```

### Task 2: Repository Hygiene

**Files:**
- Modify: `.gitignore`
- Modify: tracked generated files via git index
- Test: `git status --short`

**Step 1: Write the failing expectation**

Expectation: generated artifacts should not appear as tracked source content in `codex/develop`.

**Step 2: Update `.gitignore`**

Add patterns for:

```gitignore
__pycache__/
*.py[cod]
.DS_Store
build/
dist/
*.egg-info/
docs/_build/
```

**Step 3: Remove tracked generated artifacts from the index**

Run commands such as:

```bash
git -C E:/科研项目/RL-ADN-develop rm -r --cached build dist RL_ADN.egg-info docs/_build
```

Also remove tracked `.DS_Store` and tracked `__pycache__` entries from the index if present.

**Step 4: Verify repository hygiene**

Run: `git -C E:/科研项目/RL-ADN-develop status --short`
Expected: only intended source-file edits remain

### Task 3: Packaging Repair

**Files:**
- Modify: `setup.py`
- Modify: `rl_adn/__init__.py`
- Test: import/package smoke command

**Step 1: Write the failing test or smoke expectation**

Expectation: package metadata should use parsed requirements, and `import rl_adn` should expose a version and minimal top-level identity.

**Step 2: Implement the smallest packaging fixes**

In `setup.py`:
- parse `requirements.txt`
- pass parsed requirements into `install_requires`
- keep long description and package data

In `rl_adn/__init__.py`:
- add a package docstring
- add `__version__`

**Step 3: Run packaging smoke checks**

Run:

```bash
py -3 -c "import sys; sys.path.insert(0, r'E:/科研项目/RL-ADN-develop'); import rl_adn; print(rl_adn.__version__)"
```

Expected: prints a version string without import errors

### Task 4: Test Stabilization

**Files:**
- Modify: `tests/123_node_network_powerflow_test.py`
- Modify: `tests/25_node_network_powerflow_test.py`
- Modify: `tests/34_node_network_powerflow_test.py`
- Modify: `tests/69_node_network_powerflow_test.py`
- Create: `tests/test_package_smoke.py`

**Step 1: Write the failing smoke test**

Create a lightweight test that imports `rl_adn` and verifies package version/type.

**Step 2: Make heavy tests optional**

Update the power-flow tests to use:

```python
pytest.importorskip("pandapower")
```

at module level so collection skips cleanly when optional dependencies are missing.

**Step 3: Run targeted tests**

Run:

```bash
py -3 -m pytest E:/科研项目/RL-ADN-develop/tests/test_package_smoke.py -q
py -3 -m pytest E:/科研项目/RL-ADN-develop/tests -q
```

Expected: smoke test passes; power-flow tests skip instead of erroring when `pandapower` is absent

### Task 5: README Quickstart

**Files:**
- Modify: `README.md`

**Step 1: Rewrite the top section minimally**

Keep the paper context, but add:
- install command
- test command
- project structure note

**Step 2: Verify README accuracy**

Check that commands reference real files and current branch behavior.

**Step 3: Run final verification**

Run:

```bash
git -C E:/科研项目/RL-ADN-develop status --short
py -3 -m pytest E:/科研项目/RL-ADN-develop/tests -q
```

Expected: intended file changes only; tests pass/skip cleanly

### Task 6: Commit the Foundation Batch

**Files:**
- Modify: all files changed above

**Step 1: Review diff**

Run: `git -C E:/科研项目/RL-ADN-develop diff --stat`

**Step 2: Commit**

```bash
git -C E:/科研项目/RL-ADN-develop add .gitignore setup.py rl_adn/__init__.py README.md tests docs/plans
git -C E:/科研项目/RL-ADN-develop commit -m "chore: harden develop branch foundation"
```

**Step 3: Report next phase**

After verification, move to usability expansion:
- example cleanup
- runner scripts
- clearer public APIs
