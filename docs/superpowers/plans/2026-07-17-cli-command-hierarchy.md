# CLI Command Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the backend-oriented CLI layout with canonical account and workspace commands, hidden compatibility aliases, identity reporting, and name-based organization commands.

**Architecture:** Extend `PlatformClient` with cross-prefix identity retrieval and organization identifier resolution. Refactor the existing Click commands in place: canonical command objects live under `account`, hidden aliases reuse their leaf callbacks and parameters, and `RichGroup` renders only direct visible children. Keep `status` independent of the fatal platform-command wrapper so diagnostics degrade gracefully.

**Tech Stack:** Python, Click, Rich, httpx, pytest, `httpx.MockTransport`

---

## File structure

- Modify `biolm/platform.py`: identity request and organization identifier resolution.
- Modify `biolm/cli/__init__.py`: help rendering, canonical groups, aliases, `whoami`, and status context.
- Modify `tests/test_platform_client.py`: mock identity route and client behavior.
- Modify `tests/test_cli_help.py`: help rendering and final hierarchy assertions.
- Modify `tests/test_cli_platform.py`: migrate existing registration/path tests; add canonical, alias, identity, and status coverage.
- Modify `tests/test_cli_login.py`: canonical login/logout coverage while retaining alias coverage.
- Modify user-facing docs:
  - `README.md`
  - `docs/index.rst`
  - `docs/cli/index.rst`
  - `docs/cli/usage/workspaces.rst`
  - `docs/cli/login.rst`
  - `docs/cli/logout.rst`
  - `docs/cli/status.rst`
  - `docs/cli/hub.rst`
  - `docs/cli/usage/protocols.rst`
  - `docs/sdk/workspaces.rst`
  - `docs/sdk/finetune.rst`
  - `docs/guide/authentication.rst`
  - `docs/guide/quickstart.rst`
  - `docs/guide/managing-datasets.rst`
  - `docs/guide/protocol-workflows.rst`
  - `docs/guide/biolm-hub.rst`
  - `docs/guide/pipeline-workflows.rst`
- Create `docs/cli/account.rst` and `docs/cli/whoami.rst` for the existing per-command Sphinx page pattern.
- Leave historical design/plan docs under `docs/superpowers/` unchanged except when they are this plan or the approved hierarchy design.

### Task 1: Add identity and organization resolution to `PlatformClient`

**Files:**
- Modify: `tests/test_platform_client.py`
- Modify: `biolm/platform.py`

- [ ] **Step 1: Extend the fake API and write failing identity tests**

Add a FakeConsole route for `GET /api/users/me/` that returns representative identity fields plus unrelated billing fields. Test that:

```python
def test_get_current_user_uses_absolute_api_url(client, fake):
    identity = client.get_current_user()
    assert identity["username"] == "astewart"
    assert fake.requests[-1][1] == "/api/users/me/"
```

Also assert API errors become `PlatformError`. Do not rely on a failing `_url` absolute-path assertion: current `_url()` already leaves absolute URLs unchanged. Absolute URL passthrough remains an implementation requirement for clarity and future-proofing, but RED comes from the missing method and fake route.

- [ ] **Step 2: Run identity tests and confirm RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_platform_client.py -k "current_user" -q
```

Expected: `AttributeError` / missing `get_current_user()`.

- [ ] **Step 3: Implement `get_current_user()` with absolute URL handling**

Implement:

```python
def _url(self, path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return path.lstrip("/")

def get_current_user(self) -> Dict[str, Any]:
    return self._request("GET", "{}/api/users/me/".format(self._origin))
```

- [ ] **Step 4: Run identity tests and confirm GREEN**

Run the focused command from Step 2. Expected: PASS.

- [ ] **Step 5: Write failing organization resolution tests**

Cover exact slug, exact name, numeric `int`, all-digit `str`, missing identifiers, and a name/slug cross-collision:

```python
assert client.get_organization("Acme")["id"] == 20
assert client.get_organization("acme")["id"] == 20
assert client.get_organization(20)["id"] == 20
assert client.get_organization("20")["id"] == 20
```

Assert `invite_to_organization("Acme", ...)` records a request whose path ends with `/orgs/20/invite/` (FakeConsole stores `/console/api/orgs/20/invite/`). Assert ambiguous and missing values raise clear `PlatformError` messages without sending the mutation.

RED expectations:

- name/slug inputs currently fail with `ValueError` from `int(...)`;
- all-digit `"20"` already works via `int("20")`, so treat it as a GREEN regression/compatibility case once the resolver exists, not as the RED proof;
- preference and ambiguity cases must fail until list-then-resolve exists.

- [ ] **Step 6: Run organization tests and confirm RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_platform_client.py -k "organization_identifier or organization_resolution" -q
```

Expected: name/slug cases fail; ambiguity/missing cases fail; pure all-digit ID may already pass.

- [ ] **Step 7: Implement one organization resolver used by show and invite**

Add a private resolver that:

1. accepts `Union[int, str]`;
2. lists organizations;
3. prefers an exact ID match for `int` and all-digit strings against the caller's org list;
4. collects exact name and slug matches;
5. deduplicates matches by ID;
6. raises on zero or multiple matches;
7. returns the numeric ID.

Update `get_organization()` and `invite_to_organization()` to call it. Keep `create_organization()` in the SDK.

- [ ] **Step 8: Run all platform client tests**

Run:

```bash
pytest -p no:pytest_cov tests/test_platform_client.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit the client slice**

```bash
git add biolm/platform.py tests/test_platform_client.py
git commit -m "feat: resolve platform identity and organizations"
```

### Task 2: Make CLI help hierarchy-aware and alias-safe

**Files:**
- Modify: `tests/test_cli_help.py`
- Modify: `biolm/cli/__init__.py`

Important sequencing constraint: do **not** assert the final Account/`account`/`whoami` menu here. Those commands do not exist until Tasks 3–4. Task 2 only lands help mechanics that remain valid against the current command tree and stay compatible with later hierarchy work.

- [ ] **Step 1: Write failing help-mechanics tests**

Assert:

- hidden commands are omitted from help;
- top-level help no longer expands group children into strings like `workspace list`;
- top-level help no longer uses the `Platform` section heading;
- nested group help still lists direct visible children.

Use a temporary hidden command registration in the test, or mark an existing command hidden briefly in a fixture, to prove `cmd.hidden` filtering before aliases exist.

- [ ] **Step 2: Run help-mechanics tests and confirm RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_help.py -q
```

Expected: current `RichGroup` still expands children, ignores `hidden`, and uses `Platform`.

- [ ] **Step 3: Update `RichGroup.format_help()`**

Skip commands with `cmd.hidden`. Append only `(name, cmd)` instead of expanding `click.Group.commands`. Replace the `Platform` section with preparatory section names that can accept later `account`/`whoami` membership without flattening children. Nested groups use a single `Commands` panel of visible direct children.

- [ ] **Step 4: Add a leaf-only hidden-alias helper**

Never `_hidden_alias` a `Group`. `copy.copy(group).commands is group.commands` on Click 8.1.8, so mutating an alias group would mutate the canonical group.

Use:

```python
def _hidden_leaf_alias(parent, name, target):
    alias = copy.copy(target)
    alias.name = name
    alias.hidden = True
    parent.add_command(alias, name)
    return alias
```

For old multi-level paths such as `usage show`, create a separate `RichGroup(hidden=True)` and register copied leaf commands under it. Prefer shared helper functions for budget show behavior rather than copying `invoke_without_command` group callbacks.

- [ ] **Step 5: Hide `biolm version`**

Keep `@click.version_option` unchanged. Mark only the command-form version alias hidden. Confirm it remains callable and absent from help.

- [ ] **Step 6: Run help-mechanics tests and confirm GREEN**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_help.py -q
```

Expected: PASS for mechanics only. Existing flattened registration tests in `test_cli_platform.py` may still expect old help strings; leave them until Task 3 migrates them.

- [ ] **Step 7: Commit the help-mechanics slice**

```bash
git add biolm/cli/__init__.py tests/test_cli_help.py
git commit -m "refactor: make CLI help hierarchy-aware"
```

### Task 3: Build canonical `account` commands and hidden compatibility paths

**Files:**
- Modify: `tests/test_cli_platform.py`
- Modify: `tests/test_cli_login.py`
- Modify: `tests/test_cli_help.py`
- Modify: `biolm/cli/__init__.py`

- [ ] **Step 1: Migrate existing CLI tests before adding new ones**

Rewrite current expectations that will break:

- `test_platform_groups_and_commands_are_registered`
- `test_org_show_create_and_invite`
- `test_apikey_*` registration/help expectations
- `test_usage_*` registration/help expectations
- `test_cli_help` Authentication/`login` expectations

Replace them with canonical-path coverage and hidden-alias coverage. Explicitly assert both:

```text
biolm org create ...
biolm account org create ...
```

fail with “No such command.” Keep old invoke paths such as `usage show`, `budget show`, and `apikey create` as alias tests so option parity stays covered.

- [ ] **Step 2: Write failing canonical-path tests**

Add coverage for:

```text
biolm account login
biolm account logout
biolm account usage
biolm account budget
biolm account budget set
biolm account api-key create
biolm account api-key delete
biolm account org list
biolm account org show "Acme Labs"
biolm account org show 20
biolm account org invite acme person@example.com
biolm org show 20
```

Assert `account usage` delegates with existing options, `account budget` performs the existing show behavior, org callbacks receive string identifiers (including `"20"` with no `click.INT` coercion), aliases emit no deprecation warnings, and create remains unavailable.

- [ ] **Step 3: Run migrated/canonical tests and confirm RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py tests/test_cli_login.py tests/test_cli_help.py -k "account or alias or org_create or registered or help" -q
```

Expected: no `account` group exists; create-removal and alias assertions fail until registration.

- [ ] **Step 4: Register canonical commands**

Create `account = RichGroup(...)` under `cli`. Move or register:

- login/logout callbacks under `account`;
- usage as a leaf command;
- budget as `invoke_without_command=True`, with group-level `--format` for show and a `set` child;
- `api-key` as a nested group;
- org as a nested group without create.

Keep rendering and business logic in shared helper functions so leaf aliases can reuse callbacks safely.

- [ ] **Step 5: Register compatibility aliases**

Preserve:

- top-level hidden `login` and `logout` leaf aliases;
- hidden `usage show`;
- hidden `budget show|set` via a separate hidden budget group plus leaf aliases;
- hidden `apikey create|delete`;
- hidden `org list|show|invite`;
- hidden `version`.

Do not register `org create` anywhere.

- [ ] **Step 6: Update org arguments and docstrings**

Change Click arguments from `type=click.INT` to strings named `organization`. Pass them to the updated client. Document name or slug, while numeric script inputs continue to work through the client resolver.

- [ ] **Step 7: Add final help hierarchy assertions now that commands exist**

Assert top-level help includes Account, Workspace, Hub, Models, Protocols, and Datasets; includes direct names `status`, `account`, and `workspace`; and excludes `Platform`, `version`, `login`, `usage show`, `org create`, and expanded strings such as `workspace list`. Leave `whoami` final help membership assertions for Task 4 if that command is not yet registered.

Assert `biolm account --help` shows only its direct children and no hidden aliases.

- [ ] **Step 8: Run canonical, alias, and help tests**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py tests/test_cli_login.py tests/test_cli_help.py -q
```

Expected: PASS for both canonical and hidden legacy paths, including no deprecation warning text in alias output.

- [ ] **Step 9: Commit the account hierarchy slice**

```bash
git add biolm/cli/__init__.py tests/test_cli_platform.py tests/test_cli_login.py tests/test_cli_help.py
git commit -m "feat: organize account CLI commands"
```

### Task 4: Add `whoami` and enrich top-level `status`

**Files:**
- Modify: `tests/test_cli_platform.py`
- Modify: `tests/test_cli_help.py`
- Modify: `biolm/cli/__init__.py`

- [ ] **Step 1: Write failing `whoami` tests**

Mock one `PlatformClient` context whose `get_current_user()` returns identity plus billing fields, and assert:

- organization JSON returns only the stable identity/context allowlist;
- personal JSON sets `account_name` to `None` and `account_slug` to username;
- human output shows principal and active context;
- authentication/API failure exits nonzero without exposing unrelated user fields.

- [ ] **Step 2: Run `whoami` tests and confirm RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py -k whoami -q
```

Expected: “No such command 'whoami'.”

- [ ] **Step 3: Implement identity composition and rendering**

Add a helper that whitelists:

```python
{
    "id", "username", "email", "first_name", "last_name",
    "account_type", "account_id", "account_name", "account_slug",
    "environment_id",
}
```

Fetch `get_current_user()` and `get_context()` in one `_platform_request` callback. Render stable JSON or a Rich table. Register top-level `whoami` and update help assertions to include it under Account.

- [ ] **Step 4: Write failing status degradation tests**

Patch `PlatformClient` to:

- raise during construction (logged out);
- return a current workspace;
- raise during `current_workspace()`.

In every case, assert configured endpoints remain visible, account/workspace appears when available, and an unavailable marker replaces failed context without a nonzero exit. Do not assert `__enter__`/`_platform_request` usage for status; status constructs and closes the client manually.

- [ ] **Step 5: Run status tests and confirm RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py -k "status and not hub" -q
```

Expected: no active workspace row and no graceful platform probe behavior.

- [ ] **Step 6: Implement best-effort status context**

Do not use `_platform_request`. Print the existing configuration first, then construct and close `PlatformClient` manually inside `try/finally`. Catch `PlatformError` around construction and `current_workspace()` and render unavailable context. Preserve existing auth validation output without turning an optional context failure into a fatal exit.

- [ ] **Step 7: Run identity, status, and final help tests**

Run:

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py tests/test_cli_help.py -k "whoami or status or help" -q
```

Expected: PASS.

- [ ] **Step 8: Commit diagnostics**

```bash
git add biolm/cli/__init__.py tests/test_cli_platform.py tests/test_cli_help.py
git commit -m "feat: add identity and connection diagnostics"
```

### Task 5: Update canonical CLI and SDK documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/index.rst`
- Modify: `docs/cli/index.rst`
- Modify: `docs/cli/usage/workspaces.rst`
- Modify: `docs/cli/login.rst`
- Modify: `docs/cli/logout.rst`
- Modify: `docs/cli/status.rst`
- Modify: `docs/cli/hub.rst`
- Modify: `docs/cli/usage/protocols.rst`
- Modify: `docs/sdk/workspaces.rst`
- Modify: `docs/sdk/finetune.rst`
- Modify: `docs/guide/authentication.rst`
- Modify: `docs/guide/quickstart.rst`
- Modify: `docs/guide/managing-datasets.rst`
- Modify: `docs/guide/protocol-workflows.rst`
- Modify: `docs/guide/biolm-hub.rst`
- Modify: `docs/guide/pipeline-workflows.rst`
- Create: `docs/cli/account.rst`
- Create: `docs/cli/whoami.rst`

- [ ] **Step 1: Search documentation for legacy public paths**

Run:

```bash
rg -n "biolm (login|logout|usage show|budget|apikey|org |version)" README.md docs
```

Update every non-historical user-facing match. Leave historical design/plan artifacts under `docs/superpowers/` alone unless they are this plan or the approved hierarchy design.

- [ ] **Step 2: Update user-facing examples**

Use only canonical paths:

- `biolm account login|logout`;
- `biolm account usage`;
- `biolm account budget [set]`;
- `biolm account api-key create|delete`;
- `biolm account org list|show|invite`;
- `biolm status`, `biolm whoami`, and `biolm --version`.

Document organization input as an exact name or slug and omit organization creation. Mention that short aliases such as `biolm login` still work, but do not make aliases the documented primary syntax.

- [ ] **Step 3: Update Sphinx Click targets/toctrees**

Create `docs/cli/account.rst` and `docs/cli/whoami.rst`. Retarget `login.rst` / `logout.rst` from `biolm.cli:login` / `biolm.cli:logout` to the canonical `account` children, and update `:prog:` lines accordingly. Add `cli/account` and `cli/whoami` to the CLI toctree in `docs/index.rst` (the actual Sphinx toctree; `docs/cli/index.rst` is `:orphan:` and only holds list-tables). Update list-table / click-target copy in `docs/cli/index.rst`. Ensure hidden compatibility commands are not intentionally documented.

- [ ] **Step 4: Document new SDK methods**

Add `get_current_user`, name/slug organization resolution, and retained numeric compatibility to `docs/sdk/workspaces.rst`.

- [ ] **Step 5: Verify no user-facing legacy syntax remains**

Run the search from Step 1. Expected: matches only in historical design/plan artifacts or explicit compatibility notes.

- [ ] **Step 6: Build documentation**

Run:

```bash
make docs
```

Expected: PASS. If docs dependencies are missing, record that separately and do not invent a substitute Sphinx command unless `make docs` is unavailable.

- [ ] **Step 7: Commit documentation**

```bash
git add README.md docs
git commit -m "docs: document task-oriented CLI commands"
```

### Task 6: Full verification and review

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run focused tests**

```bash
pytest -p no:pytest_cov tests/test_platform_client.py tests/test_cli_platform.py tests/test_cli_help.py tests/test_cli_login.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete test suite**

```bash
pytest -p no:pytest_cov -q
```

Expected: PASS. Distinguish pre-existing environmental/plugin failures from regressions.

- [ ] **Step 3: Exercise command help manually**

```bash
biolm --help
biolm account --help
biolm account org --help
biolm --version
biolm version
```

Expected: concise canonical help, no Platform category or hidden aliases, no org create, and both version forms work.

- [ ] **Step 4: Check lints and inspect the final diff**

Read IDE diagnostics for changed Python files. Run:

```bash
git status --short
git diff --check
git diff origin/feat/platform-workspaces...HEAD
```

- [ ] **Step 5: Request code review**

Dispatch a code reviewer against the approved design and this plan. Fix any correctness, security, compatibility, or test gaps and rerun focused/full verification.

- [ ] **Step 6: Push the branch and confirm PR state**

```bash
git push origin feat/platform-workspaces
gh pr view 18 --json url,state,isDraft,headRefName
```

Expected: PR #18 contains the new commits and remains available for further work.
