# Usage Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add account-scoped monthly usage summaries to `PlatformClient` and the `biolm usage show` CLI.

**Architecture:** Extend the existing synchronous platform client with query-parameter support and a plain-dictionary usage method. Account selection reuses the account-resolution and same-session context-switch pattern used by API-key creation. The CLI delegates to one client callback, passes JSON through unchanged, and renders a compact summary plus model charges.

**Tech Stack:** Python, httpx, Click, Rich, pytest, httpx.MockTransport

---

## File structure

- `biolm/platform.py`: HTTP query support and account-scoped usage retrieval.
- `biolm/cli/__init__.py`: `usage show` command and table rendering.
- `tests/test_platform_client.py`: fake endpoint and client contract tests.
- `tests/test_cli_platform.py`: CLI registration, validation, JSON, table, and error tests.
- `README.md`, `docs/cli/index.rst`, `docs/cli/usage/workspaces.rst`,
  `docs/sdk/workspaces.rst`: user-facing usage documentation.

### Task 1: Platform client

**Files:**
- Modify: `tests/test_platform_client.py`
- Modify: `biolm/platform.py`

- [ ] **Step 1: Add a fake usage endpoint and failing query test**

Extend `FakeConsole.handler()` to return a response whose scope comes from the
current session and whose selected fields come from `request.url.params`.

```python
if path.endswith("/usage-summary/") and method == "GET":
    params = request.url.params
    return self._json(request, 200, {
        "account_type": session["account_type"],
        "account_id": session["account_id"],
        "selected_year": int(params.get("year", 2026)),
        "selected_month": int(params.get("month", 7)),
        "filter_env_id": int(params["env"]) if params.get("env") else None,
        "current_usage_amount": 12.5,
        "environment_usage_amount": 3.0,
        "environment_label": "prod",
        "env_list": [{"id": 200, "slug": "prod"}],
        "model_charges": [
            {"model_name": "esm2-8m", "total_biolm_charge": 12.5}
        ],
    }, session_id=sid)
```

Add a test that calls the wished-for API and asserts the exact `year`, `month`,
and `env` query parameters.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -p no:pytest_cov tests/test_platform_client.py -k usage_summary -q
```

Expected: failure because `get_usage_summary` does not exist.

- [ ] **Step 3: Add `_request(params=...)` and minimal usage method**

Extend `_request`:

```python
def _request(self, method, path, json=None, params=None):
    ...
    response = self._client.request(
        method_upper, endpoint, json=json, params=params
    )
```

Implement `get_usage_summary()` with:

- positive `year` and `environment_id` validation;
- month range 1–12;
- omission of unset query parameters;
- query keys `year`, `month`, and `env` (map `environment_id` to `env`);
- direct request when `account is None`;
- account resolution, same-session switch, request, and `finally` restoration
  when `account` is set.

- [ ] **Step 4: Add focused edge and scoping tests**

Cover:

- no query parameters;
- invalid year/month/environment ID before any request;
- current organization context;
- named organization and personal accounts;
- original context restored after success;
- original context restored after HTTP failure;
- unknown account error.

- [ ] **Step 5: Run focused client tests**

```bash
pytest -p no:pytest_cov tests/test_platform_client.py -q
```

Expected: all tests pass.

### Task 2: CLI

**Files:**
- Modify: `tests/test_cli_platform.py`
- Modify: `biolm/cli/__init__.py`

- [ ] **Step 1: Write failing command tests**

Add tests for:

- top-level `usage show` registration under Platform;
- default invocation delegates with all optional values `None`;
- `--year`, `--month`, `--environment-id`, and `--account` delegation;
- Click range rejection before client calls;
- JSON response equality;
- table output includes account, month, usage amounts, model names, and charges;
- null model names render as a dash;
- empty model charges print a clear message;
- `PlatformError` becomes a nonzero Click error.

- [ ] **Step 2: Run the tests and verify RED**

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py -k usage -q
```

Expected: failures because the `usage` group is absent.

- [ ] **Step 3: Implement `usage show`**

Register `usage` as a platform group. Use Click integer ranges:

```python
@click.option("--year", type=click.IntRange(min=1))
@click.option("--month", type=click.IntRange(min=1, max=12))
@click.option("--environment-id", type=click.IntRange(min=1))
@click.option("--account")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
```

Call:

```python
data = _platform_request(
    lambda client: client.get_usage_summary(
        year=year,
        month=month,
        environment_id=environment_id,
        account=account,
    )
)
```

For JSON, call `_print_json(data)`. For table output, render a summary table
with currency-neutral headings, then a model table with `Model` and `Charge`.
Use a dash for missing values.

- [ ] **Step 4: Run focused CLI tests**

```bash
pytest -p no:pytest_cov tests/test_cli_platform.py -q
```

Expected: all tests pass.

### Task 3: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/cli/index.rst`
- Modify: `docs/cli/usage/workspaces.rst`
- Modify: `docs/sdk/workspaces.rst`

- [ ] **Step 1: Document the feature**

Add `biolm usage` to the README capability row. Document:

- current-month default;
- year/month and environment filters;
- account selection;
- raw JSON output;
- no live-activity or history-list command.

Add `get_usage_summary` to the SDK autoclass member list.

- [ ] **Step 2: Run focused tests**

```bash
pytest -p no:pytest_cov tests/test_platform_client.py tests/test_cli_platform.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run the full suite**

```bash
pytest -p no:pytest_cov
```

Expected: all tests pass, with only established skips, xfail, and warnings.

- [ ] **Step 4: Review the complete diff**

Confirm:

- no live-activity endpoints were added;
- no currency is invented;
- account switching and retrieval share one session;
- server JSON is unchanged;
- no unrelated files changed.

- [ ] **Step 5: Commit and push**

Commit the implementation and push `feat/platform-workspaces` so draft PR #18
updates automatically.
