# Usage Summary Design

## Purpose

Expose BioLM's canonical monthly billing usage summary through the Python SDK
and CLI. Users must be able to inspect personal or organization spend for a
month, optionally filter by environment, and see charges grouped by model.

## Backend contract

The live console exposes:

```text
GET /console/api/usage-summary/
```

It accepts `year`, `month`, and `env` query parameters. The
`environment_id` parameter is an alias for `env`; the SDK will send `env`.

The response contains:

- effective `account_type`, `account_id`, and `institute_id`
- selected and current year/month fields
- environments available to the effective account
- the accepted environment filter, if any
- account or filtered-environment usage in `current_usage_amount`
- active or filtered environment usage and label
- `model_charges`, ordered by descending charge

The backend owns date fallback rules and scope validation. Invalid, future, or
out-of-range months fall back to the current month. An environment outside the
effective account is ignored. The SDK passes the returned values through
without reinterpreting them.

The endpoint is the source used by the production usage page and has backend
tests for personal and organization isolation, environment ownership, and
model totals. Live activity endpoints expose short-lived cache payloads with
weaker contracts and are outside this feature.

## Python API

Extend `PlatformClient` with:

```python
def get_usage_summary(
    self,
    year: Optional[int] = None,
    month: Optional[int] = None,
    environment_id: Optional[int] = None,
    account: Optional[str] = None,
) -> Dict[str, Any]: ...
```

The method builds query parameters only for supplied values. It validates
`month` as 1 through 12 and requires positive `year` and `environment_id`
values. Validation errors fail before a network request.

Without `account`, the method uses the client's current account context. When
`account` names an organization slug or personal label, the method resolves
that account without enumerating workspaces, switches the same client session,
fetches usage, and restores the original context in `finally`. This mirrors
`create_api_key(account=...)` and prevents a fresh session from silently
returning personal usage.

Extend `_request()` with an optional `params` mapping and pass it to
`httpx.Client.request`. Existing callers remain unchanged.

The method returns the backend dictionary unchanged. A dataclass would freeze
backend field names prematurely and would make JSON parity harder.

## CLI

Add a platform command group:

```text
biolm usage show
    [--year YEAR]
    [--month MONTH]
    [--environment-id ID]
    [--account ACCOUNT]
    [--format table|json]
```

Each CLI invocation creates a fresh `PlatformClient`. Therefore, `show`
without `--account` deterministically uses personal context. Passing
`--account` resolves, switches, fetches, and restores within one
`_platform_request` callback and one client session.

CLI validation uses positive integer ranges for year and environment ID and a
1–12 range for month. The server remains authoritative for historical-window
fallback.

JSON output is the unmodified response. Human-readable output contains:

1. the effective account, selected month, and accepted environment filter;
2. account or filtered usage and environment usage in USD;
3. a model-charge table with model name and USD charge.

Empty `model_charges` produces a clear "No model charges" message. Missing
optional fields render as dashes rather than raising an exception.

## Security and scoping

- Keep account switching and usage retrieval in one client session.
- Trust the response's effective account fields; stale or unauthorized
  organization context may fall back to personal scope on the server.
- Never infer that a requested environment filter was accepted; display
  `filter_env_id` from the response.
- Do not cache usage responses. The backend marks them `no-store`.
- Reuse existing OAuth Bearer and Knox Token authentication.

## Error handling

`PlatformClient` converts HTTP and response failures to `PlatformError` through
its existing request helper. The CLI converts those errors to Click errors via
`_platform_request`.

Client-side range errors use `ValueError`; CLI range errors use Click's normal
validation output. Context restoration runs after success, server errors, and
transport errors.

## Tests

Follow test-driven development:

1. Extend `FakeConsole` with `usage-summary/`, query parsing, and account-aware
   response data.
2. Verify query parameters, omitted values, and client-side ranges.
3. Verify current-context retrieval.
4. Verify named personal and organization account selection and context
   restoration.
5. Verify context restoration after a usage request error.
6. Verify CLI registration, personal default, account selection, range
   validation, JSON passthrough, table output, empty models, and error
   conversion.
7. Run focused platform and CLI tests, then the full suite.

## Documentation

Update the README capability summary, CLI index and usage guide, and SDK
workspace/platform reference. Document monthly semantics, account selection,
environment filters, JSON output, and the absence of a history-list or live
activity command.

## Non-goals

- Live activity rollups, timelines, cache hints, or container telemetry
- Warm-container controls
- Billing-history pagination
- Changing budget validation
- Recalculating or normalizing backend monetary values
- Adding a typed usage dataclass
