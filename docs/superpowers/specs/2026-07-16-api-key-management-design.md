# API Key Management Design

## Purpose

Add secure API-key creation and deletion to the Python SDK and CLI. The feature
must use the Knox token endpoints that power the BioLM console. It must preserve
the active account context so users can create personal or organization keys.

## Backend contract

The live console exposes two Knox endpoints:

- `POST /console/api/auth/generate_token/` creates a key for the active account
  context and returns `{"token": "<full secret>"}`.
- `DELETE /console/api/auth/delete_token/` accepts
  `{"token": "<full token or eight-character prefix>"}` and returns HTTP 204.

The backend returns the full secret only when it creates a key. It has no REST
endpoint for listing Knox keys. The parallel `/console/api/apikeys/` resource
uses a different token model, exposes plaintext secrets, and does not power the
console. The SDK will not use it.

## Public Python API

Extend `PlatformClient` with:

```python
def create_api_key(self) -> Dict[str, str]: ...
def delete_api_key(self, token_or_prefix: str) -> None: ...
```

Both methods use the client's current session and existing `_request` path.
`create_api_key()` returns the one-time secret response. `delete_api_key()`
validates that its argument is not blank, sends it in the DELETE body, and
returns `None` after a successful 204 response.

Account ownership comes from the active server-side account context. A caller
must use `switch_workspace()` or `set_context()` before creating an
organization key. The token belongs to the account, not to the selected
environment.

The SDK will not add `list_api_keys()` until the backend provides a Knox list
endpoint that returns prefixes and metadata without secrets.

## CLI

Add an `apikey` command group:

```text
biolm apikey create [--workspace ACCOUNT/ENV] [--format table|json]
biolm apikey delete TOKEN_OR_PREFIX [--yes] [--format table|json]
```

`create` optionally switches the client to `--workspace` before creating the
key. Human-readable output states that the secret appears once and must be
stored securely. JSON output contains only the server response.

`delete` asks for confirmation unless the user passes `--yes`. It accepts the
full token or its eight-character prefix. Human-readable and JSON output
confirm deletion without echoing the supplied token.

The top-level help groups `apikey` with other platform commands. The commands
reuse `_platform_request` so authentication and errors match workspace,
organization, and budget commands.

## Security

- Print a full token only in the direct response to `apikey create`.
- Never include a token in list output because no list command exists.
- Never echo a token or prefix after deletion.
- Never write a token to configuration, logs, or documentation examples.
- Keep OAuth Bearer and Knox Token authentication in `CredentialsProvider`.
- Reuse `PlatformClient`'s cookie session and unsafe-method handling.

## Error handling

`PlatformClient` converts HTTP failures to `PlatformError` through its existing
request helper. The CLI converts `PlatformError`, validation errors, and network
errors to Click errors through `_platform_request`.

Blank delete arguments fail before any network request. Cancellation at the
confirmation prompt exits without deleting the key. A missing, unauthorized, or
expired key surfaces the backend's 404 or 403 response.

## Tests

Follow test-driven development:

1. Extend `FakeConsole` with Knox create and delete routes.
2. Verify personal and organization key creation follows account context.
3. Verify creation returns the full token and deletion accepts a prefix.
4. Verify blank deletion fails before a request.
5. Verify CLI registration, workspace switching, one-time token output,
   confirmation, `--yes`, JSON output, and error conversion.
6. Verify help and docs never promise a list command.
7. Run focused platform and CLI tests, then the full suite.

## Documentation

Update the CLI index, workspace or authentication guide, and README capability
summary. Explain that `--workspace` selects ownership, created secrets appear
once, and users may delete keys by full token or eight-character prefix.

## Non-goals

- Listing or naming Knox keys
- Adopting the unused `ApiKey` model
- Changing backend token storage or authorization rules
- Adding usage or activity reporting
- Managing volumes or Jupyter sessions
