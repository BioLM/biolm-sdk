# CLI Command Hierarchy Design

## Purpose

Organize the CLI around user tasks instead of backend architecture. Account
operations belong under `account`; workspace operations remain a top-level
group. Common diagnostics stay short and discoverable.

## Canonical command tree

The public CLI will expose this hierarchy:

```text
biolm --version
biolm status
biolm whoami

biolm account login
biolm account logout
biolm account usage
biolm account budget
biolm account budget set AMOUNT
biolm account api-key create
biolm account api-key delete TOKEN_OR_PREFIX
biolm account org list
biolm account org show ORGANIZATION
biolm account org invite ORGANIZATION EMAIL

biolm workspace list
biolm workspace show [PATH]
biolm workspace switch PATH
biolm workspace create NAME
```

The existing `model`, `protocol`, `dataset`, and `hub` groups remain unchanged.

`biolm account usage` is a leaf command because no other usage action exists.
`biolm account budget` shows the current budget when invoked without a
subcommand. `biolm account budget set` remains the mutating form.

The CLI will not expose organization creation. Users must create organizations
through the BioLM console.

## Help behavior

Top-level help will list direct commands and groups without expanding every
group's children. Group help will list that group's direct children. This
produces a short top-level menu and lets the hierarchy provide the organization.

`RichGroup` currently lists every registered command and does not honor
`hidden=True`. The help renderer must skip hidden commands, stop expanding
group children at the top level, and remove the `Platform` category.

Top-level help sections after the change:

- Account: `status`, `whoami`, `account`
- Workspace: `workspace`
- Hub: `hub`
- Models: `model`
- Protocols: `protocol`
- Datasets: `dataset`

`biolm --version` is canonical. The `biolm version` compatibility command will
remain callable but hidden from help and documentation.

## Compatibility aliases

Existing scripts will continue to work through hidden aliases:

```text
biolm login
biolm logout
biolm usage show
biolm budget show
biolm budget set AMOUNT
biolm apikey create
biolm apikey delete TOKEN_OR_PREFIX
biolm org list
biolm org show ORGANIZATION
biolm org invite ORGANIZATION EMAIL
biolm version
```

Each alias is a separate Click `Command` or `Group` with `hidden=True` that
shares the canonical callback and parameters. Sharing one command object under
two names is not sufficient because hiding one name would hide both. Aliases
will not print deprecation warnings. The removed `biolm org create` command
will not receive an alias.

## Status and identity

`biolm status` remains top-level. It answers "what am I connected to?" by
reporting configured endpoints, authentication state, and the active
account/workspace when available. It must remain useful when the user is logged
out or a service is unavailable. Unavailable diagnostics must not hide the
remaining configuration.

`status` does not use `_platform_request`. Constructing `PlatformClient`
raises when credentials are missing, and `_platform_request` converts
`PlatformError` into a fatal Click error. `status` must catch those errors for
client construction and each optional platform probe, then continue printing
endpoints and auth presence with account/workspace shown as unavailable.

`biolm whoami` answers "who is authenticated?" It fetches the authenticated
principal from `/api/users/me/` and the active account from
`/console/api/account-context/` through one authenticated `PlatformClient`
session. Those endpoints live under different URL prefixes. The client stays
rooted at `/console/api/` for platform routes, while `get_current_user()`
requests the absolute URL `{origin}/api/users/me/`. `_request` / `_url` must
pass absolute URLs through unchanged so httpx does not join them to the console
base. Auth headers already match: `/api/users/me/` accepts Knox `Token`, OAuth
Bearer, and JWT.

Human output includes the user's username, email, and display name when
present, followed by the active personal or organization account and
environment. JSON output uses stable fields and excludes unrelated user profile
and billing fields:

```json
{
  "id": 1,
  "username": "alice",
  "email": "alice@example.com",
  "first_name": "Alice",
  "last_name": "Example",
  "account_type": "organization",
  "account_id": 7,
  "account_name": "Acme Labs",
  "account_slug": "acme",
  "environment_id": 22
}
```

Personal account context from `/console/api/account-context/` does not include
`name` or `slug`. For `account_type == "user"`, `account_name` is `null` and
`account_slug` is the username returned by `/api/users/me/`, matching the
personal label used in workspace paths. Organization context continues to use
`account_details.name` and `account_details.slug`.

Authentication errors produce a nonzero Click error. Missing optional identity
or context fields render as an em dash in human output and `null` in JSON.

`PlatformClient.get_current_user()` will expose the authenticated identity for
SDK users.

## Organization resolution

Organization commands accept an exact organization name or slug. The client
will fetch the caller's organizations, compare the supplied value exactly
against both fields, and resolve the result to the numeric ID required by the
backend detail and invitation endpoints.

Resolution fails before mutation when no organization matches. Because
organization names and slugs use separate uniqueness constraints, an exact
name match and an exact slug match can identify different organizations. In
that case the client reports an ambiguous identifier instead of guessing.

Numeric organization IDs remain accepted by the Python SDK and by both the
canonical CLI and hidden aliases so existing scripts that pass an ID continue
to work. When the supplied value is all decimal digits, resolution prefers an
exact organization `id` match before name or slug matching. CLI help and
documentation describe name or slug input only.

`--account` options on usage, API-key, and workspace commands keep their
existing slug-or-personal-label semantics. Only organization commands accept
organization display names. That difference is intentional.

## Error handling

Account and organization mutating or lookup commands, including their aliases,
share the existing `_platform_request` boundary, which closes the client and
converts `PlatformError` to `ClickException`. `status` is exempt, as described
above. Organization lookup errors name the rejected identifier. `whoami` never
prints credential material or the complete `/api/users/me/` response.

## Testing

Client tests will cover current-user retrieval over the absolute `/api/users/me/`
URL, absolute-URL passthrough in `_request`, organization name and slug
resolution, numeric-ID preference for all-digit identifiers, missing and
ambiguous identifiers, and invitation resolution.

CLI tests will cover:

- the concise canonical hierarchy and hidden aliases;
- top-level help sections without `Platform` or expanded group children;
- direct account usage and budget commands;
- the removal of organization creation;
- organization name, slug, and numeric-ID delegation;
- `whoami` table and JSON output for both personal and organization contexts;
- top-level status account/workspace diagnostics and graceful degradation when
  credentials or platform probes fail;
- unchanged behavior for each compatibility alias.

Documentation will use only canonical commands.
