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
`hidden=True`. The help renderer must skip hidden commands and remove the
`Platform` category. Top-level help sections should follow user domains such as
Account, Workspace, Hub, Models, Protocols, and Datasets.

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
out or a service is unavailable; unavailable diagnostics should not hide the
remaining configuration.

`biolm whoami` answers "who is authenticated?" It calls `/api/users/me/` and
the platform account-context endpoint through one authenticated
`PlatformClient` session. Human output includes the user's username, email, and
display name when present, followed by the active personal or organization
account and environment. JSON output uses stable fields and excludes unrelated
user profile and billing fields:

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

Authentication errors produce a nonzero Click error. Missing optional identity
or context fields render as an em dash in human output and `null` in JSON.

`PlatformClient.get_current_user()` will expose the authenticated identity for
SDK users. Its request uses the same credentials and persistent client as
account-context requests.

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
to work. CLI help and documentation describe name or slug input only.

## Error handling

Canonical commands and aliases share the existing `_platform_request` boundary,
which closes the client and converts `PlatformError` to `ClickException`.
Organization lookup errors name the rejected identifier. `whoami` never prints
credential material or the complete `/api/users/me/` response.

## Testing

Client tests will cover current-user retrieval, organization name and slug
resolution, missing and ambiguous identifiers, numeric-ID compatibility, and
invitation resolution.

CLI tests will cover:

- the concise canonical hierarchy and hidden aliases;
- direct account usage and budget commands;
- the removal of organization creation;
- organization name and slug delegation;
- `whoami` table and JSON output;
- top-level status account/workspace diagnostics and graceful degradation;
- unchanged behavior for each compatibility alias.

Documentation will use only canonical commands.
