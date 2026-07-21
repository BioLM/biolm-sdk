LLTP Config Schema Reference
============================

Project compose for Lab-in-the-Loop runs. ``lltp.yaml`` names connectors, auth
env vars, and experiments. It is **not** the LLTP wire protocol (orders,
catalogs, status envelopes). Those live in the
`LLTP spec <https://github.com/BioLM/lltp>`_.

Create a starter file with ``biolm lab init``. Discovery walks from the current
directory upward until it finds ``lltp.yaml``. See :doc:`../guide/lab` for
install, submit/status/results, and the SeqFrame bridge.

Minimal example
---------------

.. code-block:: yaml

    version: 1
    default_connector: adaptyv
    connectors:
      adaptyv:
        auth:
          token_env: ADAPTYV_API_TOKEN
        defaults:
          service_id: adaptyv-lltp.expression-v1
    experiments:
      express:
        connector: adaptyv
        service_id: adaptyv-lltp.expression-v1

Recommended example
-------------------

.. code-block:: yaml

    version: 1
    default_connector: adaptyv
    connectors:
      adaptyv:
        auth:
          token_env: ADAPTYV_API_TOKEN
        defaults:
          service_id: adaptyv-lltp.expression-v1
      twist:
        auth:
          token_env: TWIST_END_USER_TOKEN
          jwt_env: TWIST_STAGING_JWT_TOKEN
          email_env: TWIST_STAGING_USER_EMAIL
        defaults:
          service_id: twist-lltp.dna-synthesis-v1
          wait_for_scoring: false
    experiments:
      express:
        connector: adaptyv
        service_id: adaptyv-lltp.expression-v1
      synthesize:
        connector: twist
        service_id: twist-lltp.dna-synthesis-v1
        wait_for_scoring: false

Top-level fields
----------------

- ``version`` — integer; currently ``1`` (default when omitted).
- ``default_connector`` — connector name used when submit does not name one
  (and no experiment supplies it).
- ``connectors`` — map of connector name → connector config (see below).
- ``experiments`` — map of experiment name → experiment config (see below).

Connectors
----------

Each entry under ``connectors`` is a named lab provider package (for example
``adaptyv`` or ``twist`` from
`lltp-connectors <https://github.com/BioLM/lltp-connectors>`_).

- ``auth`` — credentials and env-var pointers (see Auth).
- ``defaults`` — parameters merged into every submit for this connector unless
  an experiment or call overrides them. Common keys:

  - ``service_id`` — connector service identifier (required for most submits).
  - ``wait_for_scoring`` — Twist-style option; connector-specific.

Other keys under ``defaults`` are passed through to the connector.

Experiments
-----------

Named presets for ``biolm lab submit --experiment <name>`` /
``biolm.lab.submit(..., experiment=...)``.

**Required**

- ``connector`` — key into ``connectors``.

**Optional (and any extra keys)**

Everything except ``connector`` becomes experiment ``params``. Typical fields:

- ``service_id`` — overrides connector ``defaults.service_id``.
- ``wait_for_scoring`` — overrides connector default when present.

Auth
----

Auth is BYOK. Environment variables always win over inline secrets when the
corresponding ``*_env`` key is set.

Supported auth keys (inline / env pair):

- ``token`` / ``token_env`` — API token (Adaptyv and similar).
- ``jwt`` / ``jwt_env`` — Twist JWT.
- ``email`` / ``email_env`` — Twist user email.
- ``end_user_token`` / ``end_user_token_env`` — Twist end-user token.

Prefer ``*_env`` so secrets stay out of the YAML file.

What is not in this file
------------------------

- Run history and status — stored under ``.biolm/lltp/<run_id>.json``.
- LLTP wire envelopes (orders, catalogs, results) — see the
  `LLTP spec <https://github.com/BioLM/lltp>`_ and language SDKs
  (`lltp-py <https://github.com/BioLM/lltp-py>`_,
  `lltp-js <https://github.com/BioLM/lltp-js>`_).
- SeqFrame Parquet layout — :doc:`seqframe-schema`.

Python
------

.. code-block:: python

    from biolm.lab.config import load_config, write_example_config

    write_example_config("lltp.yaml")  # or: biolm lab init
    cfg = load_config()
    print(cfg.default_connector, list(cfg.experiments))
