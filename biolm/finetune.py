"""Programmatic finetuning client for BioLM.

Wraps the ``/api/finetune/`` endpoints (the SDK-facing companion to the browser
console finetuning UI) so callers can launch and track XGBoost and DSM
finetuning runs from Python without a session cookie.

Auth follows the same rules as the rest of the SDK: pass ``api_key=`` or set
``BIOLMAI_TOKEN`` / ``BIOLM_TOKEN`` (see :class:`biolm.core.http.CredentialsProvider`).
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

# Terminal run states — used by :meth:`Finetune.wait`.
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "error"}

Rows = Union[List[Dict[str, Any]], str]


def _run_sync(coro):
    """Run *coro* to completion whether or not an event loop is active."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _base_url() -> str:
    """Resolve the ``/api/finetune`` base from the configured BioLM domain."""
    from biolm.core.const import BIOLMAI_BASE_DOMAIN

    return f"{BIOLMAI_BASE_DOMAIN.rstrip('/')}/api/finetune"


def _client(api_key: Optional[str], base_url: Optional[str]):
    """Build an :class:`HttpClient` pointed at the finetune API."""
    import httpx

    from biolm.core.http import CredentialsProvider, HttpClient

    final_base = base_url if base_url is not None else _base_url()
    headers = CredentialsProvider.get_auth_headers(api_key)
    timeout = httpx.Timeout(60.0, connect=10.0)
    return HttpClient(final_base, headers, timeout)


def _raise_for_status(resp, action: str):
    """Translate an error response into a Python exception, mirroring protocols.py."""
    if resp.status_code == 401:
        raise PermissionError(
            "Authentication required. Set BIOLMAI_TOKEN (or BIOLM_TOKEN) "
            "or run `biolm login`."
        )
    if resp.status_code == 402:
        raise PermissionError(
            "Finetuning is not enabled for this account. "
            "Contact BioLM to enable finetuning."
        )
    if resp.status_code == 404:
        raise FileNotFoundError(f"Run not found while trying to {action}.")
    if resp.status_code >= 400:
        detail = resp.text
        try:
            body = resp.json()
            detail = body.get("error") or body.get("detail") or body.get("errors") or detail
        except Exception:
            pass
        raise ValueError(f"Failed to {action}: {detail} (status {resp.status_code})")


async def _post(endpoint: str, payload: dict, api_key, base_url, action: str) -> dict:
    client = _client(api_key, base_url)
    try:
        resp = await client.post(endpoint, payload)
        _raise_for_status(resp, action)
        return resp.json()
    finally:
        await client.close()


async def _get(endpoint: str, api_key, base_url, action: str) -> dict:
    client = _client(api_key, base_url)
    try:
        resp = await client.get(endpoint)
        _raise_for_status(resp, action)
        return resp.json()
    finally:
        await client.close()


async def _delete(endpoint: str, api_key, base_url, action: str) -> dict:
    client = _client(api_key, base_url)
    try:
        # HttpClient exposes get/post; reach the shared async client for DELETE.
        http = await client.get_async_client()
        endpoint = endpoint.lstrip("/")
        if not endpoint.endswith("/"):
            endpoint += "/"
        resp = await http.delete(endpoint)
        _raise_for_status(resp, action)
        return resp.json()
    finally:
        await client.close()


def _drop_none(d: dict) -> dict:
    """Strip keys whose value is ``None`` so server-side defaults apply."""
    return {k: v for k, v in d.items() if v is not None}


class Finetune:
    """Launch and track BioLM finetuning runs.

    All methods are classmethods returning plain dicts. ``*_data`` arguments
    accept a list of row dicts (``[{"sequence": ..., "label": ...}, ...]``) or a
    raw CSV string; they are sent inline as JSON.
    """

    # -- XGBoost ----------------------------------------------------------

    @classmethod
    async def xgboost_async(
        cls,
        *,
        train_data: Rows,
        embedding_models: List[str],
        task_type: str = "classification",
        target_column: str = "label",
        text_column: str = "sequence",
        test_data: Optional[Rows] = None,
        validation_data: Optional[Rows] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_splits: int = 5,
        seed: int = 42,
        hyperopt: bool = False,
        hyperopt_n_trials: Optional[int] = None,
        antibody_mode: bool = False,
        heavy_column: str = "heavy",
        light_column: str = "light",
        modality: str = "protein",
        run_name: Optional[str] = None,
        environment_id: Optional[Union[int, str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """Launch an XGBoost finetune (optionally with Ray Tune hyperopt)."""
        payload = _drop_none(
            {
                "run_name": run_name,
                "task_type": task_type,
                "target_column": target_column,
                "text_column": text_column,
                "embedding_models": list(embedding_models),
                "train_data": train_data,
                "test_data": test_data,
                "validation_data": validation_data,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_splits": n_splits,
                "seed": seed,
                "hyperopt": hyperopt,
                "hyperopt_n_trials": hyperopt_n_trials,
                "antibody_mode": antibody_mode,
                "heavy_column": heavy_column,
                "light_column": light_column,
                "modality": modality,
                "environment_id": environment_id,
            }
        )
        return await _post("xgboost/runs/", payload, api_key, base_url, "create XGBoost run")

    @classmethod
    def xgboost(cls, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`xgboost_async`."""
        return _run_sync(cls.xgboost_async(**kwargs))

    # -- DSM Stage 1 ------------------------------------------------------

    @classmethod
    async def dsm_stage1_async(
        cls,
        *,
        train_data: Rows,
        test_data: Optional[Rows] = None,
        valid_data: Optional[Rows] = None,
        sequence_col: str = "sequence",
        lr: float = 1e-4,
        batch_size: int = 8,
        grad_accum: int = 16,
        max_steps: int = 50000,
        max_length: int = 2048,
        save_every: int = 1000,
        fp16: bool = False,
        run_name: Optional[str] = None,
        environment_id: Optional[Union[int, str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """Launch DSM Stage 1 (single-chain masked-LM finetune)."""
        payload = _drop_none(
            {
                "run_name": run_name,
                "sequence_col": sequence_col,
                "train_data": train_data,
                "test_data": test_data,
                "valid_data": valid_data,
                "lr": lr,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "max_steps": max_steps,
                "max_length": max_length,
                "save_every": save_every,
                "fp16": fp16,
                "environment_id": environment_id,
            }
        )
        return await _post(
            "dsm/stage1/runs/", payload, api_key, base_url, "create DSM Stage 1 run"
        )

    @classmethod
    def dsm_stage1(cls, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`dsm_stage1_async`."""
        return _run_sync(cls.dsm_stage1_async(**kwargs))

    # -- DSM Stage 2 ------------------------------------------------------

    @classmethod
    async def dsm_stage2_async(
        cls,
        *,
        stage1_checkpoint: str,
        paired_data: Rows,
        unpaired_data: Optional[Rows] = None,
        heavy_col: str = "heavy",
        light_col: str = "light",
        use_mixed_training: bool = False,
        fp16: bool = False,
        lr: float = 5e-5,
        batch_size: int = 4,
        unpaired_batch_size: int = 8,
        grad_accum: int = 16,
        max_steps: int = 25000,
        max_length: int = 300,
        unpaired_ratio: float = 2.0,
        run_name: Optional[str] = None,
        environment_id: Optional[Union[int, str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """Launch DSM Stage 2 (paired multichain finetune from a Stage 1 checkpoint)."""
        payload = _drop_none(
            {
                "run_name": run_name,
                "stage1_checkpoint": stage1_checkpoint,
                "paired_data": paired_data,
                "unpaired_data": unpaired_data,
                "heavy_col": heavy_col,
                "light_col": light_col,
                "use_mixed_training": use_mixed_training,
                "fp16": fp16,
                "lr": lr,
                "batch_size": batch_size,
                "unpaired_batch_size": unpaired_batch_size,
                "grad_accum": grad_accum,
                "max_steps": max_steps,
                "max_length": max_length,
                "unpaired_ratio": unpaired_ratio,
                "environment_id": environment_id,
            }
        )
        return await _post(
            "dsm/stage2/runs/", payload, api_key, base_url, "create DSM Stage 2 run"
        )

    @classmethod
    def dsm_stage2(cls, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`dsm_stage2_async`."""
        return _run_sync(cls.dsm_stage2_async(**kwargs))

    # -- DSM RL -----------------------------------------------------------

    @classmethod
    async def dsm_rl_async(
        cls,
        *,
        seed_sequences: Union[List[str], str],
        oracle_type: str = "esmc",
        stability_objective: str = "thermostability",
        training_mode: str = "online",
        algorithm: str = "ppo",
        num_episodes: int = 100,
        samples_per_episode: int = 64,
        learning_rate: float = 3e-4,
        batch_size: int = 8,
        mask_ratio: float = 0.3,
        mutation_rate: float = 0.1,
        run_name: Optional[str] = None,
        environment_id: Optional[Union[int, str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """Launch DSM RL protein optimization against an oracle."""
        payload = _drop_none(
            {
                "run_name": run_name,
                "seed_sequences": seed_sequences,
                "oracle_type": oracle_type,
                "stability_objective": stability_objective,
                "training_mode": training_mode,
                "algorithm": algorithm,
                "num_episodes": num_episodes,
                "samples_per_episode": samples_per_episode,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "mask_ratio": mask_ratio,
                "mutation_rate": mutation_rate,
                "environment_id": environment_id,
            }
        )
        return await _post("dsm/rl/runs/", payload, api_key, base_url, "create DSM RL run")

    @classmethod
    def dsm_rl(cls, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`dsm_rl_async`."""
        return _run_sync(cls.dsm_rl_async(**kwargs))

    # -- Run tracking -----------------------------------------------------

    @classmethod
    async def list_runs_async(
        cls,
        *,
        dag: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """List the caller's finetune runs (paginated)."""
        params = []
        if dag:
            params.append(f"dag={dag}")
        if status:
            params.append(f"status={status}")
        params.append(f"page={page}")
        params.append(f"page_size={page_size}")
        endpoint = "runs/?" + "&".join(params)
        return await _get(endpoint, api_key, base_url, "list runs")

    @classmethod
    def list_runs(cls, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`list_runs_async`."""
        return _run_sync(cls.list_runs_async(**kwargs))

    @classmethod
    async def get_run_async(
        cls, run_id: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> dict:
        """Fetch sanitized detail (status, results) for a single run."""
        return await _get(f"runs/{run_id}/", api_key, base_url, "get run")

    @classmethod
    def get_run(cls, run_id: str, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`get_run_async`."""
        return _run_sync(cls.get_run_async(run_id, **kwargs))

    @classmethod
    async def progress_async(
        cls, run_id: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> dict:
        """Fetch lightweight status + telemetry channel id for a run."""
        return await _get(f"runs/{run_id}/progress/", api_key, base_url, "get progress")

    @classmethod
    def progress(cls, run_id: str, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`progress_async`."""
        return _run_sync(cls.progress_async(run_id, **kwargs))

    @classmethod
    async def cancel_async(
        cls, run_id: str, *, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> dict:
        """Cancel an in-flight run."""
        return await _delete(f"runs/{run_id}/", api_key, base_url, "cancel run")

    @classmethod
    def cancel(cls, run_id: str, **kwargs) -> dict:
        """Synchronous wrapper for :meth:`cancel_async`."""
        return _run_sync(cls.cancel_async(run_id, **kwargs))

    @classmethod
    def wait(
        cls,
        run_id: str,
        *,
        poll_interval: float = 15.0,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> dict:
        """Block until *run_id* reaches a terminal state, returning its detail."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            prog = cls.progress(run_id, api_key=api_key, base_url=base_url)
            if str(prog.get("status", "")).lower() in TERMINAL_STATUSES:
                return cls.get_run(run_id, api_key=api_key, base_url=base_url)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Run {run_id} did not finish within {timeout}s "
                    f"(last status: {prog.get('status')})."
                )
            time.sleep(poll_interval)

