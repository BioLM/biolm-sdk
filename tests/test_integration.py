# New file with client integration tests

import pytest

from biolm.core.http import BioLMApiClient

# ---------------------------------------------------------------------------
# Minimal payload builders (items + optional params) for each model/action.
# These mirror the logic used by backend test_endpoints_v3.py but rely on the
# python client rather than raw HTTP requests.
# ---------------------------------------------------------------------------

# Removed unused JSON + Pydantic schema imports to keep this test independent
# of the main biolm_web codebase and avoid heavy dependencies. Only retain
# the essential pytest and BioLM client imports.

# ---------------------------------------------------------------------------
# CASE MATRIX (same as backend integration tests, limited to a subset so the
# CI run stays fast)
# ---------------------------------------------------------------------------

CASES = [
    ("esmc-300m", "predict"),
    ("evo-15-8k-base", "generate"),
    ("igbert-paired", "encode"),
    ("igbert-paired", "predict"),
    ("igbert-paired", "generate"),
    ("nanobert", "encode"),
    ("nanobert", "predict"),
    ("nanobert", "generate"),
    ("omni-dna-1b", "encode"),
    ("omni-dna-1b", "predict"),
    ("antifold", "encode"),
    ("antifold", "predict"),
    ("antifold", "generate"),
    ("dna-chisel", "predict"),
    ("evo2-1b-base", "encode"),
    ("evo2-1b-base", "predict"),
    ("evo2-1b-base", "generate"),
]


# ---------------------------------------------------------------------------
# Helper: build items / params for BioLMApiClient
# ---------------------------------------------------------------------------

def build_items_params(slug: str, action: str):
    """Return (items, params or None) suitable for BioLMApiClient.* call."""

    # Protein / DNA strings used in several places
    heavy_seq = "EVQLVESGGGLVQ"
    light_seq = "DIQMTQ"

    if slug == "esmc-300m":
        return [{"sequence": "ACD<mask>E"}], None
    if slug == "evo-15-8k" or slug == "evo-15-8k-base":
        return [{"prompt": "ACTG"}], None
    if slug == "igbert-paired":
        if action == "encode" or action == "predict":
            return [{"heavy": heavy_seq, "light": light_seq}], None
        if action == "generate":
            seq_with_star = heavy_seq + light_seq + "*"
            return [{"sequence": seq_with_star}], None
    if slug == "nanobert":
        if action == "encode" or action == "predict":
            return [{"sequence": "EVQLVESGGG"}], None
        if action == "generate":
            return [{"sequence": "EVQLVESGGG"}], None
    if slug == "omni-dna-1b":
        if action == "encode":
            return [{"sequence": "ACTGACTG"}], None
        if action == "predict":
            return [{"sequence": "ACTGACTG"}], None
    if slug == "antifold":
        pdb_example = (
            "ATOM      1  N   MET A   1       0.000   0.000   0.000  1.00 20.00           N  \n"
            "ATOM      2  C   MET B   1       1.000   0.000   0.000  1.00 20.00           C  \n"
            "END                                                                \n"
        )
        params_common = {"heavy_chain": "A", "light_chain": "B"}
        if action == "encode":
            return [{"pdb": pdb_example}], params_common
        if action == "predict":
            return [{"pdb": pdb_example}], params_common
        if action == "generate":
            # Provide both heavy and light chain regions explicitly
            params_common["regions"] = ["all"]
            return [{"pdb": pdb_example}], params_common
    if slug == "dna-chisel":
        return [{"sequence": "ATGCATGC"}], None
    if slug == "evo2-1b-base":
        if action == "encode":
            return [{"sequence": "ACTGACTG"}], None
        if action == "predict":
            return [{"sequence": "ACTGACTG"}], None
        if action == "generate":
            return [{"prompt": "ACTG"}], None
    raise ValueError(f"Unhandled case {slug}/{action}")


# ---------------------------------------------------------------------------
# Parametrised tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("slug,action", CASES)
async def test_client_roundtrip(slug, action):
    # Skip alphafold2 tests
    if slug == "alphafold2":
        pytest.skip("Skipping alphafold2 tests")
    
    items, params = build_items_params(slug, action)
    client = BioLMApiClient(slug, raise_httpx=False, unwrap_single=False, retry_error_batches=False)

    if action == "encode":
        result = await client.encode(items=items, params=params, stop_on_error=False) if params else await client.encode(items=items, stop_on_error=False)
    elif action == "predict":
        result = await client.predict(items=items, params=params, stop_on_error=False) if params else await client.predict(items=items, stop_on_error=False)
    elif action == "generate":
        result = await client.generate(items=items, params=params, stop_on_error=False) if params else await client.generate(items=items, stop_on_error=False)
    else:
        pytest.skip(f"unsupported action {action}")

    # Basic sanity checks – type & non-empty results
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == len(items), "Result length mismatch"
    for res in result:
        assert isinstance(res, dict), "Each item must be a dict"
        assert "error" not in res, f"Backend returned error: {res}"

    # Explicitly close HTTP + websocket resources to avoid cross-test leaks
    await client.shutdown() 