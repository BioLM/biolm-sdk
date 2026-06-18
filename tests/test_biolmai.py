import json
import random

import httpx
import pytest

from biolm import biolm
from biolm.client import BioLM
from biolm.models import Model

N = 6

def return_shuffle(list_):
    copy_ = list(list_)
    random.shuffle(copy_)
    return copy_

def insert_random_single_occurence(text, so):
    idx = random.randint(0, len(text))
    return text[:idx] + so + text[idx:]

@pytest.mark.parametrize("entity,action,type_,items,params,expected_type,expected_key", [
    # Positive: encode single sequence
    ("esm2-8m", "encode", "sequence", "MSILVTRPSPAGEEL", None, dict, "embeddings"),
    # Positive: fold single sequence
    ("esmfold", "predict", "sequence", "MENDELMENDEL", None, dict, "pdb"),
    # Positive: encode batch
    ("esmfold", "predict", "sequence", [
        "MSILVTRPSPAGE",
        "ACDEFGHIKLMNP"
    ], None, list, "pdb"),
    # Positive: predict with mask (key is dynamic: esm1v-n1 through esm1v-n5)
    ("esm1v-all", "predict", "sequence", "MSILSPAG<mask>ELVSRLR", None, dict, "esm1v-n*"),
    # Positive: predict single as list (key is dynamic: esm1v-n1 through esm1v-n5)
    ("esm1v-all", "predict", "sequence", ["MSILSPAG<mask>ELVSRLR"], None, dict, "esm1v-n*"),
    # Positive: generate with params
    ("progen2-oas", "generate", "context", "M", {"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}, list, "sequence"),
    # Negative: invalid action
    ("esm2-8m", "invalid_action", "sequence", "MSILVTRPSP", None, ValueError, None),
    # Negative: invalid type (not a valid key for the model)
    ("esm2-8m", "encode", "not_a_type", "MSILVTRPSPA", None, dict, "error"),
    # Edge: empty items list
    ("esm2-8m", "encode", "sequence", [], None, list, None),
    # Edge: None as items
    ("esm2-8m", "encode", "sequence", None, None, dict, "error"),
    # Edge: dict as items
    ("esm2-8m", "encode", "sequence", {'hi': 'bye'}, None, dict, "error"),
])
def test_biolm_run_single_items(entity, action, type_, items, params, expected_type, expected_key):
    # Handle expected Exception types
    if isinstance(expected_type, type) and issubclass(expected_type, BaseException):
        with pytest.raises(expected_type):
            BioLM(entity=entity, action=action, type=type_, items=items, params=params, raise_httpx=True)
        return

    # Handle all other expected types
    result = BioLM(entity=entity, action=action, type=type_, items=items, params=params, raise_httpx=False)
    assert isinstance(result, expected_type)
    if expected_key is not None:
        # Special handling for esm1v-all which returns dynamic keys (esm1v-n1 through esm1v-n5)
        def check_key_in_result(res, key, idx=None):
            if key == "esm1v-n*":
                # Check that any key matches esm1v-n{1-5} pattern
                matching_keys = [k for k in res.keys() if k.startswith("esm1v-n") and len(k) == 8 and k[7] in "12345"]
                idx_msg = f" at index {idx}" if idx is not None else ""
                assert len(matching_keys) > 0, f"No esm1v-n{{1-5}} key found in result{idx_msg}. Keys: {list(res.keys())}"
            else:
                assert key in res, f"Expected key '{key}' not found in result{idx_msg if idx is not None else ''}"
        
        if isinstance(result, list):
            if isinstance(expected_key, list):
                assert len(result) == len(expected_key), "Length of result and expected_key must match"
                for idx, (res, key) in enumerate(zip(result, expected_key)):
                    assert isinstance(res, dict), f"Result at index {idx} is not a dict"
                    check_key_in_result(res, key, idx)
            else:
                for idx, res in enumerate(result):
                    assert isinstance(res, dict), f"Result at index {idx} is not a dict"
                    check_key_in_result(res, expected_key, idx)
        else:
            assert not isinstance(expected_key, list), "expected_key must not be a list when result is a dict"
            check_key_in_result(result, expected_key)


def test_biolm_predict_batch_valid():
    result = BioLM(entity="esmfold", action="predict", type="sequence",
                  items=["MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"] * N)
    assert isinstance(result, list)
    assert len(result) == N
    assert all(isinstance(r, dict) for r in result)
    assert all("mean_plddt" in r for r in result)

def test_biolm_predict_invalid_sequence_no_raise_httpx():
    bad_seq = "MSILVTRPSPAGEELVSRLRTLGQVQLAALGESDLLFALSQH"  # (FYI, no <mask>)
    result = BioLM(entity="esm1v-all", action="predict", type="sequence", items=bad_seq, raise_httpx=False)
    assert isinstance(result, dict)
    assert "error" in result

def test_biolm_predict_invalid_sequence_raise_httpx():
    bad_seq = "MSILVTRPSPAGHFPLIFSPQQLPQ"  # (FYI, no <mask>)
    with pytest.raises(httpx.HTTPStatusError):
        result = BioLM(entity="esm1v-all", action="predict", type="sequence", items=bad_seq, raise_httpx=True)

def test_biolm_predict_good_and_invalid_sequences_no_raise_httpx():
    good_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    total = 24  # 3 batches of 8
    batch_size = 8  # Current maxSize of esm2-8m
    # Start with all good
    all_seqs = [good_seq for _ in range(total)]

    # Insert bads at scattered positions, but not in every batch
    bad_indices = [10, 11,]
    for idx in bad_indices:
        all_seqs[idx] = good_seq + "i1"

    # Now, at least one batch (e.g., batch 0, or 2) should be good
    result = BioLM(
        entity="esm2-8m",
        action="encode",
        type="sequence",
        items=all_seqs,
        raise_httpx=False,
        stop_on_error=False
    )
    assert isinstance(result, list)
    assert len(result) == len(all_seqs)
    assert any("error" in r for r in result)
    assert any("embeddings" in r for r in result)



def test_biolm_predict_good_and_invalid_sequences_raise_httpx():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHH"
    seqs = ["".join(return_shuffle(list(base_seq)))[:30] for _ in range(int(N / 2))]
    bad_seqs = ["".join(return_shuffle(list(base_seq)))[:30] + "i1" for _ in range(int(N / 2))]
    all_seqs = seqs + bad_seqs
    random.shuffle(all_seqs)
    with pytest.raises(Exception):
        result = BioLM(entity="esm2-8m", action="encode", type="sequence", items=all_seqs, raise_httpx=True)

@pytest.mark.skip("API failing to validate this payload")
def test_biolm_predict_too_long_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    bad_seqs = ["".join(return_shuffle(list(base_seq))) * 1000 for _ in range(int(N / 2))]
    all_seqs = bad_seqs
    result = BioLM(entity="esm2-8m", action="encode", type="sequence", items=all_seqs)
    assert isinstance(result, list)
    assert len(result) == len(all_seqs)
    assert any("error" in r for r in result)
    assert any("embeddings" in r for r in result)

@pytest.mark.skip("API failing to validate this payload")
def test_biolm_predict_good_and_too_long_sequences():
    base_seq = "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQLADQLAALGESDLLFALSQH"
    seqs = ["".join(return_shuffle(list(base_seq)))[:30] for _ in range(int(N / 2))]
    bad_seqs = ["".join(return_shuffle(list(base_seq))) * 1000 for _ in range(int(N / 2))]
    all_seqs = seqs + bad_seqs
    result = BioLM(entity="esm2-8m", action="encode", type="sequence", items=all_seqs)
    assert isinstance(result, list)
    assert len(result) == len(all_seqs)
    assert any("error" in r for r in result)
    assert any("embeddings" in r for r in result)

def test_biolm_predict_to_disk(tmp_path):
    file_path = tmp_path / "out.jsonl"
    BioLM(entity="esmfold", action="predict", type="sequence", items="MDNELE", output='disk', file_path=str(file_path))
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert isinstance(data, dict)
    assert "mean_plddt" in data
    assert "pdb" in data

def test_biolm_batch_predict_to_disk_dict_items(tmp_path):
    items = [{"sequence": "MDNELE"}, {"sequence": "VALID"}]
    file_path = tmp_path / "batch.jsonl"
    BioLM(entity="esmfold", action="predict", items=items, output='disk', file_path=str(file_path), unwrap_single=True)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert isinstance(rec, dict)
        assert "mean_plddt" in rec

def test_biolm_batch_predict_to_disk_sequence_items(tmp_path):
    items = ["MDNELE", "VALID"]
    file_path = tmp_path / "batch.jsonl"
    BioLM(entity="esmfold", action="predict", type="sequence", items=items, output='disk', file_path=str(file_path), unwrap_single=True)
    assert file_path.exists()
    lines = file_path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert isinstance(rec, dict)
        assert "mean_plddt" in rec

def test_biolm_generate_num_samples():
    entity = "progen2-oas"
    action = "generate"
    type_ = "context"
    items = "M"
    params = {"temperature": 0.7, "top_p": 0.6, "num_samples": 2, "max_length": 17}
    expected_type = list
    expected_key = "sequence"

    result = BioLM(entity=entity, action=action, type=type_, items=items, params=params, raise_httpx=False)
    assert isinstance(result, expected_type)
    assert len(result) == 2, f"Expected 2 samples, got {len(result)}"
    for idx, res in enumerate(result):
        assert isinstance(res, dict), f"Result at index {idx} is not a dict"
        assert expected_key in res, f"Expected key '{expected_key}' not found in result at index {idx}"


def test_encode_single_same_via_biolm_biolm_fn_model():
    """Encode single sequence gives same shape via BioLM(), biolm(), and Model()."""
    items = "MSILVTRPSPAGEEL"
    kw = dict(entity="esm2-8m", action="encode", type="sequence", items=items, raise_httpx=False)
    r_biolm = BioLM(**kw)
    r_fn = biolm(**kw)
    r_model = Model("esm2-8m", raise_httpx=False).encode(items=items, type="sequence", progress=False)
    for name, r in [("BioLM", r_biolm), ("biolm()", r_fn), ("Model", r_model)]:
        assert isinstance(r, dict), f"{name}: expected dict"
        assert "embeddings" in r, f"{name}: expected embeddings key"
    assert set(r_biolm.keys()) == set(r_fn.keys()) == set(r_model.keys())


def test_predict_single_same_via_biolm_biolm_fn_model():
    """Predict single sequence gives same shape via BioLM(), biolm(), and Model() (default unwrap_single=True)."""
    items = [{"sequence": "MDNELE"}]
    r_biolm = BioLM(entity="esmfold", action="predict", items=items, raise_httpx=False)
    r_fn = biolm(entity="esmfold", action="predict", items=items, raise_httpx=False)
    r_model = Model("esmfold", raise_httpx=False).predict(items=items, progress=False)
    for name, r in [("BioLM", r_biolm), ("biolm()", r_fn), ("Model", r_model)]:
        assert isinstance(r, dict), f"{name}: expected dict (unwrap_single=True)"
        assert "mean_plddt" in r and "pdb" in r, f"{name}: expected keys"
