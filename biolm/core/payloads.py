def INST_DAT_TXT(batch, include_batch_size=False):
    d = {"instances": []}
    for _, row in batch.iterrows():
        inst = {"data": {"text": row.text}}
        d["instances"].append(inst)
    if include_batch_size is True:
        d["batch_size"] = len(d["instances"])
    return d

def PARAMS_ITEMS(batch, key="sequence", params=None, include_batch_size=False):
    d = {"items": []}
    for _, row in batch.iterrows():
        inst = {key: row.text}
        d["items"].append(inst)
    if include_batch_size is True:
        d["batch_size"] = len(d["items"])
    if isinstance(params, dict):
        d["params"] = params
    return d


def predict_resp_many_in_one_to_many_singles(
    resp_json, status_code, batch_id, local_err, batch_size, response_key = "results"
):
    expected_root_key = response_key
    to_ret = []
    if not local_err and status_code and status_code == 200:
        list_of_individual_seq_results = resp_json[expected_root_key]
    elif local_err:
        list_of_individual_seq_results = [{"error": resp_json}]
    elif status_code and status_code != 200 and isinstance(resp_json, dict):
        list_of_individual_seq_results = [resp_json] * batch_size
    else:
        raise ValueError("Unexpected response in parser")
    for idx, item in enumerate(list_of_individual_seq_results):
        d = {"status_code": status_code, "batch_id": batch_id, "batch_item": idx}
        if not status_code or status_code != 200:
            d.update(item)  # Put all resp keys at root there
        else:
            # We just append one item, mimicking a single seq in POST req/resp
            d[expected_root_key] = []
            d[expected_root_key].append(item)
        to_ret.append(d)
    return to_ret
