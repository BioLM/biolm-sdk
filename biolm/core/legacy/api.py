"""References to API endpoints."""
import warnings

warnings.warn(
    "The biolmai.core.legacy.api module is deprecated. "
    "Please use biolmai.models.Model instead.",
    DeprecationWarning,
    stacklevel=2
)

import datetime
import inspect
import logging
import time
from functools import lru_cache

log = logging.getLogger("biolm_util")

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pass

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from biolm.core import auth
from biolm.core.asynch import async_api_call_wrapper
from biolm.core.const import MULTIPROCESS_THREADS
from biolm.core.payloads import INST_DAT_TXT, PARAMS_ITEMS, predict_resp_many_in_one_to_many_singles


@lru_cache(maxsize=64)
def validate_endpoint_action(allowed_classes, method_name, api_class_name):
    action_method_name = method_name.split(".")[-1]
    if action_method_name not in allowed_classes:
        err = "Only {} supported on {}"
        err = err.format(list(allowed_classes), api_class_name)
        raise AssertionError(err)


def text_validator(text, c):
    """Validate some text against a class-based validator, returning a string
    if invalid, or None otherwise."""
    try:
        c(text)
    except Exception as e:
        return str(e)

def combine_validation(x, y):
    if x is None and y is None:
        return None
    elif isinstance(x, str) and y is None:
        return x
    elif x is None and isinstance(y, str):
        return y
    elif isinstance(x, str) and isinstance(y, str):
        return f"{x}\n{y}"


def validate_action(action):
    def validate(f):
        def wrapper(*args, **kwargs):
            # Get class instance at runtime, so you can access not just
            # APIEndpoints, but any *parent* classes of that,
            # like ESMFoldSinglechain.
            class_obj_self = args[0]
            try:
                is_method = inspect.getfullargspec(f)[0][0] == "self"
            except Exception:
                is_method = False

            # Is the function we decorated a class method?
            if is_method:
                name = f"{f.__module__}.{class_obj_self.__class__.__name__}.{f.__name__}"
            else:
                name = f"{f.__module__}.{f.__name__}"

            if is_method:
                # Splits name, e.g. 'biolmai.api.ESMFoldSingleChain.predict'
                action_method_name = name.split(".")[-1]
                validate_endpoint_action(
                    class_obj_self.action_class_strings,
                    action_method_name,
                    class_obj_self.__class__.__name__,
                )

            input_data = args[1]
            # Validate each row's text/input based on class attribute `seq_classes`
            if action == "predict":
                input_classes = class_obj_self.predict_input_classes
            elif action == "encode":
                input_classes = class_obj_self.encode_input_classes
            elif action == "generate":
                input_classes = class_obj_self.generate_input_classes
            elif action == "transform":
                input_classes = class_obj_self.transform_input_classes
            for c in input_classes:
                # Validate input data against regex
                if class_obj_self.multiprocess_threads:
                    validation = input_data.text.apply(text_validator, args=(c,))
                else:
                    validation = input_data.text.apply(text_validator, args=(c,))
                if "validation" not in input_data.columns:
                    input_data["validation"] = validation
                else:
                    # masking and loc may be more performant option
                    input_data["validation"] = input_data["validation"].combine(validation, combine_validation)

            # Mark your batches, excluding invalid rows
            valid_dat = input_data.loc[input_data.validation.isnull(), :].copy()
            N = class_obj_self.batch_size  # N rows will go per API request
            # JOIN back, which is by index
            if valid_dat.shape[0] != input_data.shape[0]:
                valid_dat["batch"] = np.arange(valid_dat.shape[0]) // N
                input_data = input_data.merge(
                    valid_dat.batch, left_index=True, right_index=True, how="left"
                )
            else:
                input_data["batch"] = np.arange(input_data.shape[0]) // N
            res = f(class_obj_self, input_data, **kwargs)
            return res

        return wrapper
    return validate

def convert_input(f):
    def wrapper(*args, **kwargs):
        # Get the user-input data argument to the decorated function
        # class_obj_self = args[0]
        input_data = args[1]
        # Make sure we have expected input types
        acceptable_inputs = (str, list, tuple, np.ndarray, pd.DataFrame)
        if not isinstance(input_data, acceptable_inputs):
            err = "Input must be one or many DNA or protein strings"
            raise ValueError(err)
        # Convert single-sequence input to list
        if isinstance(input_data, str):
            input_data = [input_data]
        # Make sure we don't have a matrix
        if isinstance(input_data, np.ndarray) and len(input_data.shape) > 1:
            err = "Detected Numpy matrix - input a single vector or array"
            raise AssertionError(err)
        # Make sure we don't have a >=2D DF
        if isinstance(input_data, pd.DataFrame) and len(input_data.shape) > 1:
            err = "Detected Pandas DataFrame - input a single vector or Series"
            raise AssertionError(err)
        input_data = pd.DataFrame(input_data, columns=["text"])
        return f(args[0], input_data, **kwargs)

    return wrapper


class APIEndpoint:
     # Overwrite in parent classes as needed
    batch_size = 3
    params = None
    action_classes = ()
    api_version = 3

    predict_input_key = "sequence"
    encode_input_key = "sequence"
    generate_input_key = "context"

    predict_input_classes = ()
    encode_input_classes = ()
    generate_input_classes = ()
    transform_input_classes = ()

    def __init__(self, multiprocess_threads=None, compress_requests=True, compress_threshold=256):
        # Check for instance-specific threads, otherwise read from env var
        if multiprocess_threads is not None:
            self.multiprocess_threads = multiprocess_threads
        else:
            self.multiprocess_threads = MULTIPROCESS_THREADS  # Could be False
        # Get correct auth-like headers
        self.auth_headers = auth.get_user_auth_header()
        self.action_class_strings = tuple(
            [c.__name__.replace("Action", "").lower() for c in self.action_classes]
        )
        self.compress_requests = compress_requests
        self.compress_threshold = compress_threshold

    def post_batches(self, dat, slug, action, payload_maker, resp_key, key="sequence", params=None):
        keep_batches = dat.loc[~dat.batch.isnull(), ["text", "batch"]]
        if keep_batches.shape[0] == 0:
            pass  # Do nothing - we made nice JSON errors to return in the DF
            # err = "No inputs found following local validation"
            # raise AssertionError(err)
        if keep_batches.shape[0] > 0:
            api_resps = async_api_call_wrapper(
                keep_batches, slug, action, payload_maker, resp_key, api_version=self.api_version, key=key, params=params, compress_requests=self.compress_requests, compress_threshold=self.compress_threshold,
            )
            if isinstance(api_resps, pd.DataFrame):
                batch_res = api_resps.explode("api_resp")  # Should be lists of results
                len_res = batch_res.shape[0]
            else:
                batch_res = pd.DataFrame({"api_resp": api_resps})
                len_res = batch_res.shape[0]
            orig_request_rows = keep_batches.shape[0]
            # For 'generate' actions, models may return multiple results per item
            # (e.g., hyper-mpnn with batch_size > 1), so skip the 1:1 check
            if action != "generate" and len_res != orig_request_rows:
                err = "Response rows ({}) mismatch with input rows ({})"
                err = err.format(len_res, orig_request_rows)
                raise AssertionError(err)

            # Stack the results horizontally w/ original rows of batches
            keep_batches["prev_idx"] = keep_batches.index
            keep_batches.reset_index(drop=False, inplace=True)
            batch_res.reset_index(drop=True, inplace=True)
            keep_batches["api_resp"] = batch_res
            keep_batches.set_index("prev_idx", inplace=True)
            dat = dat.join(keep_batches.reindex(["api_resp"], axis=1))
        else:
            dat["api_resp"] = None
        return dat

    def unpack_local_validations(self, dat, response_key):
        dat.loc[dat.api_resp.isnull(), "api_resp"] = (
            dat.loc[~dat.validation.isnull(), "validation"]
            .apply(
                predict_resp_many_in_one_to_many_singles, args=(None, None, True, None), response_key=response_key
            )
            .explode()
        )

        return dat

    @convert_input
    @validate_action("predict")
    def predict(self, dat, params=None):
        if self.api_version == 1:
            dat = self.post_batches(dat, self.slug, "predict", INST_DAT_TXT, "predictions")
            dat = self.unpack_local_validations(dat, "predictions")
        else:
            dat = self.post_batches(dat, self.slug, "predict", PARAMS_ITEMS, "results", key=self.predict_input_key, params=params)
            dat = self.unpack_local_validations(dat,"results")
        return dat.api_resp.replace(np.nan, None).tolist()

    def infer(self, dat, params=None):
        return self.predict(dat, params)

    @convert_input
    @validate_action("transform")  # api v1 legacy action
    def transform(self, dat):
        dat = self.post_batches(
            dat, self.slug, "transform", INST_DAT_TXT, "predictions"
        )
        dat = self.unpack_local_validations(dat,"predictions")
        return dat.api_resp.replace(np.nan, None).tolist()

    @convert_input
    @validate_action("encode")
    def encode(self, dat, params=None):

        dat = self.post_batches(dat, self.slug, "encode", PARAMS_ITEMS, "results", key=self.encode_input_key, params=params)
        dat = self.unpack_local_validations(dat, "results")
        return dat.api_resp.replace(np.nan, None).tolist()

    @convert_input
    @validate_action("generate")
    def generate(self, dat, params=None):
        if self.api_version == 1:
            dat = self.post_batches(dat, self.slug, "generate", INST_DAT_TXT, "generated")
            dat = self.unpack_local_validations(dat, "predictions")
        else:
            dat = self.post_batches(dat, self.slug, "generate", PARAMS_ITEMS, "results", key=self.generate_input_key, params=params)
            dat = self.unpack_local_validations(dat, "results")

        return dat.api_resp.replace(np.nan, None).tolist()


def retry_minutes(sess, URL, HEADERS, dat, timeout, mins):
    """Retry for N minutes."""
    HEADERS.update({"Content-Type": "application/json"})
    attempts, max_attempts = 0, 5
    try:
        now = datetime.datetime.now()
        try_until = now + datetime.timedelta(minutes=mins)
        while datetime.datetime.now() < try_until and attempts < max_attempts:
            response = None
            try:
                log.info(f"Trying {datetime.datetime.now()}")
                response = sess.post(URL, headers=HEADERS, data=dat, timeout=timeout)
                if response.status_code not in (400, 404):
                    response.raise_for_status()
                if "error" in response.json():
                    raise ValueError(response.json().dumps())
                else:
                    break
            except Exception as e:
                log.warning(e)
                if response:
                    log.warning(response.text)
                time.sleep(5)  # Wait 5 seconds between tries
            attempts += 1
        if response is None:
            err = "Got Nonetype response"
            raise ValueError(err)
        elif "Server Error" in response.text:
            err = "Got Server Error"
            raise ValueError(err)
    except Exception:
        return response
    return response


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=None,
    session=None,
):
    if status_forcelist is None:
        status_forcelist = list(range(400, 599))
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class PredictAction:
    def __str__(self):
        return "PredictAction"


class GenerateAction:
    def __str__(self):
        return "GenerateAction"


class TransformAction:
    def __str__(self):
        return "TransformAction"


class EncodeAction:
     def __str__(self):
         return "EncodeAction"


class ExplainAction:
    def __str__(self):
        return "ExplainAction"


class SimilarityAction:
    def __str__(self):
        return "SimilarityAction"


class FinetuneAction:
    def __str__(self):
        return "FinetuneAction"
