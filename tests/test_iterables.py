"""Tests for iterable/generator support in items parameter."""

import pytest

from biolm import biolm
from biolm.client import BioLM
from biolm.core.http import BioLMApi, BioLMApiClient
from biolm.models import Model


def _gen_dicts(n):
    """Generator yielding n dict items."""
    for _ in range(n):
        yield {"sequence": "MSILV"}


def _gen_strings(n):
    """Generator yielding n sequence strings."""
    for _ in range(n):
        yield "MSILV"


class TestBioLMApiIterables:
    """BioLMApi accepts generators/iterators for items."""

    def test_encode_generator_of_dicts(self):
        model = BioLMApi("esm2-8m", raise_httpx=False)
        result = model.encode(items=_gen_dicts(3))
        model.shutdown()
        assert isinstance(result, list)
        assert len(result) == 3
        assert all("embeddings" in r for r in result)

    def test_encode_iterator(self):
        model = BioLMApi("esm2-8m", raise_httpx=False)
        items = iter([{"sequence": "MSILV"}, {"sequence": "MDNELE"}])
        result = model.encode(items=items)
        model.shutdown()
        assert len(result) == 2

    def test_str_rejected(self):
        model = BioLMApi("esm2-8m")
        with pytest.raises(TypeError, match="items"):
            model.encode(items="MSILV")
        model.shutdown()

    def test_empty_generator(self):
        model = BioLMApi("esm2-8m", raise_httpx=False)

        def empty():
            if False:
                yield  # make it a generator

        result = model.encode(items=empty())
        model.shutdown()
        assert result == []


class TestBioLMApiClientIterables:
    """BioLMApiClient accepts generators/iterators for items."""

    @pytest.mark.asyncio
    async def test_encode_generator_of_dicts(self):
        model = BioLMApiClient("esm2-8m", raise_httpx=False)
        result = await model.encode(items=_gen_dicts(3))
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_str_rejected(self):
        model = BioLMApiClient("esm2-8m")
        with pytest.raises(TypeError, match="items"):
            await model.encode(items="MSILV")


class TestBiolmIterables:
    """biolm() and BioLM accept generators/iterators for items."""

    def test_biolm_generator_of_dicts(self):
        result = biolm(entity="esm2-8m", action="encode", items=_gen_dicts(2))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_biolm_generator_of_strings_with_type(self):
        result = biolm(entity="esm2-8m", action="encode", type="sequence", items=_gen_strings(2))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_biolm_generator_consumed(self):
        def track_gen():
            for i in range(2):
                yield {"sequence": "MSILV"}

        gen = track_gen()
        result = biolm(entity="esm2-8m", action="encode", items=gen)
        assert len(result) == 2
        # Generator should be exhausted
        assert list(gen) == []

    def test_biolm_class_generator_of_dicts(self):
        """BioLM() (class) accepts generators for items, same as biolm()."""
        result = BioLM(entity="esm2-8m", action="encode", items=_gen_dicts(2), raise_httpx=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("embeddings" in r for r in result)


class TestModelIterables:
    """Model().encode() accepts generators/iterators for items."""

    def test_model_encode_generator_of_dicts(self):
        model = Model("esm2-8m", raise_httpx=False)
        result = model.encode(items=_gen_dicts(2), progress=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("embeddings" in r for r in result)

    def test_model_encode_generator_of_strings_with_type(self):
        model = Model("esm2-8m", raise_httpx=False)
        result = model.encode(items=_gen_strings(2), type="sequence", progress=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all("embeddings" in r for r in result)
