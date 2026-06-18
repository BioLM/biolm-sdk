"""
Unit tests for RemaskingConfig with model_name parameter.
"""
import pytest
pytest.importorskip("pandas")

import pytest

from biolm.pipeline.mlm_remasking import (
    AGGRESSIVE_CONFIG,
    CONSERVATIVE_CONFIG,
    MODERATE_CONFIG,
    MLMRemasker,
    RemaskingConfig,
)


class TestRemaskingConfigModelName:
    """Test RemaskingConfig model_name parameter."""

    def test_default_model(self):
        """Test default model is ESM2 150M."""
        config = RemaskingConfig()
        assert config.model_name == "esm-150m"
        print(f"  ✓ Default model: {config.model_name}")

    def test_custom_model(self):
        """Test setting custom model."""
        config = RemaskingConfig(model_name="esm-650m")
        assert config.model_name == "esm-650m"
        print(f"  ✓ Custom model: {config.model_name}")

    def test_esm3_model(self):
        """Test ESM3 model."""
        config = RemaskingConfig(model_name="esm3", mask_fraction=0.20, temperature=1.5)
        assert config.model_name == "esm3"
        assert config.mask_fraction == 0.20
        assert config.temperature == 1.5
        print(f"  ✓ ESM3 model: {config.model_name}")

    def test_esmc_model(self):
        """Test ESMC model."""
        config = RemaskingConfig(model_name="esmc")
        assert config.model_name == "esmc"
        print(f"  ✓ ESMC model: {config.model_name}")

    def test_preset_configs_have_model(self):
        """Test preset configs have model_name set."""
        assert CONSERVATIVE_CONFIG.model_name == "esm-150m"
        assert MODERATE_CONFIG.model_name == "esm-150m"
        assert AGGRESSIVE_CONFIG.model_name == "esm-150m"

        print(f"  ✓ CONSERVATIVE: {CONSERVATIVE_CONFIG.model_name}")
        print(f"  ✓ MODERATE: {MODERATE_CONFIG.model_name}")
        print(f"  ✓ AGGRESSIVE: {AGGRESSIVE_CONFIG.model_name}")

    def test_remasker_uses_config_model(self):
        """Test MLMRemasker uses model from config."""
        config = RemaskingConfig(model_name="esm-650m")
        remasker = MLMRemasker(config)

        assert remasker.model_name == "esm-650m"
        print(f"  ✓ Remasker model from config: {remasker.model_name}")

    def test_remasker_override_model(self):
        """Test overriding model in MLMRemasker."""
        config = RemaskingConfig(model_name="esm-150m")
        remasker = MLMRemasker(config, model_name="esm3")

        # Override should take precedence
        assert remasker.model_name == "esm3"
        print(f"  ✓ Remasker model override: {remasker.model_name}")

    def test_config_with_all_params(self):
        """Test config with all parameters."""
        config = RemaskingConfig(
            model_name="esm-650m",
            mask_fraction=0.18,
            num_iterations=7,
            temperature=1.3,
            top_k=50,
            top_p=0.9,
            mask_strategy="blocks",
            block_size=5,
        )

        assert config.model_name == "esm-650m"
        assert config.mask_fraction == 0.18
        assert config.num_iterations == 7
        assert config.temperature == 1.3
        assert config.top_k == 50
        assert config.top_p == 0.9
        assert config.mask_strategy == "blocks"
        assert config.block_size == 5

        print(
            f"  ✓ Full config: model={config.model_name}, "
            f"mask={config.mask_fraction}, iter={config.num_iterations}"
        )


class TestDifferentESMModels:
    """Test different ESM model variants."""

    def test_esm2_models(self):
        """Test various ESM2 model sizes."""
        models = [
            "esm-150m",  # 150M (BioLM slug)
            "esm-650m",  # 650M
            "esm-3b",  # 3B
        ]

        for model in models:
            config = RemaskingConfig(model_name=model)
            assert config.model_name == model
            print(f"  ✓ {model}")

    def test_next_gen_models(self):
        """Test next-gen ESM models."""
        models = ["esm3", "esmc"]

        for model in models:
            config = RemaskingConfig(model_name=model)
            assert config.model_name == model
            print(f"  ✓ {model}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
