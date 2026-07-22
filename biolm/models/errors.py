"""Errors for BioLM definition recipes and builds."""


class RecipeError(ValueError):
    """Invalid or unreadable BioLM definition recipe."""


class BuildError(RuntimeError):
    """BioLM definition build failed."""
