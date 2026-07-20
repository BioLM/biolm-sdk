"""Input context for protocol expression evaluation."""

from __future__ import annotations

from typing import Any


class InputContext:
    """Wraps user inputs for ${{ }} evaluation (inputs-only in v1)."""

    def __init__(self, inputs: dict[str, Any], protocol: dict | None = None):
        self.inputs = dict(inputs or {})
        self.protocol = protocol or {}
        self._merge_defaults()

    def _merge_defaults(self) -> None:
        specs = self.protocol.get("inputs") or []
        if isinstance(specs, dict):
            for name, spec in specs.items():
                if name in self.inputs:
                    continue
                if isinstance(spec, dict):
                    if "default" in spec:
                        self.inputs[name] = spec["default"]
                    elif "initial" in spec:
                        self.inputs[name] = spec["initial"]
            return
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not name or name in self.inputs:
                continue
            if "default" in spec:
                self.inputs[name] = spec["default"]
            elif "initial" in spec:
                self.inputs[name] = spec["initial"]

    def as_eval_context(self) -> dict[str, Any]:
        """Context dict passed to expression_evaluator."""
        return dict(self.inputs)

    def resolve_sequences(self) -> list[str]:
        """Resolve primary sequence list from inputs per Local Profile v1."""
        inputs = self.inputs
        specs = self.protocol.get("inputs") or []

        if isinstance(specs, dict):
            for name, spec in specs.items():
                if isinstance(spec, dict) and spec.get("type") == "list_of_str":
                    if name in inputs:
                        return self._coerce_sequence_list(inputs[name], name)
        else:
            for spec in specs:
                if not isinstance(spec, dict):
                    continue
                if spec.get("type") == "list_of_str":
                    name = spec.get("name")
                    if name and name in inputs:
                        return self._coerce_sequence_list(inputs[name], name)

        for key in ("sequences", "sequence"):
            if key in inputs:
                return self._coerce_sequence_list(inputs[key], key)

        text_names: list[str] = []
        if isinstance(specs, dict):
            for name, spec in specs.items():
                if isinstance(spec, dict) and spec.get("type") == "text":
                    text_names.append(name)
        else:
            text_names = [
                spec.get("name")
                for spec in specs
                if isinstance(spec, dict) and spec.get("type") == "text"
            ]
        if len(text_names) == 1:
            name = text_names[0]
            if name in inputs:
                return self._coerce_sequence_list(inputs[name], name)

        raise ValueError(
            "Could not resolve sequences from inputs. Provide 'sequences', "
            "'sequence', or a list_of_str input."
        )

    @staticmethod
    def _coerce_sequence_list(value: Any, key: str) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            out: list[str] = []
            for i, item in enumerate(value):
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and "sequence" in item:
                    out.append(str(item["sequence"]))
                else:
                    raise ValueError(
                        f"Input '{key}[{i}]' must be a string or dict with 'sequence'."
                    )
            if not out:
                raise ValueError(f"Input '{key}' must not be empty.")
            return out
        raise ValueError(f"Input '{key}' must be a string or list of strings.")
