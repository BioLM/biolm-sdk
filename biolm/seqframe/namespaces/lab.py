"""sf.lab — Lab-in-the-Loop (LLTP) bridge (stubs for v0)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biolm.seqframe.core import SeqFrame

_LLTP_MSG = (
    "LLTP integration is not yet implemented. "
    "SeqFrame.lab.to_lltp() and from_lltp() will bridge design-to-lab workflows in a future release."
)


class LabNamespace:
    def __init__(self, sf: "SeqFrame"):
        self._sf = sf

    def to_lltp(self, **kwargs: Any) -> Any:
        raise NotImplementedError(_LLTP_MSG)

    @classmethod
    def from_lltp(cls, *args: Any, **kwargs: Any) -> "SeqFrame":
        raise NotImplementedError(_LLTP_MSG)
