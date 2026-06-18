"""API inference classes."""
import warnings

warnings.warn(
    "The biolmai.core.legacy.cls module is deprecated. "
    "Please use biolmai.models.Model instead.",
    DeprecationWarning,
    stacklevel=2
)

from biolm.core.legacy.api import APIEndpoint, GenerateAction, PredictAction, TransformAction, EncodeAction
from biolm.core.validate import (AAExtended,
                              AAExtendedPlusExtra,
                              AAUnambiguous,
                              AAUnambiguousPlusExtra,
                              DNAUnambiguous,
                              SingleOrMoreOccurrencesOf,
                              SingleOccurrenceOf,
                              PDB,
                              AAUnambiguousEmpty
                              )


class ESMFoldSingleChain(APIEndpoint):
    slug = "esmfold-singlechain"
    action_classes = (PredictAction,)
    predict_input_classes = (AAUnambiguous(),)
    batch_size = 2


class ESMFoldMultiChain(APIEndpoint):
    slug = "esmfold-multichain"
    action_classes = (PredictAction,)
    predict_input_classes = (AAExtendedPlusExtra(extra=[":"]),)
    batch_size = 2


class ESM2(APIEndpoint):
    """Example.

    .. highlight:: python
    .. code-block:: python

       {
         "items": [{
           "sequence": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQ"
         }]
       }
    """

    action_classes = (EncodeAction, PredictAction, )
    encode_input_classes = (AAUnambiguous(),)
    predict_input_classes = (SingleOrMoreOccurrencesOf(token="<mask>"), AAExtendedPlusExtra(extra=["<mask>"]))
    batch_size = 1

class ESM2_8M(ESM2):
    slug = "esm2-8m"

class ESM2_35M(ESM2):
    slug = "esm2-35m"

class ESM2_150M(ESM2):
    slug = "esm2-150m"

class ESM2_650M(ESM2):
    slug = "esm2-650m"

class ESM2_3B(ESM2):
    slug = "esm2-3b"

class ESM1v(APIEndpoint):
    """Example.

    .. highlight:: python
    .. code-block:: python

       {
          "items": [{
            "sequence": "QERLEUTGR<mask>SLGYNIVAT"
          }]
       }
    """
    action_classes = (PredictAction,)
    predict_input_classes = (SingleOccurrenceOf("<mask>"), AAExtendedPlusExtra(extra=["<mask>"]))
    batch_size = 5

class ESM1v1(ESM1v):
    slug = "esm1v-n1"

class ESM1v2(ESM1v):
    slug = "esm1v-n2"

class ESM1v3(ESM1v):
    slug = "esm1v-n3"

class ESM1v4(ESM1v):
    slug = "esm1v-n4"

class ESM1v5(ESM1v):
    slug = "esm1v-n5"

class ESM1vAll(ESM1v):
    slug = "esm1v-all"

class ESMIF1(APIEndpoint):
    slug = "esm-if1"
    action_classes = (GenerateAction,)
    generate_input_classes = PDB
    batch_size = 2
    generate_input_key = "pdb"


class ProGen2(APIEndpoint):
    action_classes = (GenerateAction,)
    generate_input_classes = (AAUnambiguousEmpty(),)
    batch_size = 1

class ProGen2Oas(ProGen2):
    slug = "progen2-oas"

class ProGen2Medium(ProGen2):
    slug = "progen2-medium"

class ProGen2Large(ProGen2):
    slug = "progen2-large"

class ProGen2BFD90(ProGen2):
    slug = "progen2-bfd90"

class AbLang(APIEndpoint):
    action_classes = (PredictAction, EncodeAction, GenerateAction,)
    predict_input_classes = (AAUnambiguous(),)
    encode_input_classes = (AAUnambiguous(),)
    generate_input_classes = (SingleOrMoreOccurrencesOf(token="*"), AAUnambiguousPlusExtra(extra=["*"]))
    batch_size = 32
    generate_input_key = "sequence"

class AbLangHeavy(AbLang):
    slug = "ablang-heavy"

class AbLangLight(AbLang):
    slug = "ablang-light"

class DNABERT(APIEndpoint):
    slug = "dnabert"
    action_classes = (EncodeAction,)
    encode_input_classes = (DNAUnambiguous(),)
    batch_size = 10

class DNABERT2(APIEndpoint):
    slug = "dnabert2"
    action_classes = (EncodeAction,)
    encode_input_classes = (DNAUnambiguous(),)
    batch_size = 10

class BioLMToxV1(APIEndpoint):
    """Example.

    .. highlight:: python
    .. code-block:: python

       {
         "instances": [{
           "data": {"text": "MSILVTRPSPAGEELVSRLRTLGQVAWHFPLIEFSPGQQLPQ"}
         }]
       }
    """

    slug = "biolmtox_v1"
    action_classes = (TransformAction, PredictAction,)
    predict_input_classes = (AAUnambiguous(),)
    transform_input_classes = (AAUnambiguous(),)
    batch_size = 1
    api_version = 1

class ProteInfer(APIEndpoint):
    action_classes = (PredictAction,)
    predict_input_classes = (AAExtended(),)
    batch_size = 64

class ProteInferEC(ProteInfer):
    slug = "proteinfer-ec"

class ProteInferGO(ProteInfer):
    slug = "proteinfer-go"
