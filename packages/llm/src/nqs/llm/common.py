from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from transformers.tokenization_utils import PreTrainedTokenizer


class HuggingFaceModelPrecision(Enum):
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8 = "8bits"
    FP4 = "4bits"
    FP4_NF4 = "4bits-quanttype-nf4"
    FULL = "full"


class HuggingFaceModelName(Enum):
    GEMMA_2B = "gemma-2b"
    GEMMA_7B = "gemma-7b"
    LLAMA_3 = "llama-3"
    MISTRAL_7B_V1 = "mistral-7b-v0.1"
    MISTRAL_7B_V3 = "mistral-7b-v0.3"
    MISTRAL_7B_INSTRUCT_V1 = "mistral-7b-instruct-v0.1"
    MISTRAL_7B_INSTRUCT_V3 = "mistral-7b-instruct-v0.3"
    MISTRAL_7B_INSTRUCT_V3_STR = "mistral-7b-instruct-v0.3-str"
    PHI_3_MINI_INSTRUCT_4k = "phi-3-mini-4k-instruct"
    PHI_3_MINI_INSTRUCT_128k = "phi-3-mini-128k-instruct"
    GEMINI_FLASH_GCLOUD = "gemini_flash_gcloud"
    PALM2_TEXT_BISON_GCLOUD = "palm2_text_bison_gcloud"


class ModelApi(Enum):
    HF_LOCAL = "hf_local"
    GCLOUD = "gcloud"


@dataclass
class ModelSetup:
    model_id: str
    precision: HuggingFaceModelPrecision
    input_example: Any = None
    device: str = "cuda"
    casual_kwargs: Dict[str, Any] = None  # type: ignore
    is_chat_model: bool = False
    api: ModelApi = ModelApi.HF_LOCAL


model_setups = {
    HuggingFaceModelName.GEMMA_2B: ModelSetup(
        model_id="google/gemma-2b",
        precision=HuggingFaceModelPrecision.FP16,
    ),
    HuggingFaceModelName.GEMMA_7B: ModelSetup(
        model_id="google/gemma-7b",
        precision=HuggingFaceModelPrecision.FP4,
    ),
    HuggingFaceModelName.LLAMA_3: ModelSetup(
        model_id="meta-llama/Meta-Llama-3-8B",
        precision=HuggingFaceModelPrecision.FP4,
    ),
    HuggingFaceModelName.MISTRAL_7B_V1: ModelSetup(
        model_id="mistralai/Mistral-7B-v0.1",
        precision=HuggingFaceModelPrecision.FP4,
    ),
    HuggingFaceModelName.MISTRAL_7B_V3: ModelSetup(
        model_id="mistralai/Mistral-7B-v0.3",
        precision=HuggingFaceModelPrecision.FP4,
    ),
    HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V1: ModelSetup(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        precision=HuggingFaceModelPrecision.FP4,
    ),
    HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V3_STR: ModelSetup(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        precision=HuggingFaceModelPrecision.FP4_NF4,
        input_example="""
Tu es un agent immobilier et tu reçois un client dans ton agence.
Tu dois utiliser un langague formel.
Quels sont les trois meilleurs conseils à lui donner ?
""",
    ),
    HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V3: ModelSetup(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        precision=HuggingFaceModelPrecision.FP4_NF4,
        is_chat_model=True,
        input_example=[
            {"role": "user", "content": "What is your favourite condiment?"},
            {
                "role": "assistant",
                "content": "Well, I'm quite partial to a good squeeze of \
fresh lemon juice. It adds just the right amount of zesty flavour to whatever \
I'm cooking up in the kitchen!",
            },
            {"role": "user", "content": "Do you have mayonnaise recipes?"},
        ],
    ),
    HuggingFaceModelName.PHI_3_MINI_INSTRUCT_4k: ModelSetup(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        precision=HuggingFaceModelPrecision.FP4,
        casual_kwargs={
            "attn_implementation": "eager",
            "trust_remote_code": True,
        },
        input_example="""<|user|>
Tu es un agent immobilier et tu reçois un client dans ton agence.
Tu dois utiliser un langague formel.
Quels sont les trois meilleurs conseils à lui donner ?<|end|>
<|assistant|>""",
    ),
    HuggingFaceModelName.PHI_3_MINI_INSTRUCT_128k: ModelSetup(
        model_id="microsoft/Phi-3-mini-128k-instruct",
        precision=HuggingFaceModelPrecision.FP4,
        casual_kwargs={
            "attn_implementation": "eager",
            "trust_remote_code": True,
        },
        input_example="""<|user|>
Tu es un agent immobilier et tu reçois un client dans ton agence.
Tu dois utiliser un langague formel.
Quels sont les trois meilleurs conseils à lui donner ?<|end|>
<|assistant|>""",
    ),
}


class EmbeddingModelName(Enum):
    LAJAVANESS_CAMEMBERT_LARGE = "sentence-camembert-large-v2"


@dataclass
class EmbeddingModelSetup:
    model_id: str
    model_kwargs: Dict[Any, Any]
    encode_kwargs: Dict[Any, Any]


embedding_model_setups = {
    EmbeddingModelName.LAJAVANESS_CAMEMBERT_LARGE: EmbeddingModelSetup(
        model_id="Lajavaness/sentence-camembert-large",
        model_kwargs={},
        encode_kwargs={},
    )
}


@dataclass
class LlmResponse:
    prompt: str
    out_text: str
    prob: float = -1


@dataclass
class LlmGenConfig:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = 500


class LlmModel(ABC):
    def __init__(self, name: HuggingFaceModelName) -> None:
        self.model_name = name
        self.tokenizer_name = name
        self.tokenizer: PreTrainedTokenizer = None

    @abstractmethod
    def choose_best_from(
        self, prompts: List[Any], tentatives: List[List[Any]]
    ) -> List[int]:
        pass

    @abstractmethod
    def instruct_generate(
        self, texts: List[Any], llm_config: LlmGenConfig = LlmGenConfig()
    ) -> List[LlmResponse]:
        pass

    @abstractmethod
    def chat_generate(
        self,
        texts: List[List[Dict[Any, Any]]],
        llm_config: LlmGenConfig = LlmGenConfig(),
    ) -> List[LlmResponse]:
        pass
