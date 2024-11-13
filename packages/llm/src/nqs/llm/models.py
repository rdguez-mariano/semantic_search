from typing import List

from nqs.llm.common import (
    HuggingFaceModelName,
    LlmGenConfig,
    LlmModel,
    LlmResponse,
    ModelApi,
    model_setups,
)
from nqs.llm.gcloud import GcloudLlm
from nqs.llm.huggingface import HuggingFaceLlm


class GenLlmHandler:
    def __init__(self, name: HuggingFaceModelName) -> None:
        self.llm_model: LlmModel = None  # noqa
        self.model_setup = model_setups[name]
        if model_setups[name].api == ModelApi.HF_LOCAL:
            self.llm_model = HuggingFaceLlm(name)
        if model_setups[name].api == ModelApi.GCLOUD:
            self.llm_model = GcloudLlm(name)

    def generate(
        self, texts: List[str], llm_config: LlmGenConfig
    ) -> List[LlmResponse]:
        if self.model_setup.is_chat_model:
            outputs = self.llm_model.chat_generate(texts, llm_config)
        else:
            outputs = self.llm_model.instruct_generate(texts, llm_config)
        return outputs


if __name__ == "__main__":
    llm_handler = GenLlmHandler(
        HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V3_STR
    )
    print(
        llm_handler.generate(
            [llm_handler.model_setup.input_example], LlmGenConfig()
        )
    )
