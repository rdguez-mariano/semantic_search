from dataclasses import asdict
from typing import Any, Dict, List

import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
)

from nqs.llm.common import (
    HuggingFaceModelName,
    LlmGenConfig,
    LlmModel,
    LlmResponse,
    model_setups,
)

PROJECT_ID = "resonant-truth-433914-g9"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)


class GcloudLlm(LlmModel):
    def __init__(self, name: HuggingFaceModelName) -> None:
        super().__init__(name)
        self.model_setup = model_setups[name]
        if self.model_setup.casual_kwargs["model_type"] == "GenerativeModel":
            self.model = GenerativeModel(self.model_setup.model_id)
            self.prompt_wrapper = lambda x: [x]
            self.model_caller = lambda x, config: self.model.generate_content(
                self.prompt_wrapper(x),
                generation_config=GenerationConfig(**asdict(config)),
            )
        elif (
            self.model_setup.casual_kwargs["model_type"]
            == "TextGenerationModel"
        ):
            self.model = TextGenerationModel.from_pretrained(
                self.model_setup.model_id
            )
            self.prompt_wrapper = lambda x: x
            self.model_caller = lambda x, config: self.model.predict(
                self.prompt_wrapper(x), **asdict(config)
            )
        self.tokenizer_name = HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V3_STR

    def instruct_generate(
        self, texts: List[Any], llm_config: LlmGenConfig = LlmGenConfig()
    ) -> List[LlmResponse]:
        outputs = []
        for text in texts:
            response = self.model_caller(text, llm_config)
            outputs.append(LlmResponse(prompt=text, out_text=response.text))
        return outputs

    def chat_generate(
        self,
        texts: List[List[Dict[Any, Any]]],
        llm_config: LlmGenConfig = LlmGenConfig(),
    ) -> List[LlmResponse]:
        raise NotImplementedError()

    def choose_best_from(
        self, prompts: List[Any], tentatives: List[List[Any]]
    ) -> List[int]:
        raise NotImplementedError()
