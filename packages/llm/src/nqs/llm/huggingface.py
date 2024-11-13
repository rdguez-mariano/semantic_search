import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from transformers.generation import GenerationMixin
from transformers.generation.logits_process import LogitsProcessorList
from transformers.tokenization_utils import PreTrainedTokenizer

from nqs.llm.common import (
    EmbeddingModelName,
    HuggingFaceModelName,
    HuggingFaceModelPrecision,
    LlmGenConfig,
    LlmModel,
    LlmResponse,
    ModelSetup,
    embedding_model_setups,
    model_setups,
)


class HuggingFaceLlm(LlmModel):
    def __init__(self, name: HuggingFaceModelName) -> None:
        super().__init__(name)
        self.model_setup = model_setups[name]
        self.model = get_casual_lm(name)
        self.tokenizer = get_tokenizer(name)
        self.config_key_mapping = {"max_output_tokens": "max_new_tokens"}

    def chat_generate(
        self,
        texts: List[List[Dict[str, str]]],
        llm_config: LlmGenConfig = LlmGenConfig(),
    ) -> List[LlmResponse]:

        prompts = [
            self.tokenizer.apply_chat_template(
                text,
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in texts
        ]

        return self._instruct_generate_v2(prompts, llm_config)

    def instruct_generate(
        self, texts: List[str], llm_config: LlmGenConfig = LlmGenConfig()
    ) -> List[LlmResponse]:
        return self._instruct_generate_v2(texts, llm_config)

    def _extract_model_config(
        self, llm_config: LlmGenConfig
    ) -> Dict[Any, Any]:
        if (
            isinstance(llm_config.temperature, float)
            and llm_config.temperature == 1.0
        ):
            do_sample = False
        else:
            do_sample = True
        config = {"do_sample": do_sample}
        for k, v in asdict(llm_config).items():
            if k in self.config_key_mapping.keys():
                newk = self.config_key_mapping[k]
            else:
                newk = k
            config[newk] = v
        return config

    def _instruct_generate_v2(
        self, texts: List[str], llm_config: LlmGenConfig = LlmGenConfig()
    ) -> List[LlmResponse]:
        kwargs = self._extract_model_config(llm_config)
        llm_pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            # do_sample=True,
            # temperature=0.2,
            # repetition_penalty=1.1,
            # max_new_tokens=500,
            return_full_text=False,
            **kwargs,
        )
        llm_outputs = llm_pipe(texts)
        responses = [
            LlmResponse(prompt=text, out_text=poutput[0]["generated_text"])
            for text, poutput in zip(texts, llm_outputs)
        ]
        return responses

    def _get_model_arguments(
        self,
        inputs: Any = None,
    ) -> Tuple[List[Any], Dict[Any, Any]]:
        if inputs is None:
            inputs = self.model_setup.input_example
        model_args = []
        model_kwargs = {}
        if self.model_setup.is_chat_model:
            model_args = [
                self.tokenizer.apply_chat_template(
                    inputs, return_tensors="pt"
                ).to(self.model_setup.device)
            ]
        else:
            model_kwargs.update(
                self.tokenizer(inputs, return_tensors="pt").to(
                    self.model_setup.device
                )
            )
        return model_args, model_kwargs

    def _instruct_generate_v1(
        self,
        texts: List[str],
        llm_config: LlmGenConfig = LlmGenConfig(),
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = 150,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = True,
        num_beams: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
    ) -> List[LlmResponse]:
        inputs = texts

        extra_kwargs = self._extract_model_config(llm_config)
        model_args, model_kwargs = self._get_model_arguments(inputs)
        model_kwargs.update(extra_kwargs)

        raw_outputs = self.model.generate(
            *model_args,
            **model_kwargs,
        )
        outputs = self.tokenizer.batch_decode(raw_outputs)
        responses = [
            LlmResponse(prompt=text, out_text=output)
            for text, output in zip(texts, outputs)
        ]
        return responses

    def choose_best_from(
        self, prompts: List[Any], tentatives: List[List[Any]]
    ) -> List[int]:
        return [
            self._choose_best_from(prompt, tentative_group)
            for prompt, tentative_group in zip(prompts, tentatives)
        ]

    def _choose_best_from(self, context: str, candidates: List[str]) -> int:
        # adapted from: GenerationMixin._sample
        context_tokens = (
            self.tokenizer(context, return_tensors="pt")
            .to(self.model_setup.device)
            .data["input_ids"]
        )

        model_kwargs = {
            "attention_mask": torch.ones(context_tokens.shape),
            "use_cache": True,
        }

        generation_config = self.model.generation_config
        # logits_warper = self.model._get_logits_warper(generation_config)
        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=context_tokens.shape[1],
            encoder_input_ids=context_tokens,
            logits_processor=[],
            device=context_tokens.device,
            model_kwargs=model_kwargs,
            prefix_allowed_tokens_fn=None,
        )

        # loop for each candidate
        scores = [
            self._choose_compute_probs(
                context_tokens,
                candidate,
                logits_processor,
                copy.deepcopy(model_kwargs),
            )
            for candidate in candidates
        ]

        return np.argmax(scores).item()

    def _choose_compute_probs(
        self,
        context_tokens: torch.Tensor,
        candidate: str,
        logits_processor: LogitsProcessorList,
        input_model_kwargs: Dict[Any, Any],
    ) -> float:
        candidate_tokens = (
            self.tokenizer(candidate, return_tensors="pt")
            .to(self.model_setup.device)
            .data["input_ids"]
        )
        input_ids = context_tokens

        # keep track of which sequences are already finished
        model_kwargs = self.model._get_initial_cache_position(
            input_ids, input_model_kwargs
        )

        candidate_logprobs = torch.zeros((1, 1), device=self.model.device)
        for token in candidate_tokens[0]:
            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids, **model_kwargs
            )

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # for mistral, the following is a topK warper, which puts to -inf
            # all but the top k scores
            # next_token_scores = logits_warper(input_ids, next_token_scores)

            # token selection
            # probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            log_probs = torch.nn.functional.log_softmax(
                next_token_scores, dim=-1
            )
            candidate_logprobs = torch.cat(
                [candidate_logprobs, log_probs[0][token].view(1, 1)], dim=-1
            )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, token.view(1, 1)], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
            )

        sum_logprobs = candidate_logprobs.sum()

        # normalized log likelihood (beamsearch like)
        score = sum_logprobs / (
            candidate_tokens.shape[-1]
            ** self.model.generation_config.length_penalty
        )

        return score.cpu().detach().item()


def get_embedding_model(name: EmbeddingModelName) -> HuggingFaceEmbeddings:
    emb_model_setup = embedding_model_setups[name]

    # default kwargs
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {
        "normalize_embeddings": True,  # Set `True` for cosine similarity
    }

    # complete kwargs
    encode_kwargs.update(emb_model_setup.encode_kwargs)
    model_kwargs.update(emb_model_setup.model_kwargs)

    embedding_model = HuggingFaceEmbeddings(
        model_name=emb_model_setup.model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # PATCH: can this avoid sequence lenghts > 512 without killing acc ?
    # if name == EmbeddingModelName.LAJAVANESS_CAMEMBERT_LARGE:
    #     embedding_model.client.max_seq_length = 512
    #     embedding_model.client.tokenizer.model_max_length = 512
    return embedding_model


def get_precision_kwargs_casual(ms: ModelSetup) -> Dict[Any, Any]:
    # different setups as in: https://huggingface.co/google/gemma-7b
    casual_kwargs = {}
    if ms.casual_kwargs is not None:
        casual_kwargs.update(ms.casual_kwargs)
    if ms.device == "cuda":
        if ms.precision == HuggingFaceModelPrecision.FP16:
            casual_kwargs.update({"device_map": "auto", "revision": "float16"})
        elif ms.precision == HuggingFaceModelPrecision.BF16:
            casual_kwargs.update(
                {"device_map": "auto", "torch_dtype": torch.bfloat16},
            )
        elif ms.precision == HuggingFaceModelPrecision.FP8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            casual_kwargs.update({"quantization_config": quantization_config})
        elif ms.precision == HuggingFaceModelPrecision.FP4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            casual_kwargs.update({"quantization_config": quantization_config})
        elif ms.precision == HuggingFaceModelPrecision.FP4_NF4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            casual_kwargs.update({"quantization_config": quantization_config})
    return casual_kwargs


def get_hf_model_name(name: str) -> HuggingFaceModelName:
    for item in HuggingFaceModelName:
        if item.value == name:
            return item
    return None


def get_tokenizer(name: HuggingFaceModelName) -> PreTrainedTokenizer:
    model_setup = model_setups[name]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_setup.model_id
    )

    return tokenizer


def get_casual_lm(name: HuggingFaceModelName) -> GenerationMixin:
    model_setup = model_setups[name]
    model: GenerationMixin = AutoModelForCausalLM.from_pretrained(
        model_setup.model_id,
        **get_precision_kwargs_casual(model_setup),
    )

    return model


if __name__ == "__main__":
    hf_llm = HuggingFaceLlm(HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V3_STR)

    prompt = "France has a bread law, Le Décret Pain, with strict rules on \
what is allowed in a traditional baguette. Let's think step by step."
    candidate1 = "\nThe law does not apply to traditional baguettes."
    candidate2 = "\nThe law applies to traditional baguettes."
    prompts = [prompt]
    candidates = [[candidate1, candidate2]]
    bests = hf_llm.choose_best_from(prompts, candidates)

    for best, prompt, cgroup in zip(bests, prompts, candidates):
        print(prompt + cgroup[best])
