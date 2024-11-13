import gc
import os
import time
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union

import torch
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.llms.ollama import Ollama
from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_vertexai import VertexAI
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from transformers import pipeline

from nqs.common.logger import (
    LogLevel,
    log_entering_fn,
    log_exiting_fn,
    log_msg,
)
from nqs.llm.huggingface import get_casual_lm, get_hf_model_name, get_tokenizer
from nqs.llm.parsers import (
    FirstLineExclusiveBoolOutputParser,
    FrenchMarkdownListOutputParser,
    OutputAndSourceParser,
)
from nqs.llm.rag_reader import (
    PROMPT_ANSWER_FROM_DOCS_WITH_SOURCES,
    PROMPT_FRENCH_GENERATE_QUESTIONS,
    PROMPT_FRENCH_GRADER,
)
from nqs.llm.rag_retriever import DataRetrieverHandler


class ModelName(Enum):
    MISTRAL_OLLAMA = "mistral"
    GEMINI_FLASH_VERTEXAI = "gemini-1.5-flash-001"
    PALM_2_VERTEXAI = "text-bison@001"
    LLAMA3_405B_VERTEXAI = "llama3-405b-instruct-maas"
    LLAMA3_8B_HF = "llama-3"
    MISTRAL_HF = "mistral-7b-instruct-v0.3-str"
    MISTRAL_OVH = "Mistral-7B-Instruct-v0.2"


class OnGradingDecision(Enum):
    ENOUGH_DATA = "enough_data"
    FEW_DATA = "few_data"
    NO_DATA = "no_data"
    NO_ENOUGH_DATA = "no_enough_data"


class GraphState(TypedDict):
    question: str
    context: str
    possible_answers: List[str]
    grade_model: ModelName
    generate_model: ModelName
    k_best_docs: int
    min_graded_docs: int
    max_docs_for_generation: int
    relevant_documents: List[Document]
    unrelevant_documents: List[Document]
    documents_to_grade: List[Document]
    output: Union[str, List[str], Dict[Any, Any]]
    output_source_documents: List[Document]


def get_model_name(name: str) -> Optional[ModelName]:
    return ModelName(name)


MERGE_CONTEXT = False
OLLAMA_SERVER_ENV_VAR = "OLLAMA_SERVER"
OLLAMA_SERVER = os.environ.get(OLLAMA_SERVER_ENV_VAR, "localhost")
OLLAMA_PORT_ENV_VAR = "OLLAMA_PORT"
OLLAMA_PORT = os.environ.get(OLLAMA_PORT_ENV_VAR, "11434")
OVH_AI_ENDPOINTS_ACCESS_TOKEN_ENV_VAR = "OVH_AI_ENDPOINTS_ACCESS_TOKEN"
OVH_AI_ENDPOINTS_ACCESS_TOKEN = os.environ.get(
    OVH_AI_ENDPOINTS_ACCESS_TOKEN_ENV_VAR, ""
)

data_retriever_handler = DataRetrieverHandler()


class Waiter:
    VERTEX_AI_PERIOD = 60
    VERTEX_AI_QUOTA = 5

    def __init__(self) -> None:
        self.records: Dict[str, List[float]] = {}

    def checkout_model(self, model_name: ModelName):
        if "VERTEXAI" in model_name.name:
            if model_name.value in self.records:
                last_records = self.records[model_name.value][
                    -Waiter.VERTEX_AI_QUOTA :
                ]
                if len(last_records) == Waiter.VERTEX_AI_QUOTA:
                    elapsed = time.time() - last_records[0]

                    if elapsed < Waiter.VERTEX_AI_PERIOD:
                        _wait = Waiter.VERTEX_AI_PERIOD - elapsed
                        log_msg(f"{_wait=} seconds for vertex ai")
                        time.sleep(_wait)
            else:
                self.records[model_name.value] = []
            self.records[model_name.value].append(time.time())


waiter = Waiter()


def get_llm(
    model_name: ModelName,
    temperature: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
) -> BaseLanguageModel:
    llm: BaseLanguageModel = None
    if model_name in [ModelName.MISTRAL_OLLAMA]:
        llm = Ollama(
            model=model_name.value,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_predict=max_new_tokens,
            base_url=f"http://{OLLAMA_SERVER}:{OLLAMA_PORT}",
        )
    elif model_name in [
        ModelName.GEMINI_FLASH_VERTEXAI,
        ModelName.PALM_2_VERTEXAI,
        ModelName.LLAMA3_405B_VERTEXAI,
    ]:
        llm = VertexAI(
            model_name=model_name.value,
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            verbose=True,
        )
    elif model_name in [ModelName.MISTRAL_OVH]:
        kwargs = {}
        if top_p is not None:
            kwargs.update({"top_p": top_p})
        if temperature is not None:
            kwargs.update({"temperature": temperature})
        if top_k is not None:
            log_msg(
                "top_k is not defined for ChatMistralAI",
                into_level=LogLevel.WARNING,
            )
        llm = ChatMistralAI(
            model=model_name.value,  # type: ignore
            api_key=OVH_AI_ENDPOINTS_ACCESS_TOKEN,  # type: ignore
            endpoint="https://mistral-7b-instruct-v02.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",  # type: ignore # noqa
            max_tokens=max_new_tokens,
            verbose=True,
            streaming=True,
            **kwargs,
        )
    elif model_name in [ModelName.LLAMA3_8B_HF, ModelName.MISTRAL_HF]:
        hf_model_name = get_hf_model_name(model_name.value)
        tokenizer = get_tokenizer(hf_model_name)
        model = get_casual_lm(hf_model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            temperature=temperature,
            # do_sample=True,
            # repetition_penalty=1.1,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"{model_name=} doesn't have an api yet")
    return llm


def format_docs(docs):
    return "\n\n".join(
        [
            f"Document {str(i)}:\n" + doc.page_content
            for i, doc in enumerate(docs)
        ]
    )


def retrieve(state: GraphState) -> GraphState:
    log_entering_fn()
    question = state["question"]
    k_best_docs = state["k_best_docs"]

    docs_and_scores = (
        data_retriever_handler.vectorstore().similarity_search_with_score(
            question, k=k_best_docs
        )
    )
    documents = []
    for doc, score in sorted(docs_and_scores, key=lambda x: x[1]):
        doc.metadata.update({"score": float(score)})
        documents.append(doc)
    gc.collect()
    torch.cuda.empty_cache()
    state.update(
        {
            "documents_to_grade": documents,
            "k_best_docs": 2 * state["k_best_docs"],  # for next iterations
        }
    )
    log_exiting_fn()
    return state


def grade_documents(state: GraphState) -> GraphState:
    log_entering_fn()
    question = state["question"]
    documents = state["documents_to_grade"]
    grade_model_name = state["grade_model"]

    llm = get_llm(grade_model_name, temperature=0)

    prompt = PromptTemplate(
        template=PROMPT_FRENCH_GRADER,
        input_variables=["context", "question"],
    )
    parser = FirstLineExclusiveBoolOutputParser()

    chain = prompt | llm | parser

    # Score
    filtered_docs = state["relevant_documents"] or []
    filtered_out_docs = state["unrelevant_documents"] or []
    analyzed_docs = filtered_docs + filtered_out_docs
    for d in documents:
        if d.page_content in [ad.page_content for ad in analyzed_docs]:
            continue
        waiter.checkout_model(grade_model_name)
        grade = chain.invoke(
            {
                "question": question,
                "context": d.page_content,
            }
        )
        if grade:
            log_msg("document relevant")
            filtered_docs.append(d)
        else:
            log_msg("document not relevant")
            filtered_out_docs.append(d)

    state.update(
        {
            "relevant_documents": filtered_docs,
            "unrelevant_documents": filtered_out_docs,
        }
    )
    log_exiting_fn()
    return state


def decide_next_node(state: GraphState) -> str:
    rel_docs = state["relevant_documents"]
    min_graded_docs = state.get("min_graded_docs", 3)

    if len(rel_docs) >= min_graded_docs:
        log_msg("decision: generate")
        return OnGradingDecision.ENOUGH_DATA.value
    elif len(rel_docs) == 0:
        log_msg("decision: generate example questions")
        return OnGradingDecision.NO_DATA.value
    else:
        log_msg("decision: re-retrieve")
        if state["k_best_docs"] > 10:
            return OnGradingDecision.NO_ENOUGH_DATA.value
        else:
            return OnGradingDecision.FEW_DATA.value


def generate(state: GraphState) -> GraphState:
    log_entering_fn()
    question = state["question"]
    documents = state["relevant_documents"]
    generate_model_name = state["generate_model"]
    max_docs_for_generation = state["max_docs_for_generation"]

    # Prompt
    prompt = PromptTemplate(
        template=PROMPT_ANSWER_FROM_DOCS_WITH_SOURCES,
        input_variables=["question", "context"],
    )

    # LLM Setup
    llm = get_llm(generate_model_name, temperature=0, max_new_tokens=500)

    # Chain
    rag_chain = prompt | llm | OutputAndSourceParser("Documents utilisés:\n")

    # Run
    input_documents = documents[-max_docs_for_generation:]
    waiter.checkout_model(generate_model_name)
    generation = rag_chain.invoke(
        {
            "context": format_docs(input_documents),
            "question": question,
        }
    )
    state.update(
        {
            "output": generation[0],
            "output_source_documents": [
                input_documents[i] for i in generation[1:]
            ],
        }
    )
    log_exiting_fn()
    return state


def generate_example_questions(state: GraphState) -> GraphState:
    log_entering_fn()
    generate_model_name = state["generate_model"]
    max_docs_for_generation = state["max_docs_for_generation"]
    documents = state["relevant_documents"] + state["unrelevant_documents"]
    documents = documents[:max_docs_for_generation]

    parser = FrenchMarkdownListOutputParser()

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template=PROMPT_FRENCH_GENERATE_QUESTIONS,
        input_variables=["context"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    llm = get_llm(generate_model_name, temperature=None)

    # Prompt
    chain = prompt | llm | parser
    waiter.checkout_model(generate_model_name)
    better_questions = chain.invoke({"context": format_docs(documents)})

    state.update({"output": better_questions})
    log_exiting_fn()
    return state


def empty_output(state: GraphState) -> GraphState:
    state["output"] = None  # type: ignore
    return state


def get_cgp_bot_graph() -> CompiledGraph:
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("example_questions", generate_example_questions)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_next_node,
        {
            OnGradingDecision.FEW_DATA.value: "retrieve",
            OnGradingDecision.ENOUGH_DATA.value: "generate",
            OnGradingDecision.NO_DATA.value: "example_questions",
            OnGradingDecision.NO_ENOUGH_DATA.value: "generate",
        },
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("example_questions", END)

    # Compile
    app = workflow.compile()
    return app
