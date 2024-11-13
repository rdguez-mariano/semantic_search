from typing import List, Optional

from pydantic import BaseModel

from nqs.common.logger import LogLevel, log_msg
from nqs.llm.rag import (
    ModelName,
    data_retriever_handler,
    get_cgp_bot_graph,
    get_model_name,
)
from nqs.llm.rag_retriever import DocMetadata, scrape_new_data


class LapiCgpBotParams(BaseModel):
    question: str = (
        "Quel est le nom de la personne qui a chanté avec Slimane ?"
    )
    grade_model: Optional[str] = ModelName.MISTRAL_OVH.value
    generate_model: Optional[str] = ModelName.MISTRAL_OVH.value
    k_best_docs: int = 10
    min_graded_docs: int = 1
    max_docs_for_generation: int = 10


class LapiCgpBotResponse(BaseModel):
    answer: Optional[str] = None
    example_questions: Optional[List[str]] = None
    source_docs: Optional[List[DocMetadata]] = None


cgp_bot = get_cgp_bot_graph()


def generate_answer(question: str) -> LapiCgpBotResponse:
    params = LapiCgpBotParams(
        question=question,
    )
    if params.grade_model is None:
        params.grade_model = ModelName.PALM_2_VERTEXAI.value
    if params.generate_model is None:
        params.generate_model = ModelName.PALM_2_VERTEXAI.value

    params.grade_model = get_model_name(params.grade_model)
    params.generate_model = get_model_name(params.generate_model)

    if params.grade_model is None or params.generate_model is None:
        log_msg(
            f"unsupported models: {params.grade_model=}, "
            f"{params.generate_model=}",
            into_level=LogLevel.ERROR,
        )
        return LapiCgpBotResponse()

    bot_output = cgp_bot.invoke(params.model_dump())
    output = bot_output["output"]

    if isinstance(output, str):
        source_documents = bot_output["output_source_documents"]
        docs = [DocMetadata(**sd.metadata) for sd in source_documents]

        llmresponse = LapiCgpBotResponse(
            answer=output,
            source_docs=docs,
        )
    elif isinstance(output, list):
        llmresponse = LapiCgpBotResponse(example_questions=output)
    else:
        log_msg("not implemented output", into_level=LogLevel.ERROR)
        llmresponse = LapiCgpBotResponse()
    return llmresponse


def scrape_and_initialize_retriever():
    scrape_new_data()
    data_retriever_handler.initialize()
