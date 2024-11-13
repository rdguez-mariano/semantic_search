import gc
import glob
import hashlib
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import requests  # type: ignore
import torch
from bs4 import BeautifulSoup
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters.base import Language
from transformers.tokenization_utils import PreTrainedTokenizer

from nqs.common.data import hash_from_file
from nqs.common.logger import (
    LogLevel,
    log_entering_fn,
    log_exiting_fn,
    log_msg,
)
from nqs.llm.huggingface import (
    EmbeddingModelName,
    get_embedding_model,
    get_tokenizer,
)
from nqs.llm.models import HuggingFaceModelName


@dataclass
class DocMetadata:
    title: str
    url: str
    thumb: str
    filename: str
    score: float = -1
    start_index: float = -1


NQS_LANGCENTER_DATAFOLDER_ENV_VAR = "NQS_LANGCENTER_DATAFOLDER"
NQS_LANGCENTER_DATAFOLDER = os.environ.get(
    NQS_LANGCENTER_DATAFOLDER_ENV_VAR, "./workspace/langcenter/"
)
RAW_FOLDER_NAME = "raw"
VECTOR_DBS_FOLDER_NAME = "vector_dbs"
ERROR_FOLDER_NAME = "errors"
# some kwargs are not persistent between save and load
VECTOR_DB_KWARGS = {
    "distance_strategy": DistanceStrategy.COSINE,
}
RAW_FOLDER = os.path.join(NQS_LANGCENTER_DATAFOLDER, RAW_FOLDER_NAME)

XML_SOURCES = [
    "https://vsd.fr/feed/",
    "https://vsd.fr/tele/feed/",
    "https://vsd.fr/loisirs/feed/",
    "https://vsd.fr/societe/feed/",
    "https://vsd.fr/culture/feed/",
    "https://vsd.fr/actu-people/feed/",
    "https://www.public.fr/feed",
    "https://www.public.fr/tele/feed",
    "https://www.public.fr/people/feed",
    "https://www.public.fr/mode/feed",
    "https://www.public.fr/people/familles-royales/feed",
    "https://www.public.fr/tag/exclusivite-public/feed",
]

# needed for public.fr
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; "
    "rv:92.0) Gecko/20100101 Firefox/92.0"
}


@dataclass
class RetrieverSource:
    name: str
    raw_filepaths: List[str]
    raw_hashs: List[str]
    load_method: str
    embedding_name: EmbeddingModelName
    vector_db_folder: str


class DataRetrieverHandler:
    def __init__(
        self,
        embedding_name: EmbeddingModelName = EmbeddingModelName.LAJAVANESS_CAMEMBERT_LARGE,  # noqa
        tokenizer_hf_name: HuggingFaceModelName = HuggingFaceModelName.MISTRAL_7B_INSTRUCT_V3_STR,  # noqa
        chunk_size: int = 512,
    ) -> None:
        self.embeddings_name = embedding_name
        self.embeddings = get_embedding_model(embedding_name)
        self.tokenizer_hf_name = tokenizer_hf_name
        self.reader_tokenizer = get_tokenizer(self.tokenizer_hf_name)
        self.chunk_size = chunk_size
        self.sources: List[RetrieverSource] = []
        self.initialize()

    def initialize(self):
        hash_of_files = []
        datafiles = []
        for datafile in sorted(glob.glob(RAW_FOLDER + "/*.xml")):
            filehash = hash_from_file(datafile)
            hash_of_files.append(filehash)
            datafiles.append(datafile)

        vecto_db_folder = os.path.join(
            NQS_LANGCENTER_DATAFOLDER,
            VECTOR_DBS_FOLDER_NAME,
            hashlib.sha256(
                ("-".join(hash_of_files) + self.embeddings_name.value).encode()
            ).hexdigest(),
        )
        rs = RetrieverSource(
            name="vsd+public",
            raw_filepaths=datafiles,
            raw_hashs=hash_of_files,
            load_method=get_docs_from_scraped_data.__name__,
            embedding_name=self.embeddings_name,
            vector_db_folder=vecto_db_folder,
        )
        self.sources = [rs]

        self.checkout_sources()
        self.load_sources()

    def checkout_sources(self):
        log_entering_fn()
        self.raw_docs = {}
        for rs in self.sources:
            self.raw_docs[rs.name] = rs.raw_filepaths

            if os.path.exists(rs.vector_db_folder):
                continue

            raw_knowledge_base = get_docs_from_scraped_data()
            log_msg(f"recreate vector db from {rs.raw_filepaths}")
            docs_processed = split_documents(
                self.chunk_size,
                raw_knowledge_base,
                reader_tokenizer=self.reader_tokenizer,
            )

            if (
                self.embeddings_name
                == EmbeddingModelName.LAJAVANESS_CAMEMBERT_LARGE
            ):
                # PATCH: filter out sequences len > 512 to avoid later bug
                # https://stackoverflow.com/questions/78037880/cuda-error-device-side-assert-triggered-on-tensor-todevice-cuda
                to_remove = get_camembert_problematic_ids(
                    [d.page_content for d in docs_processed], self.embeddings
                )
                if len(to_remove) > 0:
                    log_msg(
                        f"removing sequences ids {to_remove} due to sequence "
                        "length > 512 (it avoids later bug)",
                        into_level=LogLevel.WARNING,
                    )
                for idx in to_remove:
                    del docs_processed[idx]

            recreate_vector_db(
                rs.vector_db_folder, self.embeddings, docs_processed
            )
        torch.cuda.empty_cache()
        gc.collect()
        log_exiting_fn()

    def load_sources(self):
        log_entering_fn()
        self._vectorstores: Dict[str, FAISS] = {}
        for rs in self.sources:
            vectorstore: FAISS = load_vector_db(
                rs.vector_db_folder, self.embeddings
            )
            self._vectorstores[rs.name] = vectorstore
        log_exiting_fn()

    def vectorstore(self, name: Optional[str] = None) -> FAISS:
        if name is None:
            name = self.sources[0].name
        return self._vectorstores[name]


def get_camembert_problematic_ids(
    texts: List[str], embedding_model: HuggingFaceEmbeddings
) -> List[int]:
    tokenized = embedding_model.client.tokenize(texts)
    problematic_ids = tokenized["attention_mask"][:, -2:].sum(axis=1).nonzero()
    return sorted(problematic_ids.cpu().flatten().tolist())[::-1]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    reader_tokenizer: PreTrainedTokenizer,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens
    and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        reader_tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(
            Language.MARKDOWN
        ),
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


def recreate_vector_db(
    localpath: str,
    embedding_model: HuggingFaceEmbeddings,
    docs: List[LangchainDocument],
    vector_db_kwargs: Dict[str, Any] = VECTOR_DB_KWARGS,
) -> None:
    vector_db = FAISS.from_documents(
        docs,
        embedding_model,
        **vector_db_kwargs,
    )
    vector_db.save_local(localpath)


def load_vector_db(
    localpath: str,
    embedding_model: HuggingFaceEmbeddings,
    vector_db_kwargs: Dict[str, Any] = VECTOR_DB_KWARGS,
) -> FAISS:
    if os.path.exists(localpath):
        vector_db = FAISS.load_local(
            localpath,
            embedding_model,
            allow_dangerous_deserialization=True,
            **vector_db_kwargs,
        )
    else:
        vector_db = None
    return vector_db


def get_docs_from_scraped_data(
    raw_folder: str = RAW_FOLDER,
) -> List[LangchainDocument]:
    docs: List[LangchainDocument] = []
    seen_news = []
    for filename in glob.glob(f"{raw_folder}/**/*.xml", recursive=True):
        with open(filename, "r") as fp:
            source = fp.read()
        soup = BeautifulSoup(source, "lxml")
        news = soup.find_all("item")
        for article in news:
            url = article.find("guid").text
            media_thumb_element = article.find("media:thumbnail")
            media_thumb = (
                media_thumb_element["url"] if media_thumb_element else None
            )
            content = article.find("content:encoded").text
            title = article.find("title").text

            if title in seen_news:
                continue
            seen_news.append(title)

            doc = LangchainDocument(
                page_content=title + "\n\n" + content,
                metadata=asdict(
                    DocMetadata(
                        title=title,
                        url=url,
                        thumb=media_thumb,
                        filename=filename,
                    )
                ),
            )
            docs.append(doc)

    return docs


def scrape_new_data(raw_folder: str = RAW_FOLDER):
    os.makedirs(raw_folder, exist_ok=True)
    for source in XML_SOURCES:
        response = requests.get(source, headers=HEADERS)

        translationTable = str.maketrans("/:.-", "____")
        filename = os.path.join(
            raw_folder, source.lower().translate(translationTable) + ".xml"
        )
        with open(filename, "bw") as fp:
            fp.write(response.content)


if __name__ == "__main__":
    drh = DataRetrieverHandler()
