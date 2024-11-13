# nqs-llm

The handling of LLM in NQS.

## Ollama

```bash
# volume on this path
docker run -d --gpus=all -v ./ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# volume on docker local driver
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

### run a model

```bash
docker exec -it ollama ollama run mistral
```

## Google Cloud Platform (GCP)

### recreate Application Default Credentials (ADC)

```bash
gcloud auth application-default login
```

## Related repos

* [https://github.com/Nagi-ovo/CRAG-Ollama-Chat](CRAG-Ollama-Chat).

## Related links

* [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
* [transformers on github](https://github.com/huggingface/transformers)
* [mistralai on huggingface](https://huggingface.co/docs/transformers/en/model_doc/mistral#mistral)
* [LLM prompting guide on HF](https://huggingface.co/docs/transformers/main/en/tasks/prompting)
* [Huggingface Tasks](https://huggingface.co/tasks)

### HF docs on models

* [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
* [google/gemma-7b](https://huggingface.co/google/gemma-7b)

### Text classification

* [HF Encoder models](https://huggingface.co/learn/nlp-course/en/chapter1/5)
* [HF text classification](https://huggingface.co/docs/transformers/en/tasks/sequence_classification)
* [Notebooks for fine-tunning Llama 2](https://huggingface.co/docs/transformers/main/en/model_doc/llama2#resources)

### Multiple choice

* [HF Multiple choice](https://huggingface.co/docs/transformers/en/tasks/multiple_choice)
* [HF Notebook](https://github.com/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb)

### RAG

* [advanced RAG HF](https://huggingface.co/learn/cookbook/en/advanced_rag)

### vector DBs

* [How they work](https://www.pinecone.io/learn/vector-database/)
* [Langchain suggestions](https://js.langchain.com/v0.1/docs/modules/data_connection/vectorstores/#which-one-to-pick)

## Using Google cloud

[Gemini 1.5 Flash](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/gemini-1.5-flash-preview-0514?project=practical-now-320510&supportedpurview=project) main page (it has code snippets and more).
