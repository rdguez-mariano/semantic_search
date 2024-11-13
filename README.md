# Linkup tech test

## Launch the streamlit app

Ensure you have huggingface (https://huggingface.co/) and ovh endpoints (https://endpoints.ai.cloud.ovh.net/) access tokens. Both are free at the moment.

```bash
export HF_TOKEN=<your-hugging-face-token>
export OVH_AI_ENDPOINTS_ACCESS_TOKEN=<your-ovh-token>
docker compose -f docker/docker-compose.yaml up --build --detach
```

Then, with your broswer go to: http://localhost:8502

## Overview

This source code was modified from one of my own projects that is reachable at https://nqs.sytes.net. This explains naming and structure.

### Packages

1. [App](./packages/app). Contains the frontend.
2. [LLM](./packages/llm). All about the small scraping unit, computing embeddings of articles, usage of vector DB (FAISS), and RAG logics for generation of links.

#### LLM models

Multiple choices of LLMs for grading and for generation. We have google's vertex AI, Ollama (local), HuggingFace (local), OVH endpoints, etc. To use them all you need to have API tokens or setup Ollama locally (there's docker image).

#### Embeddings

The embeddings are computed locally here. It uses a version of CamemBERT so that it properly embeds french.
