# Customer Support Copilot — Dandes Electronics

A retrieval-augmented generation (RAG) customer-support assistant for Dandes Electronics. It answers questions from the supplied 50-document knowledge base, cites source IDs, and returns the required fallback answer when the information is not available.

This project compares three approaches:

-V1 — Basic RAG:one Chroma collection and top-5 retrieval.
-V2 — Advanced RAG:query transformation, metadata filtering, dense + BM25 hybrid retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
-V2 — Multi-Index RAG :separate policy, product, order, and ticket indexes, intent-based routing, hybrid retrieval, RRF, global reranking, and source-grounded answers.

## Knowledge base

The dataset is `data/dandes_electronics_kb.json`. It contains 50 records and must be used as supplied; do not modify or add records.

| `policy_index` | Return and shipping policies | 10 |
| `product_index` | Product FAQs and known issues | 20 |
| `order_index` | Order guides | 6 |
| `ticket_index` | Resolved support tickets | 14 |

## Requirements

- Windows 10/11 (commands below use PowerShell)
- Python 3.12 recommended for the tested setup
- An API key for the configured OpenAI-compatible endpoint and access to the configured chat and embedding models
- Internet access on first run to install packages and download the cross-encoder model

Install the Python packages from `requirements.txt` in the virtual environment.

## Setup 

Open a terminal at the project root (`customer_support_copilot`) and run:

py -3.12 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


If `.venv` already exists and works, activate it instead of creating it again:

.\.venv\Scripts\Activate

### Configure the API key

Create a file named `.env` in the project root. Add your own key and, if needed, model overrides supported by your endpoint:

OPENAI_API_KEY=your_api_key_here
EVALUATION_JUDGE_MODEL=gpt-6-luna
EVALUATION_EMBEDDING_MODEL=text-embedding-3-small

Replace `your_api_key_here` with your real key locally.

## Run the project

Run commands from the project root with `.venv` active. The first run may download the cross-encoder model.

### 1. Build the V1 vector store

python .\v1_basic_rag\1_offline_pipeline.py

This reads the dataset, creates one document per record, embeds it, and stores the documents in the `dandes_v1` Chroma collection.

### 2. Try V1 Basic RAG

python .\v1_basic_rag\2_online_pipeline.py

For conversational follow-up questions:

python .\v1_basic_rag\3_conversational_rag.py


### 3. Explore V2 Advanced RAG

Each technique file contains a demonstration of the relevant retrieval improvement:

python .\v2_advanced_rag\1_metadata_filtering.py
python .\v2_advanced_rag\2_hybrid_search.py
python .\v2_advanced_rag\3_reranking.py
python .\v2_advanced_rag\4_query_transformation.py

Run the combined V2 Advanced pipeline:

python .\v2_advanced_rag\5_v2_pipeline.py

The pipeline uses the same `dandes_v1` collection as V1, with advanced retrieval techniques applied before answer generation.

### 4. Build and try Multi-Index RAG

Build the four Chroma/BM25 indexes:

python .\v2_multi_index\1_build_indexes.py

Then run the multi-index copilot and choose the demo option to process the seven assignment questions. The script saves the results to `demo_output.txt`:

python .\v2_multi_index\4_v2_multi_index_pipeline.py

Follow the prompt and select **1** for the seven assignment questions or **2** to ask your own questions. The router chooses the relevant index or indexes. Unsupported questions should skip retrieval and return the exact fallback sentence:

`I don't know based on the available information.`

## Run the evaluation

The golden test set is `evaluation/golden_test_set.json` (20 questions, including in-scope, multi-source, and out-of-scope questions). Run these commands in order from the project root:

python .\evaluation\1_run_versions.py
python .\evaluation\2_retrieval_metrics.py
python .\evaluation\3_generation_metrics.py
python .\evaluation\4_comparison_table.py

The first command runs all 20 questions through all three RAG versions and saves their answers, retrieved document IDs, cited sources, and routing decisions. The generation-metrics script checkpoints completed questions and reuses those scores on a normal rerun; use `--force` only when you intentionally want to rescore.

To smoke-test the generation metrics on the first in-scope question per version before a full run:

python .\evaluation\3_generation_metrics.py 

### Evaluation outputs

The scripts create these files under `evaluation/results/`:

- `v1_results.json`
- `v2_single_index_results.json`
- `v2_multi_index_results.json`
- `retrieval_metrics.json` (plus a convenience CSV)
- `generation_metrics.json` (plus summary and per-question CSVs)

The required comparison table is saved at:

- `evaluation/comparison_table.csv`

