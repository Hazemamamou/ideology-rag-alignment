# Investigating LLMs' Alignment with Ideological Discourses in Retrieval-Augmented Generation

This repository contains the code and evaluation pipeline used in the
paper:

**Investigating LLMs' Alignment with Ideological Discourses in
Retrieval-Augmented Generation**

The project studies how large language models generate responses aligned
with specific ideological **dimensions**, under both:

-   **NoRAG** settings
-   **RAG** settings

Each question is associated with:

-   a **dimension label**
-   a **dimension description**
-   **typical vocabulary** representing ideological discourse
-   **example texts** used for evaluation

The repository supports:

-   generation with multiple LLMs
-   multiple prompt templates
-   multiple ideological dimensions
-   TF-IDF similarity evaluation
-   semantic embedding similarity evaluation
-   paper-ready summary tables by dimension

------------------------------------------------------------------------

## Repository Structure

``` text
.
├── data/
│   └── Questions.csv
│
├── scripts/
│   ├── run_norag_ideology.py
│   ├── run_rag_ideology.py
│   ├── eval_similarity_ideology.py
│   └── eval_similarity_ideology_embed.py
│
├── results/                 # generated outputs
├── results_eval/            # evaluation outputs
│
├── requirements.txt
├── README.md
└── .env                     # API keys (not committed)
```

------------------------------------------------------------------------

# 1. Environment Setup

## 1.1 Install Python

This project requires **Python 3.11**.

Check your Python version:

``` bash
python --version
```
------------------------------------------------------------------------

## 1.2 Create a virtual environment

### Windows

``` bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### Linux / macOS

``` bash
python3.11 -m venv venv
source venv/bin/activate
```

After activation your terminal should show:

    (venv)

------------------------------------------------------------------------

## 1.3 Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 1.4 Configure API keys

Add your API keys to a `.env` file in the project root:

    OPENAI_API_KEY=your_openai_key
    ANTHROPIC_API_KEY=your_anthropic_key
    GOOGLE_API_KEY=your_google_key

You only need the keys for the models you plan to use.

------------------------------------------------------------------------

# 2. Generation Pipelines

The repository supports two generation settings:

-   **NoRAG**: answers generated from model knowledge only
-   **RAG**: answers generated using retrieved ideological texts

------------------------------------------------------------------------

# 2.1 NoRAG generation

Generate ideology-aligned responses **without retrieval**.

Example:

``` bash
python scripts/run_norag_ideology.py \
  --questions_file data/Questions.csv \
  --model gpt-4o-mini \
  --template prior_knowledge \
  --dimension "Dim. 1 neg"
```

Run all dimensions:

``` bash
python scripts/run_norag_ideology.py \
  --questions_file data/Questions.csv \
  --model gpt-4o-mini \
  --template prior_knowledge \
  --dimension all
```

Run multiple templates:

``` bash
python scripts/run_norag_ideology.py \
  --questions_file data/Questions.csv \
  --model gpt-4o-mini \
  --template prior_knowledge,prior_extensive_knowledge \
  --dimension "Dim. 2 pos"
```

Run everything:

``` bash
python scripts/run_norag_ideology.py \
  --questions_file data/Questions.csv \
  --model all \
  --template all \
  --dimension all
```

------------------------------------------------------------------------

# 2.2 RAG generation

Generate ideology-aligned responses **with retrieval**.

Example:

``` bash
python scripts/run_rag_ideology.py \
  --questions_file data/Questions.csv \
  --data_root data \
  --model gpt-4o-mini \
  --mode standard \
  --template external_knowledge \
  --dimension "Dim. 1 neg"
```

Run all dimensions:

``` bash
python scripts/run_rag_ideology.py \
  --questions_file data/Questions.csv \
  --data_root data \
  --model gpt-4o-mini \
  --mode standard \
  --template external_knowledge \
  --dimension all
```

Run all templates:

``` bash
python scripts/run_rag_ideology.py \
  --questions_file data/Questions.csv \
  --data_root data \
  --model gpt-4o-mini \
  --mode standard \
  --template all \
  --dimension "Dim. 1 neg"
```

Run everything:

``` bash
python scripts/run_rag_ideology.py \
  --questions_file data/Questions.csv \
  --data_root data \
  --model all \
  --mode all \
  --template all \
  --dimension all
```

------------------------------------------------------------------------

# 3. Output Structure

## NoRAG outputs

    results/
    └── output_norag_<model>_<template>/
        ├── results_dim_1_neg.csv
        ├── Responses/
        └── Full_prompt/

## RAG outputs

    results/
    └── output_rag_<mode>_<model>_<template>/
        ├── results_dim_1_neg.csv
        ├── Responses/
        └── Full_prompt/

Each question generates **multiple responses** to capture sampling
variability.

------------------------------------------------------------------------

# 4. Evaluation

Two evaluation methods are provided:

-   **TF‑IDF cosine similarity**
-   **Semantic embedding cosine similarity**

Both compare:

-   the **Example texts** column in `Questions.csv`
-   the generated responses (`response_1` to `response_5`)

------------------------------------------------------------------------

# 4.1 TF‑IDF evaluation

Evaluate NoRAG:

``` bash
python scripts/eval_similarity_ideology.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system norag \
  --model gpt-4o-mini \
  --norag_template prior_knowledge \
  --dimension "Dim. 1 neg"
```

Evaluate RAG:

``` bash
python scripts/eval_similarity_ideology.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system rag \
  --model gpt-4o-mini \
  --mode standard \
  --rag_template external_knowledge \
  --dimension "Dim. 1 neg"
```

Evaluate everything:

``` bash
python scripts/eval_similarity_ideology.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system all \
  --model all \
  --mode all \
  --norag_template all \
  --rag_template all \
  --dimension all
```

TF‑IDF outputs:

    results_eval/tfidf/
    ├── combined_results_summary.csv
    ├── combined_results_detailed.csv
    └── dimension_pivot_table.csv

------------------------------------------------------------------------

# 4.2 Semantic embedding evaluation

Uses **OpenAI embeddings** for semantic similarity.

Example:

``` bash
python scripts/eval_similarity_ideology_embed.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system norag \
  --model gpt-4o-mini \
  --norag_template prior_knowledge \
  --dimension "Dim. 1 neg"
```

Outputs:

    results_eval/embed/
    ├── combined_similarity_summary.csv
    ├── combined_similarity_detailed.csv
    └── dimension_pivot_table.csv

------------------------------------------------------------------------

# 4.3 Merge TF‑IDF and embedding summaries

``` bash
python scripts/eval_similarity_ideology_embed.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system all \
  --merge_tfidf_file results_eval/tfidf/combined_results_summary.csv
```

Output:

    results_eval/embed/
    └── merged_embed_tfidf.csv

------------------------------------------------------------------------

# 5. Paper-style dimension table

Both evaluation scripts generate a pivot table:

  Model   Config   Dim 1   Dim 2   Dim 3   Dim 4   Dim 5   Dim 6
  ------- -------- ------- ------- ------- ------- ------- -------

Configurations:

-   **LLM** = NoRAG with prior knowledge
-   **LLM+** = NoRAG with prior + extensive knowledge
-   **RAG** = retrieved external knowledge only
-   **RAG+** = retrieved external knowledge + prior knowledge
-   **RAG++** = retrieved external knowledge + prior + extensive
    knowledge

Values are averaged across:

-   all questions within the dimension
-   both **negative and positive** variants

------------------------------------------------------------------------

# 6. Recommended Workflow

1️⃣ Generate NoRAG outputs

``` bash
python scripts/run_norag_ideology.py \
  --questions_file data/Questions.csv \
  --model gpt-4o-mini \
  --template all \
  --dimension all
```

2️⃣ Generate RAG outputs

``` bash
python scripts/run_rag_ideology.py \
  --questions_file data/Questions.csv \
  --data_root data \
  --model gpt-4o-mini \
  --mode all \
  --template all \
  --dimension all
```

3️⃣ Run TF‑IDF evaluation

``` bash
python scripts/eval_similarity_ideology.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system all \
  --model gpt-4o-mini \
  --mode all \
  --norag_template all \
  --rag_template all \
  --dimension all
```

4️⃣ Run embedding evaluation

``` bash
python scripts/eval_similarity_ideology_embed.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system all \
  --model gpt-4o-mini \
  --mode all \
  --norag_template all \
  --rag_template all \
  --dimension all
```

5️⃣ Merge both summaries

``` bash
python scripts/eval_similarity_ideology_embed.py \
  --questions_file data/Questions.csv \
  --results_root results \
  --system all \
  --merge_tfidf_file results_eval/tfidf/combined_results_summary.csv
```

------------------------------------------------------------------------

# 7. Supported Models

-   GPT‑4o‑mini
-   GPT‑3.5‑turbo
-   Gemini
-   Claude

------------------------------------------------------------------------

# 8. Notes

-   `results/` is created automatically during generation.
-   `results_eval/` is created automatically during evaluation.
-   OpenAI embeddings are required for semantic similarity evaluation.
-   API rate limits are handled with retry logic.
-   Multiple responses per prompt capture output variability.

------------------------------------------------------------------------

# 9. License

Apache License 2.0

------------------------------------------------------------------------

# 10. Citation

If you use this repository in your work, please cite:

@inproceedings{Amamou2026IdeologicalAlignment,
  title={Investigating LLMs' Alignment with Ideological Discourses in Retrieval-Augmented Generation},
  author={Amamou, Hazem and Salari, Elmira and Chen, Ru Bing and Delfino, Maria Claudia Nunes and Yamada, Rogerio and Berber-Sardinha, Tony and Kshirsagar, Shruti and Gagnon, Samuel and Davoust, Alan and Avila, Anderson R.},
  booktitle={Conference on Computational Natural Language Learning (CoNLL 2026)},
  year={2026}
}
