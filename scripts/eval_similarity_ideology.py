import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SUPPORTED_DIMENSIONS = [
    "Dim. 1 neg", "Dim. 1 pos",
    "Dim. 2 neg", "Dim. 2 pos",
    "Dim. 3 neg", "Dim. 3 pos",
    "Dim. 4 neg", "Dim. 4 pos",
    "Dim. 5 neg", "Dim. 5 pos",
    "Dim. 6 neg", "Dim. 6 pos",
]

DIMENSION_MAP = {
    "Dim. 1 neg": "dim_1_neg",
    "Dim. 1 pos": "dim_1_pos",
    "Dim. 2 neg": "dim_2_neg",
    "Dim. 2 pos": "dim_2_pos",
    "Dim. 3 neg": "dim_3_neg",
    "Dim. 3 pos": "dim_3_pos",
    "Dim. 4 neg": "dim_4_neg",
    "Dim. 4 pos": "dim_4_pos",
    "Dim. 5 neg": "dim_5_neg",
    "Dim. 5 pos": "dim_5_pos",
    "Dim. 6 neg": "dim_6_neg",
    "Dim. 6 pos": "dim_6_pos",
}

NORAG_TEMPLATE_MAP = {
    "template1": "only_prior_knowledge",
    "template2": "prior_extensive_knowledge",
    "prior_knowledge": "only_prior_knowledge",
    "prior_extensive_knowledge": "prior_extensive_knowledge",
}

RAG_TEMPLATE_MAP = {
    "template1": "only_external_knowledge",
    "template2": "prior_external",
    "template3": "prior_extensive_external",
    "external_knowledge": "only_external_knowledge",
    "prior_external": "prior_external",
    "prior_extensive_external": "prior_extensive_external",
}

SUPPORTED_RAG_MODES = ["standard", "opposed", "both"]


def parse_list_argument(value: str, allowed_values: List[str], arg_name: str) -> List[str]:
    value = value.strip()
    if value.lower() == "all":
        return allowed_values
    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in allowed_values]
    if invalid:
        raise ValueError(
            f"Invalid {arg_name}: {invalid}. Allowed values: {allowed_values} or 'all'."
        )
    return items


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def compute_tfidf_similarities(example_text: str, responses: List[str]) -> List[float]:
    if not example_text or not any(responses):
        return [0.0] * len(responses)

    try:
        texts = [example_text] + responses
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
        return [float(x) for x in sims]
    except ValueError:
        return [0.0] * len(responses)


def summarize_scores(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {
            "avg_correlation": 0.0,
            "median_correlation": 0.0,
            "std_correlation": 0.0,
            "count_responses": 0,
        }
    return {
        "avg_correlation": round(float(np.mean(scores)), 4),
        "median_correlation": round(float(np.median(scores)), 4),
        "std_correlation": round(float(np.std(scores)), 4),
        "count_responses": len(scores),
    }


def required_question_columns() -> List[str]:
    return ["Topics", "Questions", "Example texts", "dimensions"]


def load_questions(questions_file: str) -> pd.DataFrame:
    df = pd.read_csv(questions_file)
    missing = set(required_question_columns()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in questions file: {sorted(missing)}")
    return df


def build_norag_results_path(results_root: str, model: str, template: str, dimension: str) -> Path:
    dim_slug = DIMENSION_MAP[dimension]
    return Path(results_root) / f"output_norag_{model}_{template}" / f"results_{dim_slug}.csv"


def build_rag_results_path(results_root: str, model: str, mode: str, template: str, dimension: str) -> Path:
    dim_slug = DIMENSION_MAP[dimension]
    return Path(results_root) / f"output_rag_{mode}_{model}_{template}" / f"results_{dim_slug}.csv"


def evaluate_results_file(
    questions_df: pd.DataFrame,
    results_path: Path,
    model: str,
    system_type: str,
    knowledge_template: str,
    dimension_type: str,
    dimension: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not results_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    results_df = pd.read_csv(results_path)

    required_response_cols = [f"response_{i}" for i in range(1, 6)]
    missing = set(required_response_cols) - set(results_df.columns)
    if missing:
        raise ValueError(f"Missing response columns in {results_path}: {sorted(missing)}")

    # Keep only the metadata we actually need from questions_df
    q_df = (
        questions_df[questions_df["dimensions"] == dimension][
            ["Topics", "Questions", "Example texts", "dimensions"]
        ]
        .copy()
        .reset_index(drop=True)
    )

    # Keep only response columns from results_df to avoid duplicate column names
    r_df = results_df[required_response_cols].copy().reset_index(drop=True)

    # Align lengths safely
    n = min(len(q_df), len(r_df))
    if n == 0:
        return pd.DataFrame(), pd.DataFrame()

    q_df = q_df.iloc[:n].reset_index(drop=True)
    r_df = r_df.iloc[:n].reset_index(drop=True)

    merged = pd.concat([q_df, r_df], axis=1)

    detailed_rows = []
    summary_rows = []

    for _, row in merged.iterrows():
        topic = safe_text(row["Topics"])
        question = safe_text(row["Questions"])
        example_text = safe_text(row["Example texts"])

        responses = [safe_text(row[f"response_{i}"]) for i in range(1, 6)]
        sims = compute_tfidf_similarities(example_text, responses)
        stats = summarize_scores(sims)

        for idx, sim in enumerate(sims, start=1):
            detailed_rows.append({
                "dimensions": dimension,
                "topic": topic,
                "question": question,
                "model": model,
                "system_type": system_type,
                "dimension_type": dimension_type,
                "knowledge_template": knowledge_template,
                "response_num": idx,
                "correlation": round(sim, 4),
            })

        summary_rows.append({
            "dimensions": dimension,
            "topic": topic,
            "question": question,
            "model": model,
            "system_type": system_type,
            "dimension_type": dimension_type,
            "knowledge_template": knowledge_template,
            **stats,
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(detailed_rows)


def evaluate_norag(
    questions_df: pd.DataFrame,
    results_root: str,
    models: List[str],
    templates: List[str],
    dimensions: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_summary = []
    all_detailed = []

    for model in models:
        for template in templates:
            template_tag = NORAG_TEMPLATE_MAP.get(template, template)
            for dimension in dimensions:
                results_path = build_norag_results_path(results_root, model, template, dimension)
                summary_df, detailed_df = evaluate_results_file(
                    questions_df=questions_df,
                    results_path=results_path,
                    model=model,
                    system_type="NoRAG",
                    knowledge_template=template_tag,
                    dimension_type="NoRAG",
                    dimension=dimension,
                )
                if not summary_df.empty:
                    all_summary.append(summary_df)
                if not detailed_df.empty:
                    all_detailed.append(detailed_df)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    detailed = pd.concat(all_detailed, ignore_index=True) if all_detailed else pd.DataFrame()
    return summary, detailed


def evaluate_rag(
    questions_df: pd.DataFrame,
    results_root: str,
    models: List[str],
    modes: List[str],
    templates: List[str],
    dimensions: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_summary = []
    all_detailed = []

    for model in models:
        for mode in modes:
            for template in templates:
                template_tag = RAG_TEMPLATE_MAP.get(template, template)
                for dimension in dimensions:
                    results_path = build_rag_results_path(results_root, model, mode, template, dimension)
                    summary_df, detailed_df = evaluate_results_file(
                        questions_df=questions_df,
                        results_path=results_path,
                        model=model,
                        system_type="RAG",
                        knowledge_template=template_tag,
                        dimension_type=mode,
                        dimension=dimension,
                    )
                    if not summary_df.empty:
                        all_summary.append(summary_df)
                    if not detailed_df.empty:
                        all_detailed.append(detailed_df)

    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    detailed = pd.concat(all_detailed, ignore_index=True) if all_detailed else pd.DataFrame()
    return summary, detailed


def merge_embed_and_tfidf(embed_file: str, tfidf_file: str, out_file: str) -> None:
    embed_df = pd.read_csv(embed_file)
    tfidf_df = pd.read_csv(tfidf_file)

    if "dimension" in embed_df.columns and "dimensions" not in embed_df.columns:
        embed_df = embed_df.rename(columns={"dimension": "dimensions"})

    merged_df = pd.merge(
        embed_df,
        tfidf_df,
        on=["dimensions", "model", "system_type", "knowledge_template", "dimension_type"],
        suffixes=("_embed", "_tfidf"),
    )

    if "avg_correlation_embed" in merged_df.columns:
        merged_df = merged_df.rename(columns={"avg_correlation_embed": "correlation_embed"})
    if "avg_correlation_tfidf" in merged_df.columns:
        merged_df = merged_df.rename(columns={"avg_correlation_tfidf": "correlation_tfidf"})

    columns_to_drop = ["median_correlation", "std_correlation", "count_responses", "response_count"]
    merged_df = merged_df.drop(columns=columns_to_drop, errors="ignore")

    if "correlation_embed" in merged_df.columns:
        merged_df["correlation_embed"] = merged_df["correlation_embed"].round(2)
    if "correlation_tfidf" in merged_df.columns:
        merged_df["correlation_tfidf"] = merged_df["correlation_tfidf"].round(2)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_file, index=False)
    print(f"[OK] Saved merged file: {out_file}")

def normalize_model_name(model: str) -> str:
    model_map = {
        "gpt-4o-mini": "GPT-4omini",
        "gpt-3.5-turbo": "GPT-3.5",
        "google": "Gemini",
        "anthropic": "Claude",
    }
    return model_map.get(model, model)


def normalize_config_label(system_type: str, knowledge_template: str) -> str:
    """
    Convert internal config names into paper-friendly labels.
    """
    if system_type == "NoRAG":
        mapping = {
            "only_prior_knowledge": "LLM",
            "prior_extensive_knowledge": "LLM+",
        }
        return mapping.get(knowledge_template, knowledge_template)

    if system_type == "RAG":
        mapping = {
            "only_external_knowledge": "RAG",
            "prior_external": "RAG+",
            "prior_extensive_external": "RAG++",
        }
        return mapping.get(knowledge_template, knowledge_template)

    return knowledge_template


def extract_dimension_group(dimensions_value: str) -> str:
    """
    Convert:
      'Dim. 1 neg' -> 'Dim 1'
      'Dim. 1 pos' -> 'Dim 1'
    """
    match = re.match(r"Dim\.\s*(\d+)\s+(neg|pos)", str(dimensions_value).strip(), re.IGNORECASE)
    if match:
        return f"Dim {match.group(1)}"
    return str(dimensions_value)


def build_dimension_pivot_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build paper-style table:
    Model | Config | Dim 1 | Dim 2 | ... | Dim 6

    It averages over:
    - all rows/questions
    - pos/neg within the same dimension number
    """
    if summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy()

    df["Model"] = df["model"].apply(normalize_model_name)
    df["Config"] = df.apply(
        lambda row: normalize_config_label(row["system_type"], row["knowledge_template"]),
        axis=1,
    )
    df["DimensionGroup"] = df["dimensions"].apply(extract_dimension_group)

    # Average over all rows inside the same model/config/dimension group
    grouped = (
        df.groupby(["Model", "Config", "DimensionGroup"], as_index=False)["avg_correlation"]
        .mean()
    )

    pivot_df = grouped.pivot_table(
        index=["Model", "Config"],
        columns="DimensionGroup",
        values="avg_correlation",
        aggfunc="mean"
    ).reset_index()

    desired_dim_order = [f"Dim {i}" for i in range(1, 7)]
    existing_dim_cols = [c for c in desired_dim_order if c in pivot_df.columns]

    pivot_df = pivot_df[["Model", "Config"] + existing_dim_cols]

    for col in existing_dim_cols:
        pivot_df[col] = pivot_df[col].round(2)

    # Optional order for configs in the final table
    config_order = {"LLM": 0, "LLM+": 1, "RAG": 2, "RAG+": 3, "RAG++": 4}
    model_order = {"GPT-4omini": 0, "GPT-3.5": 1, "Gemini": 2, "Claude": 3}

    pivot_df["_model_order"] = pivot_df["Model"].map(model_order).fillna(999)
    pivot_df["_config_order"] = pivot_df["Config"].map(config_order).fillna(999)

    pivot_df = pivot_df.sort_values(["_model_order", "_config_order"]).drop(
        columns=["_model_order", "_config_order"]
    )

    return pivot_df
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate cosine similarity between Example texts and ideology-generation outputs."
    )
    parser.add_argument("--questions_file", type=str, default="data/Questions.csv")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--system", type=str, default="all", help="norag, rag, or all")
    parser.add_argument("--model", type=str, default="all", help="Model name(s), comma-separated, or all")
    parser.add_argument("--norag_template", type=str, default="all", help="NoRAG template(s), comma-separated, or all")
    parser.add_argument("--rag_template", type=str, default="all", help="RAG template(s), comma-separated, or all")
    parser.add_argument("--mode", type=str, default="all", help="RAG mode(s): standard, opposed, both, or all")
    parser.add_argument("--dimension", type=str, default="all", help="Dimension(s), comma-separated, or all")
    parser.add_argument("--out_dir", type=str, default="results_eval/tfidf")
    parser.add_argument("--merge_embed_file", type=str, default=None, help="Optional embedding summary CSV to merge with TF-IDF results")
    args = parser.parse_args()

    questions_df = load_questions(args.questions_file)

    model_candidates = sorted(
        list(
            {
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "google",
                "anthropic",
            }
        )
    )
    norag_template_candidates = sorted(list(NORAG_TEMPLATE_MAP.keys()))
    rag_template_candidates = sorted(list(RAG_TEMPLATE_MAP.keys()))

    selected_models = parse_list_argument(args.model, model_candidates, "model")
    selected_dimensions = parse_list_argument(args.dimension, SUPPORTED_DIMENSIONS, "dimension")
    selected_modes = parse_list_argument(args.mode, SUPPORTED_RAG_MODES, "mode")
    selected_norag_templates = parse_list_argument(args.norag_template, norag_template_candidates, "norag_template")
    selected_rag_templates = parse_list_argument(args.rag_template, rag_template_candidates, "rag_template")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = args.system.lower().strip()
    if system not in {"norag", "rag", "all"}:
        raise ValueError("system must be one of: norag, rag, all")

    final_summary_parts = []
    final_detailed_parts = []

    if system in {"norag", "all"}:
        print("[INFO] Evaluating NoRAG outputs...")
        norag_summary, norag_detailed = evaluate_norag(
            questions_df=questions_df,
            results_root=args.results_root,
            models=selected_models,
            templates=selected_norag_templates,
            dimensions=selected_dimensions,
        )
        if not norag_summary.empty:
            final_summary_parts.append(norag_summary)
        if not norag_detailed.empty:
            final_detailed_parts.append(norag_detailed)

    if system in {"rag", "all"}:
        print("[INFO] Evaluating RAG outputs...")
        rag_summary, rag_detailed = evaluate_rag(
            questions_df=questions_df,
            results_root=args.results_root,
            models=selected_models,
            modes=selected_modes,
            templates=selected_rag_templates,
            dimensions=selected_dimensions,
        )
        if not rag_summary.empty:
            final_summary_parts.append(rag_summary)
        if not rag_detailed.empty:
            final_detailed_parts.append(rag_detailed)

    final_summary = pd.concat(final_summary_parts, ignore_index=True) if final_summary_parts else pd.DataFrame()
    final_detailed = pd.concat(final_detailed_parts, ignore_index=True) if final_detailed_parts else pd.DataFrame()

    summary_file = out_dir / "combined_results_summary.csv"
    detailed_file = out_dir / "combined_results_detailed.csv"
    pivot_file = out_dir / "dimension_pivot_table.csv"

    final_summary.to_csv(summary_file, index=False)
    final_detailed.to_csv(detailed_file, index=False)

    pivot_df = build_dimension_pivot_table(final_summary)
    pivot_df.to_csv(pivot_file, index=False)

    print(f"[OK] Saved summary: {summary_file}")
    print(f"[OK] Saved detailed: {detailed_file}")
    print(f"[OK] Saved dimension pivot table: {pivot_file}")
    print(f"[INFO] Summary rows: {len(final_summary)}")
    print(f"[INFO] Detailed rows: {len(final_detailed)}")

    if args.merge_embed_file:
        merged_out = out_dir / "merged_embed_tfidf.csv"
        merge_embed_and_tfidf(
            embed_file=args.merge_embed_file,
            tfidf_file=str(summary_file),
            out_file=str(merged_out),
        )


if __name__ == "__main__":
    main()