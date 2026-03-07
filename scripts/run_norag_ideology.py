from ast import arg
import os
import re
import time
import argparse
from itertools import product
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

DEFAULT_SLEEP_TIME = 5
DEFAULT_NUM_RESPONSES = 5

SUPPORTED_MODELS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o-mini": "gpt-4o-mini",
    "google": "gemini-2.5-flash",
    "anthropic": "claude-3-haiku-20240307",
}

SUPPORTED_DIMENSIONS = [
    "Dim. 1 neg", "Dim. 1 pos",
    "Dim. 2 neg", "Dim. 2 pos",
    "Dim. 3 neg", "Dim. 3 pos",
    "Dim. 4 neg", "Dim. 4 pos",
    "Dim. 5 neg", "Dim. 5 pos",
    "Dim. 6 neg", "Dim. 6 pos",
]

TEMPLATES = {
    "prior_knowledge": """
You are a reliable AI assistant that can answer users' questions based on a particular ideology, which we call
'dimension'. This ideology will be explained to you in three ways:
(1) a dimension label,
(2) a dimension description,
(3) the lexical items that are loaded on the dimension.

Use your prior knowledge acquired during training to provide the most accurate and detailed answer to the question.
Your answer should reflect the dimension label, dimension description, and typical lexical items.
Do not use bullet points in the answer.

Question: {question}
Dimension label: {dimension_label}
Dimension description: {dimension_description}
Typical vocabulary: {typical_vocabulary}
Answer:
""",
    "prior_extensive_knowledge": """
You are a reliable AI assistant that can answer users' questions based on a particular ideology, which we call
'dimension'. This ideology will be explained to you in three ways:
(1) a dimension label,
(2) a dimension description,
(3) the lexical items that are loaded on the dimension.

Use your prior and extensive knowledge to answer the question.
Your answer should reflect the dimension label, dimension description, and typical lexical items.
Do not use bullet points in the answer.

Question: {question}
Dimension label: {dimension_label}
Dimension description: {dimension_description}
Typical vocabulary: {typical_vocabulary}
Answer:
""",
}


def sanitize_filename(text: str) -> str:
    text = text.lower().strip().replace(" ", "_")
    return re.sub(r"[^\w\-]+", "_", text)


def normalize_dimension_name(dimension: str) -> str:
    return sanitize_filename(dimension.replace(".", ""))


class ModelHandler:
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 4096):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_name == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
            return ChatAnthropic(
                api_key=api_key,
                model=SUPPORTED_MODELS["anthropic"],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        if self.model_name == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables.")
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=SUPPORTED_MODELS["google"],
                temperature=self.temperature,
            )

        if self.model_name in {"gpt-3.5-turbo", "gpt-4o-mini"}:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables.")
            return ChatOpenAI(
                api_key=api_key,
                model=SUPPORTED_MODELS[self.model_name],
                temperature=self.temperature,
            )

        raise ValueError(
            f"Unsupported model: {self.model_name}. "
            f"Choose from: {list(SUPPORTED_MODELS.keys())}"
        )


def build_prompt(template_name: str) -> ChatPromptTemplate:
    if template_name not in TEMPLATES:
        raise ValueError(f"Invalid template: {template_name}. Choose from: {list(TEMPLATES.keys())}")
    return ChatPromptTemplate.from_template(TEMPLATES[template_name])


def get_response(
    question: str,
    dimension_label: str,
    dimension_description: str,
    typical_vocabulary: str,
    model,
    prompt: ChatPromptTemplate,
) -> str:
    try:
        final_prompt = prompt.format(
            question=question,
            dimension_label=dimension_label,
            dimension_description=dimension_description,
            typical_vocabulary=typical_vocabulary,
        )
        response = model.invoke(final_prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"ERROR: {str(e)}"


def generate_responses_for_dimension(
    df_dimension: pd.DataFrame,
    model,
    prompt: ChatPromptTemplate,
    num_responses: int,
    sleep_time: int,
    limit: int | None = None,
) -> pd.DataFrame:
    if limit is not None:
        df_dimension=df_dimension.head(limit)
    if df_dimension.empty:
        print("[WARN] No rows found for this dimension.")
        return pd.DataFrame()
    rows = []                                  #total=len(df_dimension)

    for _, row in tqdm(df_dimension.iterrows(), total=len(df_dimension), desc="Generating responses"):
        question = row["Questions"]
        dimension_label = row["Dimension Label"]
        dimension_description = row["Dimensions Descriptions"]
        typical_vocabulary = row["Typical vocabulary"]
        topic = row["Topics"]

        record = {
            "Topics": topic,
            "Questions": question,
            "Dimension Label": dimension_label,
            "Dimensions Descriptions": dimension_description,
            "Typical vocabulary": typical_vocabulary,
            "dimensions": row["dimensions"],
        }

        for i in range(1, num_responses + 1):
            response_text = get_response(
                question=question,
                dimension_label=dimension_label,
                dimension_description=dimension_description,
                typical_vocabulary=typical_vocabulary,
                model=model,
                prompt=prompt,
            )

            full_prompt = prompt.format(
                question=question,
                dimension_label=dimension_label,
                dimension_description=dimension_description,
                typical_vocabulary=typical_vocabulary,
            )

            record[f"response_{i}"] = response_text
            record[f"full_prompt_response_{i}"] = (
                f"{full_prompt}\n\n### MODEL RESPONSE ###\n{response_text}"
            )

            if i < num_responses:
                time.sleep(sleep_time)

        rows.append(record)

    return pd.DataFrame(rows)


def save_text_outputs(
    df_results: pd.DataFrame,
    output_dir: str,
    dimension_slug: str,
    model_name: str,
    template_name: str,
    num_responses: int,
) -> None:
    response_dir = os.path.join(output_dir, "Responses", dimension_slug)
    full_prompt_dir = os.path.join(output_dir, "Full_prompt", dimension_slug)

    os.makedirs(response_dir, exist_ok=True)
    os.makedirs(full_prompt_dir, exist_ok=True)

    if template_name == "template1":
        template_tag = "only_prior_knowledge"
    elif template_name == "template2":
        template_tag = "prior_extensive_knowledge"
    else:
        template_tag = template_name

    for idx, row in df_results.iterrows():
        topic = row["Topics"]
        topic_slug = sanitize_filename(topic)
        question_id = str(idx + 1).zfill(2)

        for i in range(1, num_responses + 1):
            response_text = str(row.get(f"response_{i}", "")).strip()
            full_prompt_text = str(row.get(f"full_prompt_response_{i}", "")).strip()

            response_filename = (
                f"response_{dimension_slug}_{template_tag}_{topic_slug}_"
                f"norag_{model_name}_question{question_id}_{i}.txt"
            )
            full_prompt_filename = (
                f"full_prompt_response_{dimension_slug}_{template_tag}_{topic_slug}_"
                f"norag_{model_name}_question{question_id}_{i}.txt"
            )

            with open(os.path.join(response_dir, response_filename), "w", encoding="utf-8") as f:
                f.write(response_text)

            with open(os.path.join(full_prompt_dir, full_prompt_filename), "w", encoding="utf-8") as f:
                f.write(full_prompt_text)


def run_single_configuration(
    questions_file: str,
    model_name: str,
    template_name: str,
    dimension: str,
    sleep_time: int,
    num_responses: int,
    output_root: str,
    temperature: float,
) -> None:
    print(f"\nRunning model={model_name} | template={template_name} | dimension={dimension}")

    model = ModelHandler(model_name=model_name, temperature=temperature).model
    prompt = build_prompt(template_name)

    df_questions = pd.read_csv(questions_file)

    required_columns = {
        "Topics",
        "Questions",
        "Dimension Label",
        "Dimensions Descriptions",
        "Typical vocabulary",
        "dimensions",
    }
    missing = required_columns - set(df_questions.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df_dimension = df_questions[df_questions["dimensions"] == dimension].copy()

    dimension_slug = normalize_dimension_name(dimension)
    output_dir = os.path.join(output_root, f"output_norag_{model_name}_{template_name}")

    df_results = generate_responses_for_dimension(
        df_dimension=df_dimension,
        model=model,
        prompt=prompt,
        num_responses=num_responses,
        sleep_time=sleep_time,
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"results_{dimension_slug}.csv")
    df_results.to_csv(csv_path, index=False)

    save_text_outputs(
        df_results=df_results,
        output_dir=output_dir,
        dimension_slug=dimension_slug,
        model_name=model_name,
        template_name=template_name,
        num_responses=num_responses,
    )

    print(f"[OK] Saved CSV: {csv_path}")
    print(f"[OK] Saved text outputs under: {output_dir}")


def parse_list_argument(value: str, allowed_values: List[str], arg_name: str) -> List[str]:
    value = value.strip()

    if value.lower() == "all":
        return allowed_values

    items = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in items if item not in allowed_values]
    if invalid:
        raise ValueError(
            f"Invalid {arg_name}: {invalid}. Allowed values are: {allowed_values} or 'all'."
        )
    return items


def main():
    parser = argparse.ArgumentParser(
        description="Generate no-RAG ideology-aligned answers for selected models, templates, and dimensions."
    )
    parser.add_argument("--questions_file", type=str, default="data/Questions.csv", help="Path to the CSV file.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name or comma-separated list, or 'all'.")
    parser.add_argument("--template", type=str, default="template1", help="Template name or comma-separated list, or 'all'.")
    parser.add_argument("--dimension", type=str, default="all", help="Dimension name or comma-separated list, or 'all'.")
    parser.add_argument("--sleep_time", type=int, default=DEFAULT_SLEEP_TIME, help="Sleep time between API calls.")
    parser.add_argument("--num_responses", type=int, default=DEFAULT_NUM_RESPONSES, help="Number of responses per question.")
    parser.add_argument("--output_root", type=str, default="results", help="Root output directory.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions per dimension (for testing).")
    args = parser.parse_args()

    selected_models = parse_list_argument(args.model, list(SUPPORTED_MODELS.keys()), "model")
    selected_templates = parse_list_argument(args.template, list(TEMPLATES.keys()), "template")
    selected_dimensions = parse_list_argument(args.dimension, SUPPORTED_DIMENSIONS, "dimension")

    combinations = list(product(selected_models, selected_templates, selected_dimensions))
    total = len(combinations)

    print(f"Total runs: {total}")

    for idx, (model_name, template_name, dimension) in enumerate(combinations, start=1):
        print("\n" + "=" * 70)
        print(f"Run {idx}/{total}")
        print(f"Model     : {model_name}")
        print(f"Template  : {template_name}")
        print(f"Dimension : {dimension}")
        print("=" * 70)

        try:
            run_single_configuration(
                questions_file=args.questions_file,
                model_name=model_name,
                template_name=template_name,
                dimension=dimension,
                sleep_time=args.sleep_time,
                num_responses=args.num_responses,
                output_root=args.output_root,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"[ERROR] Failed for model={model_name}, template={template_name}, dimension={dimension}")
            print(str(e))


if __name__ == "__main__":
    main()