import os
import re
import time
import argparse
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_SLEEP_TIME = 5
DEFAULT_NUM_RESPONSES = 5
DEFAULT_K = 4

SUPPORTED_MODELS = {
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-4o-mini": "gpt-4o-mini",
    "google": "gemini-2.5-flash",
    "anthropic": "claude-3-haiku-20240307",
}

SUPPORTED_MODES = ["standard", "opposed", "both"]

DIMENSION_MAP = {
    "Dim. 1 neg": "f1_negative",
    "Dim. 1 pos": "f1_positive",
    "Dim. 2 neg": "f2_negative",
    "Dim. 2 pos": "f2_positive",
    "Dim. 3 neg": "f3_negative",
    "Dim. 3 pos": "f3_positive",
    "Dim. 4 neg": "f4_negative",
    "Dim. 4 pos": "f4_positive",
    "Dim. 5 neg": "f5_negative",
    "Dim. 5 pos": "f5_positive",
    "Dim. 6 neg": "f6_negative",
    "Dim. 6 pos": "f6_positive",
}

SUPPORTED_DIMENSIONS = list(DIMENSION_MAP.keys())

TEMPLATES = {
    "external_knowledge": """
You are a reliable AI assistant that can answer users' questions based on a particular ideology, which we call
'dimension'. This ideology will be explained to you in four ways:
(1) a dimension label,
(2) a dimension description,
(3) the lexical items that are loaded on the dimension,
(4) texts from Endorsed and Controversial treatments that illustrate this dimension.

Do not use your prior or extensive knowledge. Use only the external knowledge present in the retrieved context
to provide the most accurate and detailed answer to the question below.

Your answer should reflect the dimension label, dimension description, typical lexical items, and example texts.
Do not use bullet points in the answer.

Question: {question}
Dimension label: {dimension_label}
Dimension description: {dimension_description}
Typical vocabulary: {typical_vocabulary}
Context: {context}

Answer:
""",
    "external_prior_knowledge": """
You are a reliable AI assistant that can answer users' questions based on a particular ideology, which we call
'dimension'. This ideology will be explained to you in four ways:
(1) a dimension label,
(2) a dimension description,
(3) the lexical items loaded on the dimension with their factorial weights in descending order,
(4) texts from Endorsed and Controversial treatments that illustrate this dimension.

Combine the external knowledge present in the retrieved context and your prior knowledge acquired during training
to provide the most accurate and detailed answer to the question below.

Your answer should reflect the dimension label, description, typical lexical items, and example texts.
Do not use bullet points in the answer.

Question: {question}
Dimension label: {dimension_label}
Dimension description: {dimension_description}
Typical vocabulary: {typical_vocabulary}
Context: {context}

Answer:
""",
    "external_prior_extensive_knowledge": """
You are a reliable AI assistant that can answer users' questions based on a particular ideology, which we call
'dimension'. This ideology will be explained to you in four ways:
(1) a dimension label,
(2) a dimension description,
(3) the lexical items loaded on the dimension with their factorial weights in descending order,
(4) texts from Endorsed and Controversial treatments that illustrate this dimension.

Combine the external knowledge present in the retrieved context, your prior knowledge acquired during training,
and your extensive knowledge to provide the most accurate and detailed answer to the question below.

Your answer should reflect the dimension label, description, typical lexical items, and example texts.
Do not use bullet points in the answer.

Question: {question}
Dimension label: {dimension_label}
Dimension description: {dimension_description}
Typical vocabulary: {typical_vocabulary}
Context: {context}

Answer:
""",
}


def sanitize_filename(text: str) -> str:
    text = text.lower().strip().replace(" ", "_")
    return re.sub(r"[^\w\-]+", "_", text)


def normalize_dimension_name(dimension: str) -> str:
    return sanitize_filename(dimension.replace(".", ""))


class DimensionConfig:
    def __init__(self, data_dirs: List[str], desc_field: str, filter_field: str):
        self.data_dirs = data_dirs
        self.desc_field = desc_field
        self.filter_field = filter_field


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


def process_text_file(file_path: str) -> Dict[str, str]:
    encodings = ["utf-8", "latin1"]
    lines = []

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.read().splitlines()
            break
        except UnicodeDecodeError:
            continue

    data = {"Content": ""}
    for line in lines:
        if " = " in line:
            key, value = line.split(" = ", 1)
            data[key.strip()] = value.strip()
        else:
            data["Content"] += f"{line} "

    data["Content"] = data["Content"].strip()
    return data


def load_documents_from_directories(directory_paths: List[str]) -> List[Document]:
    all_data = []

    for directory_path in directory_paths:
        if not os.path.isdir(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                all_data.append(process_text_file(file_path))

    if not all_data:
        raise ValueError(f"No .txt documents found in: {directory_paths}")

    df = pd.DataFrame(all_data)
    loader = DataFrameLoader(df, page_content_column="Content")
    return loader.load()


def create_vectorstore(documents: List[Document], chunk_size: int, chunk_overlap: int) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in split_docs]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_texts(texts=texts, embedding=embeddings)


def get_rag_response(
    question: str,
    dimension_label: str,
    dimension_description: str,
    typical_vocabulary: str,
    retriever,
    model,
    prompt: ChatPromptTemplate,
) -> Dict[str, str]:
    try:
        retrieved_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)

        final_prompt = prompt.format(
            question=question,
            dimension_label=dimension_label,
            dimension_description=dimension_description,
            typical_vocabulary=typical_vocabulary,
            context=context,
        )

        response = model.invoke(final_prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        return {"response": response_text, "context": context}
    except Exception as e:
        print(f"Error in get_rag_response: {e}")
        return {"response": f"ERROR: {str(e)}", "context": "ERROR"}


def generate_responses_for_dimension(
    df_dimension: pd.DataFrame,
    retriever,
    model,
    prompt: ChatPromptTemplate,
    desc_field: str,
    num_responses: int,
    sleep_time: int,
    limit: int | None = None,
) -> pd.DataFrame:
    if limit is not None:
        df_dimension=df_dimension.head(limit)
    if df_dimension.empty:
        print("[WARN] No rows found for this dimension.")
        return pd.DataFrame()
    rows = []

    for _, row in tqdm(df_dimension.iterrows(), total=len(df_dimension), desc="Generating RAG responses"):
        question = row["Questions"]
        dimension_label = row["Dimension Label"]
        dimension_description = row[desc_field]
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

        cached_context = None

        for i in range(1, num_responses + 1):
            result = get_rag_response(
                question=question,
                dimension_label=dimension_label,
                dimension_description=dimension_description,
                typical_vocabulary=typical_vocabulary,
                retriever=retriever,
                model=model,
                prompt=prompt,
            )

            response_text = result["response"]
            cached_context = cached_context or result["context"]

            full_prompt = prompt.format(
                question=question,
                dimension_label=dimension_label,
                dimension_description=dimension_description,
                typical_vocabulary=typical_vocabulary,
                context=cached_context,
            )

            record[f"response_{i}"] = response_text
            record[f"full_prompt_response_{i}"] = (
                f"{full_prompt}\n\n### MODEL RESPONSE ###\n{response_text}"
            )

            if i < num_responses:
                time.sleep(sleep_time)

        record["context"] = cached_context
        rows.append(record)

    return pd.DataFrame(rows)


from pathlib import Path

def save_text_outputs(
    df_results: pd.DataFrame,
    output_dir: str,
    dimension_slug: str,
    model_name: str,
    template_name: str,
    num_responses: int,
) -> None:
    response_dir = Path(output_dir) / "Responses" / dimension_slug
    full_prompt_dir = Path(output_dir) / "Full_prompt" / dimension_slug

    response_dir.mkdir(parents=True, exist_ok=True)
    full_prompt_dir.mkdir(parents=True, exist_ok=True)

    template_map = {
        "template1": "only_external_knowledge",
        "template2": "prior_external",
        "template3": "prior_extensive_external",
        "external_knowledge": "only_external_knowledge",
        "prior_external": "prior_external",
        "prior_extensive_external": "prior_extensive_external",
    }
    template_tag = sanitize_filename(template_map.get(template_name, template_name))
    model_slug = sanitize_filename(model_name)
    dimension_slug = sanitize_filename(dimension_slug)

    for idx, row in df_results.iterrows():
        topic_slug = sanitize_filename(str(row["Topics"]))[:60]
        question_id = str(idx + 1).zfill(2)

        for i in range(1, num_responses + 1):
            response_text = str(row.get(f"response_{i}", "")).strip()
            full_prompt_text = str(row.get(f"full_prompt_response_{i}", "")).strip()

            response_filename = (
                f"response_{dimension_slug}_{template_tag}_{topic_slug}_"
                f"rag_{model_slug}_q{question_id}_{i}.txt"
            )
            full_prompt_filename = (
                f"full_prompt_response_{dimension_slug}_{template_tag}_{topic_slug}_"
                f"rag_{model_slug}_q{question_id}_{i}.txt"
            )

            response_path = response_dir / response_filename
            full_prompt_path = full_prompt_dir / full_prompt_filename

            # Extra safety: always ensure parent exists before writing
            response_path.parent.mkdir(parents=True, exist_ok=True)
            full_prompt_path.parent.mkdir(parents=True, exist_ok=True)

            with open(response_path, "w", encoding="utf-8") as f:
                f.write(response_text)

            with open(full_prompt_path, "w", encoding="utf-8") as f:
                f.write(full_prompt_text)


def get_mode_config(mode: str) -> DimensionConfig:
    if mode == "standard":
        return DimensionConfig(
            data_dirs=["texts_RAG_dimensions"],
            desc_field="Dimensions Descriptions",
            filter_field="dimensions",
        )
    if mode == "opposed":
        return DimensionConfig(
            data_dirs=["texts_RAG_opposed_dimensions"],
            desc_field="Dimensions Descriptions Opp",
            filter_field="dimensions",
        )
    if mode == "both":
        return DimensionConfig(
            data_dirs=["texts_RAG_dimensions", "texts_RAG_opposed_dimensions"],
            desc_field="Dimensions Descriptions",
            filter_field="dimensions",
        )
    raise ValueError(f"Invalid mode: {mode}. Choose from: {SUPPORTED_MODES}")


def run_single_configuration(
    questions_file: str,
    data_root: str,
    model_name: str,
    mode: str,
    template_name: str,
    dimension: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    sleep_time: int,
    num_responses: int,
    output_root: str,
    temperature: float,
) -> None:
    print(
        f"\nRunning model={model_name} | mode={mode} | template={template_name} | dimension={dimension}"
    )

    config = get_mode_config(mode)
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
    if mode == "opposed":
        required_columns.add("Dimensions Descriptions Opp")

    missing = required_columns - set(df_questions.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df_dimension = df_questions[df_questions[config.filter_field] == dimension].copy()
    if df_dimension.empty:
        print(f"[WARN] No rows found for dimension: {dimension}")
        return

    dimension_folder = DIMENSION_MAP[dimension]
    directory_paths = [os.path.join(data_root, data_dir, dimension_folder) for data_dir in config.data_dirs]

    documents = load_documents_from_directories(directory_paths)
    vectorstore = create_vectorstore(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    df_results = generate_responses_for_dimension(
        df_dimension=df_dimension,
        retriever=retriever,
        model=model,
        prompt=prompt,
        desc_field=config.desc_field,
        num_responses=num_responses,
        sleep_time=sleep_time,
    )

    dimension_slug = normalize_dimension_name(dimension)
    output_dir = os.path.join(output_root, f"output_rag_{mode}_{model_name}_{template_name}")
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
        description="Generate RAG-based ideology-aligned answers for selected models, modes, templates, and dimensions."
    )
    parser.add_argument("--questions_file", type=str, default="data/Questions.csv", help="Path to the CSV file.")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name, comma-separated list, or 'all'.")
    parser.add_argument("--mode", type=str, default="standard", help="Mode, comma-separated list, or 'all'.")
    parser.add_argument("--template", type=str, default="template1", help="Template name, comma-separated list, or 'all'.")
    parser.add_argument("--dimension", type=str, default="all", help="Dimension name, comma-separated list, or 'all'.")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for splitting.")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap for splitting.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_K, help="Top-k retrieved chunks.")
    parser.add_argument("--sleep_time", type=int, default=DEFAULT_SLEEP_TIME, help="Sleep time between API calls.")
    parser.add_argument("--num_responses", type=int, default=DEFAULT_NUM_RESPONSES, help="Number of responses per question.")
    parser.add_argument("--output_root", type=str, default="results", help="Root output directory.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions per dimension (for testing).")
    args = parser.parse_args()

    selected_models = parse_list_argument(args.model, list(SUPPORTED_MODELS.keys()), "model")
    selected_modes = parse_list_argument(args.mode, SUPPORTED_MODES, "mode")
    selected_templates = parse_list_argument(args.template, list(TEMPLATES.keys()), "template")
    selected_dimensions = parse_list_argument(args.dimension, SUPPORTED_DIMENSIONS, "dimension")

    combinations = list(product(selected_models, selected_modes, selected_templates, selected_dimensions))
    total = len(combinations)

    print(f"Total runs: {total}")

    for idx, (model_name, mode, template_name, dimension) in enumerate(combinations, start=1):
        print("\n" + "=" * 80)
        print(f"Run {idx}/{total}")
        print(f"Model     : {model_name}")
        print(f"Mode      : {mode}")
        print(f"Template  : {template_name}")
        print(f"Dimension : {dimension}")
        print("=" * 80)

        try:
            run_single_configuration(
                questions_file=args.questions_file,
                data_root=args.data_root,
                model_name=model_name,
                mode=mode,
                template_name=template_name,
                dimension=dimension,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                top_k=args.top_k,
                sleep_time=args.sleep_time,
                num_responses=args.num_responses,
                output_root=args.output_root,
                temperature=args.temperature,
            )
        except Exception as e:
            print(
                f"[ERROR] Failed for model={model_name}, mode={mode}, template={template_name}, dimension={dimension}"
            )
            print(str(e))


if __name__ == "__main__":
    main()