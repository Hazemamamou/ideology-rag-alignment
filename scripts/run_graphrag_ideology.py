import os
import re
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
import tiktoken
from dotenv import load_dotenv

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

load_dotenv()

sys.set_int_max_str_digits(100000)

INPUT_DIR = "input"
OUTPUT_DIR = "output"
RESULTS_DIR = "results"
SETTINGS_FILE = "settings.yaml"

SUPPORTED_MODELS = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
]

SUPPORTED_TEMPLATE_NAMES = [
    "default_graphrag_template",
    "political_discourse_analysis_template",
]

SOURCE_LABELS = {
    "texts_RAG_dimensions": "standard",
    "texts_RAG_opposed_dimensions": "opposed",
}


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


def run_command(command: List[str], error_message: str) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{error_message}: {e}") from e


def initialize_graphrag_if_needed(root: str = ".") -> None:
    settings_path = Path(root) / SETTINGS_FILE
    if settings_path.exists():
        return
    run_command(
        ["python", "-m", "graphrag.index", "--init", "--root", root],
        "Failed to initialize GraphRAG",
    )


def update_settings_file(settings_file_path: str, model: str) -> None:
    with open(settings_file_path, "r", encoding="utf-8") as f:
        settings_data = yaml.safe_load(f)

    settings_data.setdefault("llm", {})
    settings_data["llm"]["model"] = model
    settings_data["llm"]["model_supports_json"] = True

    with open(settings_file_path, "w", encoding="utf-8") as f:
        yaml.dump(settings_data, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Updated {settings_file_path} -> model={model}")


def clear_directory(dir_path: str) -> None:
    p = Path(dir_path)
    if not p.exists():
        return
    for child in p.iterdir():
        try:
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            print(f"[WARN] Could not remove {child}: {e}")


def copy_text_files_to_input(source_folder: str, input_dir: str) -> None:
    source = Path(source_folder)
    dest = Path(input_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")

    copied = 0
    for txt_file in source.rglob("*.txt"):
        target = dest / txt_file.name
        shutil.copyfile(txt_file, target)
        copied += 1

    if copied == 0:
        raise ValueError(f"No .txt files found in {source_folder}")

    print(f"[OK] Copied {copied} text files into {input_dir}")


def load_indexed_data(output_dir: str) -> Tuple:
    lancedb_uri = f"{output_dir}/lancedb"
    community_report_table = "create_final_community_reports"
    entity_table = "create_final_nodes"
    entity_embedding_table = "create_final_entities"
    relationship_table = "create_final_relationships"
    text_unit_table = "create_final_text_units"
    community_level = 2

    entity_df = pd.read_parquet(f"{output_dir}/{entity_table}.parquet")
    entity_embedding_df = pd.read_parquet(f"{output_dir}/{entity_embedding_table}.parquet")
    relationship_df = pd.read_parquet(f"{output_dir}/{relationship_table}.parquet")
    report_df = pd.read_parquet(f"{output_dir}/{community_report_table}.parquet")
    text_unit_df = pd.read_parquet(f"{output_dir}/{text_unit_table}.parquet")

    relationships = read_indexer_relationships(relationship_df)
    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
    reports = read_indexer_reports(report_df, entity_df, community_level)
    text_units = read_indexer_text_units(text_unit_df)

    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=lancedb_uri)
    store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

    return entities, relationships, reports, text_units, description_embedding_store


def build_search_engine(
    model_name: str,
    template: str,
    api_key: str,
    embedding_model: str,
    reports,
    text_units,
    entities,
    relationships,
    description_embedding_store,
    response_type: str,
    temperature: float,
    max_output_tokens: int,
):
    llm = ChatOpenAI(
        api_key=api_key,
        model=model_name,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    llm_params = {
        "max_tokens": max_output_tokens,
        "temperature": temperature,
    }

    return LocalSearch(
        llm=llm,
        context_builder=context_builder,
        system_prompt=template,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type=response_type,
    )


def process_question(
    query: str,
    question_id: str,
    dimension_name: str,
    model_name: str,
    template_name: str,
    template: str,
    api_key: str,
    embedding_model: str,
    reports,
    text_units,
    entities,
    relationships,
    description_embedding_store,
    response_type: str,
    num_responses: int,
    temperature: float,
    max_output_tokens: int,
) -> Optional[dict]:
    try:
        search_engine = build_search_engine(
            model_name=model_name,
            template=template,
            api_key=api_key,
            embedding_model=embedding_model,
            reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            description_embedding_store=description_embedding_store,
            response_type=response_type,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        responses = []
        for i in range(num_responses):
            response = search_engine.search(query).response
            responses.append(response)

            print(f"[INFO] Dimension={dimension_name} | Model={model_name} | Template={template_name}")
            print(f"[INFO] Question {question_id}: {query}")
            print(f"[INFO] Response {i + 1}: {response}")
            print("-" * 80)

        result = {
            "dimension": dimension_name,
            "question_id": question_id,
            "question": query,
        }
        for i, resp in enumerate(responses, start=1):
            result[f"response_{i}"] = resp

        return result

    except Exception as e:
        print(
            f"[ERROR] Failed question_id={question_id}, model={model_name}, template={template_name}: {e}"
        )
        return None


def map_folder_to_dimension(folder_name: str) -> str:
    if not folder_name.startswith("f"):
        return folder_name

    parts = folder_name.split("_")
    if len(parts) != 2:
        return folder_name

    dim_num = parts[0][1:]
    polarity = parts[1]
    dim_polarity = "pos" if polarity == "positive" else "neg"
    return f"Dim. {dim_num} {dim_polarity}"


def dimension_to_slug(dimension_name: str) -> str:
    match = re.match(r"Dim\.\s*(\d+)\s+(neg|pos)", dimension_name)
    if not match:
        raise ValueError(f"Could not convert dimension name to slug: {dimension_name}")
    return f"dim_{match.group(1)}_{match.group(2)}"


def get_dimension_folders(data_root: str, sources: List[str]) -> List[Tuple[str, str]]:
    dimensions = [f"f{i}_{polarity}" for i in range(1, 7) for polarity in ["positive", "negative"]]
    folders = []
    for source in sources:
        for dim in dimensions:
            folders.append((source, str(Path(data_root) / source / dim)))
    return folders


def load_templates() -> dict:
    return {
        "default_graphrag_template": """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.

If you don't know the answer, just say so. Do not make anything up.

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
""",
        "political_discourse_analysis_template": """
---Role---

You are a reliable AI assistant that answers users' questions based on a particular ideology, referred to as a "dimension". This dimension will be explained to you in four ways:
(1) a dimension label;
(2) a dimension description;
(3) lexical items loaded on the dimension;
(4) texts from Endorsed and Controversial treatments illustrating this dimension.

Do not rely on your prior knowledge or general knowledge. Use only the information present in the retrieved context.

---Goal---

Generate a coherent response of the target length and format that accurately answers the user's question, integrating all relevant data from the retrieved context and adhering strictly to the dimension. If no relevant information is available, indicate that you do not know.

---Target response length and format---

{response_type}

---Dimension context---

{context_data}

Style the response in markdown.
""",
    }


def swap_prompts_folder(custom_prompts_dir: Optional[str]) -> Optional[Path]:
    if not custom_prompts_dir:
        return None

    custom_path = Path(custom_prompts_dir)
    prompts_path = Path("prompts")
    backup_path = Path("prompts_backup_temp")

    if not custom_path.exists():
        raise FileNotFoundError(f"Custom prompts directory not found: {custom_prompts_dir}")

    if prompts_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        prompts_path.rename(backup_path)

    shutil.copytree(custom_path, prompts_path)
    print(f"[OK] Replaced prompts/ with contents from {custom_prompts_dir}")
    return backup_path


def restore_prompts_folder(backup_path: Optional[Path]) -> None:
    prompts_path = Path("prompts")
    if prompts_path.exists():
        shutil.rmtree(prompts_path)

    if backup_path and backup_path.exists():
        backup_path.rename(prompts_path)
        print("[OK] Restored original prompts/ folder")


def save_dimension_results(
    rows: List[dict],
    results_root: str,
    source_name: str,
    model_name: str,
    template_name: str,
    dimension_name: str,
) -> None:
    if not rows:
        return

    source_label = SOURCE_LABELS.get(source_name, source_name)
    dim_slug = dimension_to_slug(dimension_name)

    out_dir = Path(results_root) / f"output_graphrag_{source_label}_{model_name}_{template_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    csv_path = out_dir / f"results_{dim_slug}.csv"
    json_path = out_dir / f"results_{dim_slug}.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {json_path}")


def process_dimension_folder(
    source_name: str,
    dimension_folder: str,
    questions_df: pd.DataFrame,
    models: List[str],
    templates: Dict[str, str],
    api_key: str,
    embedding_model: str,
    response_type: str,
    num_responses: int,
    temperature: float,
    max_output_tokens: int,
    input_dir: str,
    output_dir: str,
    settings_file: str,
    results_dir: str,
) -> None:
    clear_directory(input_dir)
    clear_directory(output_dir)

    copy_text_files_to_input(dimension_folder, input_dir)
    update_settings_file(settings_file, models[0])

    run_command(["python", "-m", "graphrag.index", "--root", "."], "Failed to build GraphRAG index")

    entities, relationships, reports, text_units, description_embedding_store = load_indexed_data(output_dir)

    folder_name = Path(dimension_folder).name
    dimension_name = map_folder_to_dimension(folder_name)

    relevant_questions = questions_df[questions_df["dimensions"] == dimension_name]
    if relevant_questions.empty:
        print(f"[WARN] No questions found for dimension {dimension_name} (folder={folder_name})")
        return

    for model_name in models:
        for template_name, template in templates.items():
            rows = []

            for idx, question_row in relevant_questions.iterrows():
                query = question_row["Questions"]
                question_id = str(idx)

                response = process_question(
                    query=query,
                    question_id=question_id,
                    dimension_name=dimension_name,
                    model_name=model_name,
                    template_name=template_name,
                    template=template,
                    api_key=api_key,
                    embedding_model=embedding_model,
                    reports=reports,
                    text_units=text_units,
                    entities=entities,
                    relationships=relationships,
                    description_embedding_store=description_embedding_store,
                    response_type=response_type,
                    num_responses=num_responses,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                if response:
                    rows.append(response)

            save_dimension_results(
                rows=rows,
                results_root=results_dir,
                source_name=source_name,
                model_name=model_name,
                template_name=template_name,
                dimension_name=dimension_name,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run GraphRAG generation for ideology-alignment questions."
    )
    parser.add_argument("--questions_file", type=str, default="data/Questions.csv")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--sources",
        type=str,
        default="texts_RAG_dimensions,texts_RAG_opposed_dimensions",
        help="Comma-separated source folders under data_root.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model(s), comma-separated, or all")
    parser.add_argument("--template", type=str, default="all", help="Template(s), comma-separated, or all")
    parser.add_argument("--response_type", type=str, default="multiple paragraphs")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--num_responses", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_output_tokens", type=int, default=2000)
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--settings_file", type=str, default=SETTINGS_FILE)
    parser.add_argument("--custom_prompts_dir", type=str, default=None, help="Optional replacement for prompts/")
    parser.add_argument("--limit_dimensions", type=int, default=None, help="Process only the first N dimension folders")
    parser.add_argument("--limit_questions", type=int, default=None, help="Limit questions per dimension")
    parser.add_argument("--init_if_needed", action="store_true", help="Initialize GraphRAG if settings.yaml is missing")
    args = parser.parse_args()

    api_key = os.getenv("GRAPHRAG_API_KEY")
    if not api_key:
        raise RuntimeError("GRAPHRAG_API_KEY not found in environment variables or .env")

    if args.init_if_needed:
        initialize_graphrag_if_needed()

    questions_df = pd.read_csv(args.questions_file)
    if "dimensions" not in questions_df.columns or "Questions" not in questions_df.columns:
        raise ValueError("Questions CSV must contain at least 'dimensions' and 'Questions' columns")

    selected_models = parse_list_argument(args.model, SUPPORTED_MODELS, "model")
    selected_templates = parse_list_argument(args.template, SUPPORTED_TEMPLATE_NAMES, "template")
    all_templates = load_templates()
    templates = {k: v for k, v in all_templates.items() if k in selected_templates}

    source_folders = [s.strip() for s in args.sources.split(",") if s.strip()]
    dimension_folders = get_dimension_folders(args.data_root, source_folders)

    if args.limit_dimensions is not None:
        dimension_folders = dimension_folders[: args.limit_dimensions]

    backup_path = None
    try:
        backup_path = swap_prompts_folder(args.custom_prompts_dir)

        for source_name, dimension_folder in dimension_folders:
            if not Path(dimension_folder).exists():
                print(f"[WARN] Dimension folder not found: {dimension_folder}")
                continue

            print("\n" + "=" * 90)
            print(f"Processing dimension folder: {dimension_folder}")
            print("=" * 90)

            folder_name = Path(dimension_folder).name
            dimension_name = map_folder_to_dimension(folder_name)

            local_questions_df = questions_df[questions_df["dimensions"] == dimension_name].copy()
            if args.limit_questions is not None:
                local_questions_df = local_questions_df.head(args.limit_questions)

            if local_questions_df.empty:
                print(f"[WARN] No questions for dimension {dimension_name}")
                continue

            process_dimension_folder(
                source_name=source_name,
                dimension_folder=dimension_folder,
                questions_df=local_questions_df,
                models=selected_models,
                templates=templates,
                api_key=api_key,
                embedding_model=args.embedding_model,
                response_type=args.response_type,
                num_responses=args.num_responses,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                settings_file=args.settings_file,
                results_dir=args.results_dir,
            )

    finally:
        restore_prompts_folder(backup_path)


if __name__ == "__main__":
    main()