import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import mlflow
import ollama
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError, Field, model_validator


# ============================================================================
# CONFIGURATION
# ============================================================================
class ExperimentConfig:
    INPUT_CANDIDATES = [
        "datasets/clinical_clusters.csv",
    ]
    OUTPUT_DIR = "outputs/phase2_enrichment"

    EXPERIMENT_NAME = "phase2_semantic_enrichment"
    TRACKING_URI = "http://127.0.0.1:5000"

    MODELS = ["llama3.1:8b", "mistral:7b"]
    PROMPT_STRATEGIES = ["zero_shot_enrichment"]
    NROWS = [10]

    TEMPERATURE = 0.0

# ============================================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================================

class ClinicalTerminologyInference(BaseModel):
    """Semantic string for medical entity linking and embedding.
    
    Dense clinical format optimized for BERT-based embedding models (SapBERT/ADAv2).
    Output structure: [Main Analyte Name and common synonyms] | [Specimen/Body System] | [Measurement type]
    """
    model_config = ConfigDict(extra="forbid")
    
    # Semantic components for embedding
    analyte: str = Field(description="Main analyte name with common synonyms (e.g., Potassium, K+)")
    specimen: str = Field(description="Specimen type or body system (e.g., Serum, Blood, Urine)")
    measurement_type: str = Field(description="Measurement type: 'quantitative' or 'qualitative'")


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def resolve_input_path(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Input not found: {p}")

    for c in ExperimentConfig.INPUT_CANDIDATES:
        p = Path(c)
        if p.exists():
            return p

    raise FileNotFoundError(
        "No input CSV found. Expected one of: "
        + ", ".join(ExperimentConfig.INPUT_CANDIDATES)
    )


def build_profile(row: pd.Series) -> Dict[str, Any]:
    return {
        "valueuom": row.get("valueuom", ""),
        "ref_range_lower": row.get("ref_range_lower", ""),
        "ref_range_upper": row.get("ref_range_upper", ""),
        "valuenum_min": row.get("valuenum_min", ""),
        "valuenum_max": row.get("valuenum_max", ""),
        "valuenum_mean": row.get("valuenum_mean", ""),
        "valuenum_median": row.get("valuenum_median", ""),
        "most_frequent_value": row.get("most_frequent_value", ""),
        "abnormal_pct": row.get("abnormal_pct", ""),
        "most_frequent_comment": row.get("most_frequent_comment", ""),
        "sample_count": row.get("sample_count", ""),
    }


def load_prompt_template(prompt_strategy: str) -> str:
    """Load prompt from MLflow prompt registry."""
    try:
        prompt_obj = mlflow.genai.load_prompt(f"prompts:/{prompt_strategy}@latest")
        
        # Handle different MLflow prompt object shapes
        if isinstance(prompt_obj, str):
            return prompt_obj
        
        # Try common attributes
        for attr in ("template", "text", "prompt", "content"):
            value = getattr(prompt_obj, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        
        return str(prompt_obj)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load prompt '{prompt_strategy}' from MLflow registry. "
            f"Make sure it's registered as 'prompts:/{prompt_strategy}@latest'. "
            f"Error: {str(e)}"
        )


def build_prompt(profile: Dict[str, Any], prompt_template: str) -> str:
    """Build prompt by replacing placeholders with profile values using regex.
    
    Uses regex instead of .format() to avoid conflicts with JSON braces in the template.
    """
    result = prompt_template
    
    # Replace all placeholders like {key} with values from profile or empty string
    placeholders = [
        "valueuom", "ref_range_lower", "ref_range_upper",
        "valuenum_min", "valuenum_max", "valuenum_mean", "valuenum_median",
        "most_frequent_value", "abnormal_pct", "most_frequent_comment",
        "sample_count"
    ]
    
    for key in placeholders:
        value = profile.get(key, "")
        result = re.sub(r"\{" + key + r"\}", str(value), result)
    
    # Replace profile_json placeholder
    profile_json_str = json.dumps(profile, ensure_ascii=False, indent=2)
    result = re.sub(r"\{profile_json\}", profile_json_str, result)
    
    return result


def query_ollama(model: str, prompt: str) -> Dict[str, Any]:
    response = ollama.generate(
        model=model,
        prompt=prompt,
        format=ClinicalTerminologyInference.model_json_schema(),  # force JSON conforming to schema
        options={"temperature": ExperimentConfig.TEMPERATURE},
    )
    return {
        "text": response.get("response", "").strip(),
        "total_duration": response.get("total_duration", 0),
        "eval_count": response.get("eval_count", 0),
        "prompt_eval_count": response.get("prompt_eval_count", 0),
    }

def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    if "```json" in cleaned:
        start = cleaned.find("```json") + 7
        end = cleaned.find("```", start)
        return cleaned[start:end].strip()
    if "```" in cleaned:
        start = cleaned.find("```") + 3
        end = cleaned.find("```", start)
        return cleaned[start:end].strip()
    return cleaned


def validate_llm_output(response_text: str) -> Tuple[bool, Optional[ClinicalTerminologyInference], str]:
    raw_json = _extract_json_block(response_text)
    try:
        payload = json.loads(raw_json)

        # check exact keys (strict output)
        expected_keys = set(ClinicalTerminologyInference.model_fields.keys())
        if set(payload.keys()) != expected_keys:
            return False, None, raw_json

        parsed = ClinicalTerminologyInference(**payload)
        return True, parsed, raw_json
    except (json.JSONDecodeError, ValidationError):
        return False, None, raw_json


def single_run(
    model: str,
    df: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    prompt_strategy: str,
    nrows: int,
) -> pd.DataFrame:
    prompt_template = load_prompt_template(prompt_strategy)

    rows: List[Dict[str, Any]] = []
    validation_ok = 0
    failures = 0
    total_latency_ns = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model)
        mlflow.log_param("prompt_strategy", prompt_strategy)
        mlflow.log_param("nrows", nrows)
        mlflow.log_param("input_rows", len(df))
        mlflow.log_param("temperature", ExperimentConfig.TEMPERATURE)

        sample_prompt_logged = False
        sample_response_logged = False

        for idx, row in df.iterrows():
            profile = build_profile(row)
            prompt = build_prompt(profile, prompt_template)

            try:
                resp = query_ollama(model, prompt)
                total_latency_ns += int(resp.get("total_duration", 0))
                ok, parsed, raw_json = validate_llm_output(resp["text"])

                if ok and parsed is not None:
                    validation_ok += 1
                    result = parsed.model_dump()
                else:
                    result = {
                        "analyte": None,
                        "specimen": None,
                        "measurement_type": None,
                    }

                if not sample_prompt_logged:
                    mlflow.log_text(prompt, artifact_file="sample_prompt.txt")
                    sample_prompt_logged = True

                if not sample_response_logged:
                    mlflow.log_text(resp["text"], artifact_file="sample_response.txt")
                    sample_response_logged = True

                rows.append(
                    {
                        "model": model,
                        "inferred_analyte": result["analyte"],
                        "inferred_specimen": result["specimen"],
                        "inferred_measurement_type": result["measurement_type"],
                        "validation_success": ok,
                        "raw_response": resp["text"],
                        "parsed_json": raw_json,
                    }
                )

            except Exception as e:
                failures += 1
                rows.append(
                    {
                        "model": model,
                        "inferred_analyte": None,
                        "inferred_specimen": None,
                        "inferred_measurement_type": None,
                        "validation_success": False,
                        "raw_response": str(e),
                        "parsed_json": "",
                    }
                )

            if (idx + 1) % 20 == 0:
                print(f"[{model} | {prompt_strategy} | {nrows}] Processed {idx + 1}/{len(df)} rows")

        out_df = pd.DataFrame(rows)

        success_rate = (validation_ok / len(df)) if len(df) else 0.0
        mean_latency_sec = (total_latency_ns / 1e9 / len(df)) if len(df) else 0.0

        mlflow.log_metric("validation_success_count", validation_ok)
        mlflow.log_metric("validation_success_rate", success_rate)
        mlflow.log_metric("failures", failures)
        mlflow.log_metric("mean_latency_sec", mean_latency_sec)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("experiment_date", datetime.now().strftime("%Y-%m-%d"))

        print(f"✓ Completed: {model} | {prompt_strategy} | {nrows} rows")
        return out_df


def run_experiment(input_csv: Optional[str] = None) -> None:
    input_path = resolve_input_path(input_csv)
    output_dir = Path(ExperimentConfig.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_outputs: Dict[str, List[pd.DataFrame]] = {model: [] for model in ExperimentConfig.MODELS}

    for model in ExperimentConfig.MODELS:
        for prompt_strategy in ExperimentConfig.PROMPT_STRATEGIES:
            for nrows in ExperimentConfig.NROWS:
                df = pd.read_csv(input_path, nrows=nrows)
                if "itemid" in df.columns:
                    df = df.drop(columns=["itemid"])

                run_name = f"{model.replace(':', '_')}_{prompt_strategy}_{nrows}_phase2_enrichment"
                result_df = single_run(
                    model=model,
                    df=df,
                    output_dir=output_dir,
                    run_name=run_name,
                    prompt_strategy=prompt_strategy,
                    nrows=nrows,
                )
                model_outputs[model].append(result_df)

    for model, dfs in model_outputs.items():
        if dfs:
            combined_model_df = pd.concat(dfs, ignore_index=True)
            model_json = output_dir / f"enriched_{model.replace(':', '_')}.json"
            combined_model_df.to_json(
                model_json,
                orient="records",
                indent=2,
                force_ascii=False,
            )
            mlflow.log_artifact(str(model_json))
            print(f"✓ Model output saved: {model_json}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(ExperimentConfig.TRACKING_URI)
    mlflow.set_experiment(ExperimentConfig.EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {ExperimentConfig.TRACKING_URI}")

    run_experiment()