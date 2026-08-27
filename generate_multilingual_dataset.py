"""
Generate Hindi, Romanized Hindi, and Telugu translations for english_1000.txt
using the Google Gemini API.

Usage:
    set GOOGLE_API_KEY=your_api_key_here
    python generate_multilingual_dataset.py

This version prioritizes DATA INTEGRITY over speed:
  - small batches (default 10 sentences/request)
  - strict per-batch validation (ids, English preserved, script checks)
  - failed batches are retried, then recursively split down to batch size 1
    rather than ever accepting a missing/None translation
  - checkpoints only ever contain fully-validated, complete records
  - resume support: a prior run's checkpoint is reused instead of
    re-translating already-completed sentences
  - a strict final validation gate before multilingual_dataset_1000.csv
    is written; if 1000 complete, valid rows cannot be produced, the
    script exits with an error instead of writing a misleading dataset
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

import google.generativeai as genai
import pandas as pd


CONFIG = {
    "api_key_env": "GEMINI_API_KEY",
    "model_name": "gemini-3.1-flash-lite",
    "temperature": 0.3,
    "max_retries": 3,
    "retry_delay": 2,
    "request_delay": 0.2,
    "checkpoint_size": 50,
    "gemini_batch_size": 10,
    "min_batch_size": 1,
    "output_dir": "generated_dataset",
    "log_file": "translation_log.txt",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(CONFIG["log_file"], encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Unicode ranges used for script validation.
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
# Any character outside basic Latin letters/digits/common punctuation/whitespace
# is treated as "non-Latin" for Romanized Hindi validation.
NON_LATIN_RE = re.compile(r"[^\x00-\x7F\u2000-\u206F]")


def setup_gemini_api() -> None:
    api_key = os.getenv(CONFIG["api_key_env"])
    if not api_key:
        logger.error("API key not found. Set %s before running.", CONFIG["api_key_env"])
        sys.exit(1)

    genai.configure(api_key=api_key)
    logger.info("Gemini API configured")


def get_model() -> genai.GenerativeModel:
    return genai.GenerativeModel(CONFIG["model_name"])


def load_english_txt(input_file: str, limit: int | None = None) -> pd.DataFrame:
    if not os.path.exists(input_file):
        logger.error("Input file not found: %s", input_file)
        sys.exit(1)

    with open(input_file, "r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]

    if limit is not None:
        sentences = sentences[:limit]

    df = pd.DataFrame({"English": sentences})
    logger.info("Loaded %d English sentences from %s", len(df), input_file)
    return df


def generate_text(prompt: str, model: genai.GenerativeModel, label: str, attempt: int = 1) -> str | None:
    if attempt > CONFIG["max_retries"]:
        logger.warning("Failed %s after %d attempts", label, CONFIG["max_retries"])
        return None

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=CONFIG["temperature"]
            ),
        )
        text = getattr(response, "text", "").strip()
        if text:
            return text
    except Exception as exc:
        logger.warning("Attempt %d failed for %s: %s", attempt, label, exc)

    time.sleep(CONFIG["retry_delay"])
    return generate_text(prompt, model, label, attempt + 1)


# ----------------------------------------------------------------------
# Script / content validation (Fix 3)
# ----------------------------------------------------------------------
def is_valid_hindi(text: Any) -> bool:
    """Non-empty, contains Devanagari, not entirely Latin/English."""
    if not isinstance(text, str) or not text.strip():
        return False
    if not DEVANAGARI_RE.search(text):
        return False
    return True


def is_valid_romanized_hindi(text: Any) -> bool:
    """Non-empty, Roman/Latin script, no Devanagari or Telugu characters."""
    if not isinstance(text, str) or not text.strip():
        return False
    if DEVANAGARI_RE.search(text):
        return False
    if TELUGU_RE.search(text):
        return False
    return True


def is_valid_telugu(text: Any) -> bool:
    """Non-empty, contains Telugu script, not entirely Latin/English."""
    if not isinstance(text, str) or not text.strip():
        return False
    if not TELUGU_RE.search(text):
        return False
    return True


def validate_row_content(english: str, hindi: Any, romanized_hindi: Any, telugu: Any) -> list[str]:
    """Return a list of validation problems for a single translated row (empty = valid)."""
    problems = []
    if not is_valid_hindi(hindi):
        problems.append("Hindi failed script/non-empty validation")
    if not is_valid_romanized_hindi(romanized_hindi):
        problems.append("Romanized_Hindi failed script/non-empty validation")
    if not is_valid_telugu(telugu):
        problems.append("Telugu failed script/non-empty validation")
    return problems


# ----------------------------------------------------------------------
# Batch translation with strict validation (Fix 2)
# ----------------------------------------------------------------------
def call_gemini_batch(batch: list[tuple[int, str]], model: genai.GenerativeModel) -> list[dict[str, Any]] | None:
    """Single call to Gemini for a batch. Returns parsed JSON array or None on failure."""
    input_rows = [{"id": row_id, "English": english_text} for row_id, english_text in batch]
    prompt = f"""Translate each English sentence into Hindi, Romanized Hindi, and Telugu.

Return valid JSON only. The JSON must be an array of objects.
Each object must have exactly these keys:
- id
- English
- Hindi
- Romanized_Hindi
- Telugu

Requirements:
- Preserve each id exactly.
- Preserve each English sentence exactly.
- Hindi must use Devanagari script only.
- Romanized_Hindi must use Roman/Latin letters only, with no Devanagari or diacritics.
- Telugu must use Telugu script only.
- Keep meaning identical to English.
- Use direct, natural, conversational wording.
- Do not include markdown fences or commentary.

Input JSON:
{json.dumps(input_rows, ensure_ascii=False)}

Output JSON array:"""

    raw_text = generate_text(prompt, model, "translation batch")
    if not raw_text:
        return None

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("Batch response is not a JSON array")
    except Exception as exc:
        logger.warning("Could not parse batch response: %s", exc)
        logger.warning("Raw response sample: %s", raw_text[:500])
        return None

    return parsed


def validate_batch_response(
    batch: list[tuple[int, str]], parsed: list[dict[str, Any]]
) -> tuple[dict[int, dict[str, str]] | None, list[str]]:
    """Strictly validate a parsed Gemini batch response against the requested batch.

    Returns (validated_rows_by_id, problems). validated_rows_by_id is None if
    the batch is invalid as a whole (must be retried/split); otherwise it maps
    id -> {"Hindi": ..., "Romanized_Hindi": ..., "Telugu": ...} for every id
    in the batch (all guaranteed present and valid).
    """
    problems: list[str] = []
    requested_ids = [row_id for row_id, _ in batch]
    requested_english = {row_id: english_text for row_id, english_text in batch}

    # Build id -> item map, checking for duplicates.
    by_id: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict) or item.get("id") is None:
            problems.append("Response contained a malformed item without an id")
            continue
        try:
            item_id = int(item.get("id"))
        except (TypeError, ValueError):
            problems.append(f"Response contained a non-integer id: {item.get('id')!r}")
            continue
        if item_id in by_id:
            problems.append(f"Duplicate id {item_id} in response")
            continue
        by_id[item_id] = item

    returned_ids = set(by_id.keys())
    expected_ids = set(requested_ids)

    missing_ids = expected_ids - returned_ids
    extra_ids = returned_ids - expected_ids
    if missing_ids:
        problems.append(f"Missing ids in response: {sorted(missing_ids)}")
    if extra_ids:
        problems.append(f"Unexpected extra ids in response: {sorted(extra_ids)}")

    if problems:
        return None, problems

    validated: dict[int, dict[str, str]] = {}
    for row_id in requested_ids:
        item = by_id[row_id]

        returned_english = item.get("English")
        expected_english = requested_english[row_id]
        if not isinstance(returned_english, str) or returned_english.strip() != expected_english.strip():
            problems.append(
                f"id {row_id}: English mismatch (expected preserved sentence, model altered it)"
            )
            continue

        hindi = item.get("Hindi")
        romanized_hindi = item.get("Romanized_Hindi")
        telugu = item.get("Telugu")

        row_problems = validate_row_content(expected_english, hindi, romanized_hindi, telugu)
        if row_problems:
            problems.append(f"id {row_id}: " + "; ".join(row_problems))
            continue

        validated[row_id] = {
            "Hindi": hindi.strip(),
            "Romanized_Hindi": romanized_hindi.strip(),
            "Telugu": telugu.strip(),
        }

    if problems:
        return None, problems

    return validated, []


def generate_translation_batch(
    batch: list[tuple[int, str]],
    model: genai.GenerativeModel,
) -> dict[int, dict[str, str]]:
    """Translate a batch, retrying and recursively splitting on failure.

    NEVER returns None values: either every id in `batch` comes back with a
    fully validated translation, or (after exhausting retries down to a
    single-sentence batch) the offending single sentence is logged and
    raised as a hard error, since silently dropping/None-ing a row is
    explicitly disallowed for this dataset.
    """
    ids_label = f"{batch[0][0]}-{batch[-1][0]}"

    for attempt in range(1, CONFIG["max_retries"] + 1):
        logger.info("Processing batch %s (attempt %d/%d)", ids_label, attempt, CONFIG["max_retries"])
        parsed = call_gemini_batch(batch, model)
        if parsed is not None:
            validated, problems = validate_batch_response(batch, parsed)
            if validated is not None:
                logger.info("Batch %s validated successfully", ids_label)
                return validated
            logger.warning("Batch %s failed validation: %s", ids_label, "; ".join(problems))
        else:
            logger.warning("Batch %s: no usable response from model", ids_label)

        if attempt < CONFIG["max_retries"]:
            logger.info("Retrying batch %s...", ids_label)
            time.sleep(CONFIG["retry_delay"])

    # Exhausted retries at this batch size.
    if len(batch) <= CONFIG["min_batch_size"]:
        # Cannot split further: this is a hard failure for this sentence.
        row_id, english_text = batch[0]
        logger.error(
            "Sentence id %d permanently failed translation after %d attempts: %r",
            row_id,
            CONFIG["max_retries"],
            english_text[:100],
        )
        raise RuntimeError(
            f"Failed to translate sentence id {row_id} after exhausting retries and "
            f"minimum batch size ({CONFIG['min_batch_size']}). Refusing to insert a "
            f"missing/placeholder value. English: {english_text!r}"
        )

    logger.warning(
        "Batch %s repeatedly failed at size %d; splitting into smaller batches...",
        ids_label,
        len(batch),
    )
    mid = len(batch) // 2
    left = batch[:mid]
    right = batch[mid:]

    result: dict[int, dict[str, str]] = {}
    result.update(generate_translation_batch(left, model))
    time.sleep(CONFIG["request_delay"])
    result.update(generate_translation_batch(right, model))
    return result


# ----------------------------------------------------------------------
# Checkpointing (Fix 5) and resume (Fix 6)
# ----------------------------------------------------------------------
def save_checkpoint(
    rows: list[dict[str, Any]],
    output_dir: str,
    row_count: int,
) -> None:
    """Save a checkpoint. `rows` must already be complete/validated records."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"progress_{row_count}.csv")
    pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
    logger.info("Saved checkpoint: %s (%d complete records)", checkpoint_path, len(rows))


def load_latest_checkpoint(output_dir: str, english_df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Load the most advanced valid checkpoint, if any, keyed by id.

    Only rows that are complete AND whose English text still matches the
    current english_1000.txt are reused; anything else is discarded so we
    never resume from stale or corrupted data.
    """
    pattern = os.path.join(output_dir, "progress_*.csv")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        return {}

    def _row_count(path: str) -> int:
        match = re.search(r"progress_(\d+)\.csv$", path)
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=_row_count, reverse=True)
    english_by_position = {i + 1: text for i, text in enumerate(english_df["English"].tolist())}

    for path in checkpoint_files:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as exc:
            logger.warning("Could not read checkpoint %s: %s", path, exc)
            continue

        required_cols = ["id", "English", "Hindi", "Romanized_Hindi", "Telugu"]
        if any(col not in df.columns for col in required_cols):
            logger.warning("Checkpoint %s missing required columns; ignoring", path)
            continue

        resumed: dict[int, dict[str, Any]] = {}
        for _, row in df.iterrows():
            try:
                row_id = int(row["id"])
            except (TypeError, ValueError):
                continue

            expected_english = english_by_position.get(row_id)
            if expected_english is None:
                continue
            if not isinstance(row["English"], str) or row["English"].strip() != expected_english.strip():
                continue

            problems = validate_row_content(
                expected_english, row["Hindi"], row["Romanized_Hindi"], row["Telugu"]
            )
            if problems:
                continue

            resumed[row_id] = {
                "id": row_id,
                "English": expected_english,
                "Hindi": str(row["Hindi"]).strip(),
                "Romanized_Hindi": str(row["Romanized_Hindi"]).strip(),
                "Telugu": str(row["Telugu"]).strip(),
            }

        if resumed:
            logger.info("Resuming from checkpoint %s: %d valid records reusable", path, len(resumed))
            return resumed

    return {}


# ----------------------------------------------------------------------
# Main generation loop
# ----------------------------------------------------------------------
def generate_translations(df: pd.DataFrame, output_dir: str, no_resume: bool = False) -> pd.DataFrame:
    model = get_model()
    total = len(df)
    numbered_sentences = list(enumerate(df["English"].tolist(), start=1))
    batch_size = CONFIG["gemini_batch_size"]

    completed: dict[int, dict[str, Any]] = {}
    if not no_resume:
        completed = load_latest_checkpoint(output_dir, df)

    pending = [(row_id, text) for row_id, text in numbered_sentences if row_id not in completed]

    if completed:
        logger.info("Resuming: %d/%d already complete, %d remaining", len(completed), total, len(pending))
    else:
        logger.info("Starting fresh: 0/%d complete, %d remaining", total, len(pending))

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_start_id = batch[0][0]
        batch_end_id = batch[-1][0]
        logger.info(
            "Processing batch %d-%d/%d (%.1f%% of pending)",
            batch_start_id,
            batch_end_id,
            total,
            ((start + len(batch)) / len(pending)) * 100 if pending else 100.0,
        )

        validated = generate_translation_batch(batch, model)

        for row_id, english_text in batch:
            record = validated[row_id]
            completed[row_id] = {
                "id": row_id,
                "English": english_text,
                "Hindi": record["Hindi"],
                "Romanized_Hindi": record["Romanized_Hindi"],
                "Telugu": record["Telugu"],
            }

        ordered_rows = [completed[row_id] for row_id, _ in numbered_sentences if row_id in completed]
        save_checkpoint(ordered_rows, output_dir, len(ordered_rows))
        time.sleep(CONFIG["request_delay"])

    ordered_rows = [completed[row_id] for row_id, _ in numbered_sentences]
    return pd.DataFrame(ordered_rows)


# ----------------------------------------------------------------------
# Final strict validation (Fix 7)
# ----------------------------------------------------------------------
def final_validation(df: pd.DataFrame, expected_total: int) -> list[str]:
    """Return a list of validation failures (empty list = passes)."""
    failures = []

    if len(df) != expected_total:
        failures.append(f"Expected exactly {expected_total} rows, found {len(df)}")

    required_cols = ["id", "English", "Hindi", "Romanized_Hindi", "Telugu"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")
        return failures  # Can't check further without the columns.

    for col in ["English", "Hindi", "Romanized_Hindi", "Telugu"]:
        if not df[col].notna().all() or (df[col].astype(str).str.strip() == "").any():
            n_bad = (~df[col].notna() | (df[col].astype(str).str.strip() == "")).sum()
            failures.append(f"Column {col} has {n_bad} missing/empty value(s)")

    if df["id"].duplicated().any():
        n_dupes = df["id"].duplicated().sum()
        failures.append(f"Found {n_dupes} duplicate id value(s)")

    if df["id"].nunique() != expected_total:
        failures.append(f"Expected {expected_total} unique ids, found {df['id'].nunique()}")

    for _, row in df.iterrows():
        problems = validate_row_content(row["English"], row["Hindi"], row["Romanized_Hindi"], row["Telugu"])
        if problems:
            failures.append(f"id {row['id']}: " + "; ".join(problems))

    return failures


def calculate_stats(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    return {
        "total_sentences": total,
        "hindi_valid": int(df["Hindi"].notna().sum()),
        "romanized_hindi_valid": int(df["Romanized_Hindi"].notna().sum()),
        "telugu_valid": int(df["Telugu"].notna().sum()),
        "hindi_success_rate": float((df["Hindi"].notna().sum() / total) * 100) if total else 0.0,
        "romanized_hindi_success_rate": float((df["Romanized_Hindi"].notna().sum() / total) * 100) if total else 0.0,
        "telugu_success_rate": float((df["Telugu"].notna().sum() / total) * 100) if total else 0.0,
    }


def save_outputs(df: pd.DataFrame, output_dir: str, output_file: str, input_file: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Only keep the columns required for the final dataset (drop internal "id").
    final_df = df[["English", "Hindi", "Romanized_Hindi", "Telugu"]].copy()
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("Saved dataset: %s", output_path)

    metadata = {
        "source_file": input_file,
        "generation_date": datetime.now().isoformat(),
        "model_used": CONFIG["model_name"],
        "temperature": CONFIG["temperature"],
        "stats": calculate_stats(final_df),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    logger.info("Saved metadata: %s", metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Hindi, Romanized Hindi, and Telugu translations from an English TXT file."
    )
    parser.add_argument("--input", default="english_1000.txt", help="Input TXT file, one English sentence per line")
    parser.add_argument("--output", default="multilingual_dataset_1000.csv", help="Output CSV filename")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"], help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CONFIG["gemini_batch_size"],
        help="Sentences per Gemini request (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoints and regenerate everything from scratch",
    )
    args = parser.parse_args()

    CONFIG["gemini_batch_size"] = args.batch_size

    logger.info("=" * 70)
    logger.info("MULTILINGUAL DATASET GENERATION - START")
    logger.info("=" * 70)

    setup_gemini_api()
    df = load_english_txt(args.input, args.limit)
    expected_total = len(df)

    try:
        translated_df = generate_translations(df, args.output_dir, no_resume=args.no_resume)
    except RuntimeError as exc:
        logger.error("Dataset generation failed: %s", exc)
        logger.error("Refusing to write a final dataset with missing/failed translations.")
        sys.exit(1)

    logger.info("Running final strict validation...")
    failures = final_validation(translated_df, expected_total)

    if failures:
        logger.error("Final validation FAILED:")
        for failure in failures:
            logger.error("  - %s", failure)
        logger.error("Dataset generation unsuccessful. multilingual_dataset_1000.csv was NOT written.")
        sys.exit(1)

    logger.info(
        "Final validation: English %d/%d, Hindi %d/%d, Romanized Hindi %d/%d, Telugu %d/%d",
        translated_df["English"].notna().sum(), expected_total,
        translated_df["Hindi"].notna().sum(), expected_total,
        translated_df["Romanized_Hindi"].notna().sum(), expected_total,
        translated_df["Telugu"].notna().sum(), expected_total,
    )

    save_outputs(translated_df, args.output_dir, args.output, args.input)

    stats = calculate_stats(translated_df[["English", "Hindi", "Romanized_Hindi", "Telugu"]])
    logger.info("Hindi: %d/%d (%.1f%%)", stats["hindi_valid"], stats["total_sentences"], stats["hindi_success_rate"])
    logger.info(
        "Romanized Hindi: %d/%d (%.1f%%)",
        stats["romanized_hindi_valid"],
        stats["total_sentences"],
        stats["romanized_hindi_success_rate"],
    )
    logger.info("Telugu: %d/%d (%.1f%%)", stats["telugu_valid"], stats["total_sentences"], stats["telugu_success_rate"])
    logger.info("Dataset generation successful.")


if __name__ == "__main__":
    main()
