"""
Multi-tokenizer comparison analysis (v2).

Compares tokenization of an aligned English / Hindi / Romanized-Hindi / Telugu
dataset across five tokenizers:

    - GPT-4 / GPT-3.5      (tiktoken: cl100k_base)
    - GPT-3                (tiktoken: p50k_base)
    - mBERT                (bert-base-multilingual-cased)
    - XLM-RoBERTa          (xlm-roberta-base)
    - IndicBERT            (ai4bharat/IndicBERTv2-MLM-only)

This script ONLY analyzes an existing CSV
(``multilingual_dataset_1000.csv`` by default, with columns English, Hindi,
Romanized_Hindi, Telugu). It does not generate or modify any dataset.

====================================================================
TOKEN-COUNTING METHODOLOGY (read this before trusting the numbers)
====================================================================
tiktoken (cl100k_base / p50k_base) has no special/added tokens for plain
text encoding, so ``encode(text)`` already returns content-token counts.

Hugging Face tokenizers (mBERT, XLM-RoBERTa, IndicBERT) add model-specific
special tokens ([CLS]/[SEP], <s>/</s>, etc.) when ``add_special_tokens=True``.
To make counts comparable across backends, this script ALWAYS counts content
tokens only:

    - HF tokenizers: ``tokenizer.encode(text, add_special_tokens=False)``
    - tiktoken:       ``tokenizer.encode(text)``  (no special tokens exist)

This is implemented once, centrally, in ``TokenizerRegistry.tokenize_ids``,
so every metric below (mean tokens, fertility, overhead ratio, parity, STRR,
Gini) is computed from the same content-token counts.

====================================================================
METRICS IMPLEMENTED
====================================================================
1. Average token count (mean / std / median) per Tokenizer x Language.

2. Fertility = total content tokens / total whitespace-delimited words,
   per Tokenizer x Language. (Word segmentation = str.split() on
   whitespace; this is a deliberate, documented simplification -- it is
   NOT linguistically informed word segmentation for Hindi/Telugu, but it
   is applied identically to every language so it is comparable within
   this experiment.)

3. Tokenization Overhead Ratio (target language tokens / English tokens):
     - Aggregate ratio  = sum(target tokens) / sum(English tokens)
     - Per-sentence ratio distribution (mean, median, std, 95% CI)
   This is explicitly NOT called a "fairness score".

4. Tokenization Parity (Ahia et al., "Do All Languages Cost the Same?";
   framing used in "Tokenization and Representation Biases in
   Multilingual Models on Dialectal NLP Tasks" and "TokLens: A
   Multilingual Lens on Tokenizer Quality for LLMs"): for an aligned
   parallel corpus, parity is the ratio of target-language tokenization
   cost to English tokenization cost -- i.e. Parity_Ratio =
   total_target_tokens / total_english_tokens. For this aligned-sentence
   experimental design this is mathematically identical to the aggregate
   Tokenization Overhead Ratio from metric 3. The script does NOT pretend
   these are independent numbers -- ``compute_parity`` calls the same
   aggregate-ratio function used for overhead and the printed/CSV output
   says so explicitly.

5. Single-Token Retention Rate (STRR), computed at the WORD level:
   for every whitespace-delimited word in a language's sentences,
   tokenize just that word, strip special tokens, and check whether it
   maps to exactly one content token. STRR = (# words with exactly one
   content token) / (total words).

   METHODOLOGICAL CAVEAT -- isolated-word, not contextual, retention: each
   word is tokenized ON ITS OWN, outside of the sentence it came from. BPE-
   and unigram-style tokenizers can (and often do) merge across word
   boundaries or split differently depending on surrounding context,
   whitespace, and punctuation attached in-sentence (e.g. a leading space
   before a token, or a trailing comma). Retokenizing a bare word therefore
   does not always reproduce the token sequence that word would receive
   inside its original sentence. STRR as implemented here measures
   STANDALONE single-token retention -- i.e. "if you handed the tokenizer
   this word by itself, would it come back as one content token?" -- and
   should be interpreted as a property of the tokenizer's vocabulary
   coverage for that word in isolation, not as a claim about how often the
   word survives as a single token when embedded in running text.

6. Gini coefficient (Ahia et al. framing; also used in "Parity-Aware
   Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization")
   over the 4 language-level mean-tokens-per-sentence values for each
   tokenizer. Gini = 0 means equal average tokenization cost across
   English/Hindi/Romanized Hindi/Telugu for that tokenizer.

   METHODOLOGICAL CAVEAT -- raw token-cost inequality vs. tokenization
   efficiency: this Gini is a DESCRIPTIVE inequality measure computed on RAW
   mean token counts per aligned sentence, not a length-normalized or
   information-theoretic efficiency measure. Its four inputs already bake in
   whatever surface-form differences exist between the languages (e.g.
   Devanagari/Telugu script needing more sub-word pieces than Latin script
   for a tokenizer trained mostly on English, or the aligned translations
   simply using a different number of words to say the same thing). A high
   Gini says the *token cost* is unevenly distributed across languages for
   that tokenizer; it does NOT by itself mean the tokenizer is "inefficient"
   for any one language, and it does not control for word count, sentence
   length, or information content -- those are what fertility (metric 2)
   measures instead. Two tokenizers can show similar Gini values for very
   different underlying reasons (e.g. one uniformly costly across all four
   languages, another tight on English and comparatively fine elsewhere) if
   their raw token-count spreads happen to be proportionally similar. Read
   Gini and fertility together, not Gini alone, as an efficiency verdict:
   Gini describes distributional (in)equality of raw token cost across
   languages; fertility describes tokenization efficiency within each
   language.

====================================================================
STATISTICS
====================================================================
The dataset is a paired/aligned parallel corpus (each row = the same
sentence in 4 languages), so per-sentence ratios (metric 3) are computed
row-wise on paired data, and the 95% CI on the sentence-level ratio uses
a standard normal approximation of the mean (t/normal CI on the paired
per-sentence ratios). No unpaired/independent-samples tests are used.

====================================================================
USAGE
====================================================================
    pip install tiktoken transformers torch pandas numpy scipy matplotlib seaborn sentencepiece
    python multi_tokenizer_analysis.py --input generated_dataset/multilingual_dataset_1000.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Run: pip install tiktoken")

try:
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers")

try:
    from scipy import stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


CONFIG = {
    "input_file": os.path.join("generated_dataset", "multilingual_dataset_1000.csv"),
    "output_dir": "multi_tokenizer_results",
    "confidence_level": 0.95,
    "dpi": 300,
    "required_columns": ["English", "Hindi", "Romanized_Hindi", "Telugu"],
    "languages": ["English", "Hindi", "Romanized_Hindi", "Telugu"],
    "target_languages": ["Hindi", "Romanized_Hindi", "Telugu"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f"multi_tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tokenizer registry
# ----------------------------------------------------------------------
class TokenizerRegistry:
    """Load tokenizer backends and expose a uniform content-token API."""

    def __init__(self) -> None:
        self.tokenizers: dict[str, Any] = {}
        self.backends: dict[str, str] = {}
        self._initialize_tokenizers()

    def _initialize_tokenizers(self) -> None:
        logger.info("Initializing tokenizers...")

        if TIKTOKEN_AVAILABLE:
            tiktoken_models = {
                "GPT-4 / GPT-3.5 (cl100k_base)": "cl100k_base",
                "GPT-3 (p50k_base)": "p50k_base",
            }
            for name, encoding_name in tiktoken_models.items():
                try:
                    self.tokenizers[name] = tiktoken.get_encoding(encoding_name)
                    self.backends[name] = "tiktoken"
                    logger.info("Loaded %s", name)
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", name, exc)

        if TRANSFORMERS_AVAILABLE:
            hf_models = {
                "mBERT": "bert-base-multilingual-cased",
                "XLM-RoBERTa": "xlm-roberta-base",
                "IndicBERT": "ai4bharat/IndicBERTv2-MLM-only",
            }
            for name, model_id in hf_models.items():
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    self.tokenizers[name] = tokenizer
                    self.backends[name] = "huggingface"
                    logger.info("Loaded %s", name)
                except Exception as exc:
                    logger.warning("Failed to load %s (%s): %s", name, model_id, exc)

    def get_available(self) -> list[str]:
        return list(self.tokenizers.keys())

    def tokenize_ids(self, text: Any, tokenizer_name: str) -> list[int] | None:
        """Return CONTENT-ONLY token ids for `text` (no special tokens).

        This is the single place special-token handling happens: every
        metric in this script goes through this method (directly, or via
        `tokenize_count`), so tiktoken and Hugging Face backends are
        always compared on a like-for-like basis.
        """
        if tokenizer_name not in self.tokenizers or pd.isna(text):
            return None

        text = str(text)
        if text.strip() == "":
            return None

        try:
            tokenizer = self.tokenizers[tokenizer_name]
            if self.backends[tokenizer_name] == "tiktoken":
                # tiktoken has no added/special tokens for plain encoding.
                return tokenizer.encode(text)
            # Hugging Face: explicitly exclude special tokens ([CLS], [SEP],
            # <s>, </s>, padding, etc.) so only content tokens are counted.
            return tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            logger.warning("Tokenization failed for %s: %s", tokenizer_name, exc)
            return None

    def tokenize_count(self, text: Any, tokenizer_name: str) -> int | None:
        ids = self.tokenize_ids(text, tokenizer_name)
        if ids is None:
            return None
        return len(ids)


# ----------------------------------------------------------------------
# Data loading / validation
# ----------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        logger.error("Input file not found: %s", filepath)
        sys.exit(1)

    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        logger.error("Error loading %s: %s", filepath, exc)
        sys.exit(1)

    required_cols = CONFIG["required_columns"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error("Missing required columns: %s", ", ".join(missing_cols))
        sys.exit(1)

    n_before = len(df)
    df = df.dropna(subset=required_cols).copy()
    df = df[(df[required_cols].astype(str) != "").all(axis=1)].copy()
    df = df.reset_index(drop=True)
    logger.info(
        "Loaded %d rows from %s; %d valid aligned rows after dropping missing/empty cells",
        n_before,
        filepath,
        len(df),
    )
    if len(df) == 0:
        logger.error("No valid rows remain after validation.")
        sys.exit(1)
    return df


def whitespace_words(text: Any) -> list[str]:
    if pd.isna(text):
        return []
    return str(text).split()


# ----------------------------------------------------------------------
# Metric 1: token counts per Tokenizer x Language
# ----------------------------------------------------------------------
def compute_token_counts(
    df: pd.DataFrame, registry: TokenizerRegistry
) -> dict[str, dict[str, np.ndarray]]:
    """Return {tokenizer_name: {language: np.array(content_token_counts)}}"""
    logger.info("Computing per-sentence content-token counts...")
    counts: dict[str, dict[str, np.ndarray]] = {}

    for tokenizer_name in registry.get_available():
        per_lang: dict[str, pd.Series] = {}
        for lang in CONFIG["languages"]:
            per_lang[lang] = df[lang].apply(
                lambda text: registry.tokenize_count(text, tokenizer_name)
            )

        valid_idx = np.ones(len(df), dtype=bool)
        for lang in CONFIG["languages"]:
            series = per_lang[lang]
            valid_idx &= series.notna().to_numpy() & (series.fillna(0).to_numpy() > 0)

        valid_count = int(valid_idx.sum())
        logger.info("Valid tokenizations for %s: %d/%d", tokenizer_name, valid_count, len(df))
        if valid_count < 10:
            logger.warning("Skipping %s: too few valid samples (%d)", tokenizer_name, valid_count)
            continue

        counts[tokenizer_name] = {
            lang: per_lang[lang][valid_idx].astype(float).to_numpy()
            for lang in CONFIG["languages"]
        }

    return counts


def summarize_token_counts(counts: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for tokenizer_name, per_lang in counts.items():
        for lang, arr in per_lang.items():
            rows.append(
                {
                    "Tokenizer": tokenizer_name,
                    "Language": lang,
                    "Valid_Samples": len(arr),
                    "Mean_Tokens": float(np.mean(arr)),
                    "Std_Tokens": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "Median_Tokens": float(np.median(arr)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metric 2: fertility
# ----------------------------------------------------------------------
def compute_fertility(
    df: pd.DataFrame, registry: TokenizerRegistry, counts: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    """Fertility = content tokens / whitespace-delimited words, per sentence.

    Word segmentation uses whitespace-delimited words (str.split()) for
    every language, per the experiment specification. This is a coarse
    but consistent segmentation choice.
    """
    logger.info("Computing fertility...")
    word_counts = {lang: df[lang].apply(lambda t: len(whitespace_words(t))) for lang in CONFIG["languages"]}

    rows = []
    for tokenizer_name in counts:
        # Recompute the valid mask the same way compute_token_counts did,
        # so word counts line up with the token-count arrays.
        per_lang_tok = {}
        for lang in CONFIG["languages"]:
            per_lang_tok[lang] = df[lang].apply(
                lambda text: registry.tokenize_count(text, tokenizer_name)
            )
        valid_idx = np.ones(len(df), dtype=bool)
        for lang in CONFIG["languages"]:
            series = per_lang_tok[lang]
            valid_idx &= series.notna().to_numpy() & (series.fillna(0).to_numpy() > 0)

        for lang in CONFIG["languages"]:
            tok_arr = per_lang_tok[lang][valid_idx].astype(float).to_numpy()
            wc_arr = word_counts[lang][valid_idx].astype(float).to_numpy()
            valid_words = wc_arr > 0
            tok_arr = tok_arr[valid_words]
            wc_arr = wc_arr[valid_words]
            if len(wc_arr) == 0:
                continue
            fertility = tok_arr / wc_arr
            rows.append(
                {
                    "Tokenizer": tokenizer_name,
                    "Language": lang,
                    "Mean_Fertility": float(np.mean(fertility)),
                    "Std_Fertility": float(np.std(fertility, ddof=1)) if len(fertility) > 1 else 0.0,
                    "Median_Fertility": float(np.median(fertility)),
                }
            )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metric 3: Tokenization Overhead Ratio
# ----------------------------------------------------------------------
def confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        mean = float(values[0]) if n == 1 else float("nan")
        return mean, mean
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1)) / np.sqrt(n)
    if SCIPY_AVAILABLE:
        margin = sem * scipy_stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    else:
        # normal approximation fallback
        margin = sem * 1.959963985
    return mean - margin, mean + margin


def compute_overhead_ratios(counts: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Tokenization Overhead Ratio = target-language tokens / English tokens.

    Aggregate ratio = sum(target) / sum(English) (paired, aligned rows).
    Per-sentence ratio = row-wise target/English ratio -> mean/median/std/CI.
    """
    logger.info("Computing tokenization overhead ratios...")
    rows = []
    for tokenizer_name, per_lang in counts.items():
        english = per_lang["English"]
        for target_lang in CONFIG["target_languages"]:
            target = per_lang[target_lang]
            aggregate_ratio = float(np.sum(target) / np.sum(english))
            per_sentence_ratio = target / english
            ci_lower, ci_upper = confidence_interval(per_sentence_ratio, CONFIG["confidence_level"])
            rows.append(
                {
                    "Tokenizer": tokenizer_name,
                    "Target_Language": target_lang,
                    "Aggregate_Token_Ratio": aggregate_ratio,
                    "Mean_Sentence_Ratio": float(np.mean(per_sentence_ratio)),
                    "Median_Sentence_Ratio": float(np.median(per_sentence_ratio)),
                    "Std_Sentence_Ratio": float(np.std(per_sentence_ratio, ddof=1))
                    if len(per_sentence_ratio) > 1
                    else 0.0,
                    "CI_Lower": ci_lower,
                    "CI_Upper": ci_upper,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metric 4: Tokenization Parity
# ----------------------------------------------------------------------
def compute_parity(overhead_df: pd.DataFrame) -> pd.DataFrame:
    """Tokenization Parity Ratio relative to English.

    For this aligned-sentence design, Parity_Ratio is mathematically the
    same quantity as the Aggregate_Token_Ratio computed in
    compute_overhead_ratios (total target tokens / total English tokens).
    We therefore reuse that value directly rather than recomputing an
    "independent" number, and label the relationship explicitly.
    """
    logger.info(
        "Computing tokenization parity (identical by construction to the "
        "aggregate overhead ratio for this aligned corpus)..."
    )
    rows = []
    if overhead_df.empty:
        return pd.DataFrame(columns=["Tokenizer", "Language", "Parity_Ratio"])

    for _, row in overhead_df.iterrows():
        rows.append(
            {
                "Tokenizer": row["Tokenizer"],
                "Language": row["Target_Language"],
                "Parity_Ratio": row["Aggregate_Token_Ratio"],
            }
        )
    # English vs itself, for completeness (always 1.0 by definition).
    for tokenizer_name in overhead_df["Tokenizer"].unique():
        rows.append({"Tokenizer": tokenizer_name, "Language": "English", "Parity_Ratio": 1.0})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metric 5: STRR (word-level)
# ----------------------------------------------------------------------
def compute_strr(df: pd.DataFrame, registry: TokenizerRegistry) -> pd.DataFrame:
    """Single-Token Retention Rate, computed strictly at the word level.

    For every whitespace-delimited word:
      1. tokenize the word IN ISOLATION (outside its original sentence)
      2. exclude special tokens (handled inside tokenize_ids)
      3. check whether exactly one content token represents it
      4. count as retained if so

    STRR = retained_words / total_words, per Tokenizer x Language.

    NOTE ON WHAT THIS MEASURES: because each word is re-tokenized on its
    own rather than read off the original sentence's token stream, this is
    a measure of STANDALONE single-token retention (vocabulary coverage for
    the word by itself), not of CONTEXTUAL word retention within a
    sentence. Tokenizers can split the same word differently depending on
    leading whitespace, punctuation, or neighboring characters when it
    appears in context, so STRR here should not be read as "this fraction
    of words survive as one token in real sentences" -- only as "this
    fraction of words map to one content token when tokenized alone."
    """
    logger.info("Computing STRR (word-level)...")
    rows = []
    for tokenizer_name in registry.get_available():
        for lang in CONFIG["languages"]:
            total_words = 0
            retained_words = 0
            for text in df[lang]:
                for word in whitespace_words(text):
                    ids = registry.tokenize_ids(word, tokenizer_name)
                    if ids is None:
                        continue
                    total_words += 1
                    if len(ids) == 1:
                        retained_words += 1
            if total_words == 0:
                continue
            strr = retained_words / total_words
            rows.append(
                {
                    "Tokenizer": tokenizer_name,
                    "Language": lang,
                    "STRR": strr,
                    "STRR_Percent": strr * 100.0,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Metric 6: Gini coefficient
# ----------------------------------------------------------------------
def gini_coefficient(values: np.ndarray) -> float:
    """Standard Gini coefficient over a small array of non-negative values.

    This is a purely descriptive inequality statistic over whatever raw
    values it is given -- it carries no independent efficiency notion of
    its own. See the module docstring and `compute_gini` for what this
    means when the input is raw mean token counts across languages.
    """
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("Gini coefficient requires non-negative values.")
    if np.sum(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * sorted_vals)) / (n * np.sum(sorted_vals)))


def compute_gini(token_count_summary: pd.DataFrame) -> pd.DataFrame:
    """Gini across the four language-level mean-tokens-per-aligned-sentence
    values, per tokenizer (Ahia et al. / Parity-Aware BPE methodology).

    DESCRIPTIVE INEQUALITY MEASURE, NOT AN EFFICIENCY SCORE: this operates
    on RAW mean token counts per language (from `token_count_summary`,
    metric 1), so it describes how unequally token cost is distributed
    across English/Hindi/Romanized_Hindi/Telugu for a given tokenizer. It
    does not normalize for word count, sentence length, or information
    content -- for a per-language efficiency view, see fertility
    (`compute_fertility`, metric 2) instead. A low Gini means the four
    languages cost roughly the same number of tokens on average for this
    tokenizer; it does not mean the tokenizer is efficient in an absolute
    sense (all four could be uniformly costly and still show low Gini).
    """
    logger.info("Computing Gini coefficients...")
    rows = []
    for tokenizer_name, group in token_count_summary.groupby("Tokenizer"):
        means = []
        for lang in CONFIG["languages"]:
            match = group[group["Language"] == lang]
            if len(match) == 0:
                means = None
                break
            means.append(match["Mean_Tokens"].iloc[0])
        if means is None:
            continue
        gini = gini_coefficient(np.array(means))
        rows.append({"Tokenizer": tokenizer_name, "Gini_Coefficient": gini})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Consolidated summary
# ----------------------------------------------------------------------
def build_summary_table(
    token_counts_df: pd.DataFrame,
    overhead_df: pd.DataFrame,
    strr_df: pd.DataFrame,
    gini_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    tokenizers = sorted(token_counts_df["Tokenizer"].unique())
    for tokenizer_name in tokenizers:
        english_row = token_counts_df[
            (token_counts_df["Tokenizer"] == tokenizer_name) & (token_counts_df["Language"] == "English")
        ]
        row: dict[str, Any] = {
            "Tokenizer": tokenizer_name,
            "English_Mean_Tokens": round(english_row["Mean_Tokens"].iloc[0], 2) if len(english_row) else np.nan,
        }
        for target_lang in CONFIG["target_languages"]:
            oh = overhead_df[
                (overhead_df["Tokenizer"] == tokenizer_name) & (overhead_df["Target_Language"] == target_lang)
            ]
            row[f"{target_lang}_Aggregate_Ratio"] = (
                round(oh["Aggregate_Token_Ratio"].iloc[0], 3) if len(oh) else np.nan
            )
        for lang in CONFIG["languages"]:
            s = strr_df[(strr_df["Tokenizer"] == tokenizer_name) & (strr_df["Language"] == lang)]
            row[f"{lang}_STRR_Percent"] = round(s["STRR_Percent"].iloc[0], 1) if len(s) else np.nan
        gini_row = gini_df[gini_df["Tokenizer"] == tokenizer_name]
        row["Gini_Coefficient"] = round(gini_row["Gini_Coefficient"].iloc[0], 4) if len(gini_row) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def save_figure(fig: plt.Figure, output_dir: str, filename: str, dpi: int) -> None:
    plt.tight_layout()
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    logger.info("Saved %s", filepath)
    plt.close(fig)


def fig_token_heatmap(token_counts_df: pd.DataFrame, output_dir: str, dpi: int) -> None:
    pivot = token_counts_df.pivot(index="Tokenizer", columns="Language", values="Mean_Tokens")
    pivot = pivot[CONFIG["languages"]]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Mean content tokens/sentence"})
    ax.set_title("Average Tokens by Tokenizer and Language", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "fig1_avg_tokens_heatmap.png", dpi)


def fig_fertility(fertility_df: pd.DataFrame, output_dir: str, dpi: int) -> None:
    pivot = fertility_df.pivot(index="Tokenizer", columns="Language", values="Mean_Fertility")
    pivot = pivot[[c for c in CONFIG["languages"] if c in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_ylabel("Fertility (content tokens / whitespace word)", fontsize=11, fontweight="bold")
    ax.set_title("Fertility by Tokenizer and Language", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Language")
    save_figure(ax.get_figure(), output_dir, "fig2_fertility.png", dpi)


def fig_overhead(overhead_df: pd.DataFrame, output_dir: str, dpi: int) -> None:
    pivot = overhead_df.pivot(index="Tokenizer", columns="Target_Language", values="Aggregate_Token_Ratio")
    pivot = pivot[[c for c in CONFIG["target_languages"] if c in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.axhline(y=1.0, color="green", linestyle="--", linewidth=2, label="Parity (1.0x)")
    ax.set_ylabel("Tokenization Overhead / Parity Ratio (target / English)", fontsize=11, fontweight="bold")
    ax.set_title("Tokenization Overhead & Parity Ratios (aggregate)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Target language")
    save_figure(ax.get_figure(), output_dir, "fig3_overhead_parity_ratio.png", dpi)


def fig_strr(strr_df: pd.DataFrame, output_dir: str, dpi: int) -> None:
    pivot = strr_df.pivot(index="Tokenizer", columns="Language", values="STRR_Percent")
    pivot = pivot[[c for c in CONFIG["languages"] if c in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_ylabel("STRR (%)", fontsize=11, fontweight="bold")
    ax.set_title("Single-Token Retention Rate by Tokenizer and Language", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Language")
    save_figure(ax.get_figure(), output_dir, "fig4_strr.png", dpi)


def fig_gini(gini_df: pd.DataFrame, output_dir: str, dpi: int) -> None:
    sorted_df = gini_df.sort_values("Gini_Coefficient")
    fig, ax = plt.subplots(figsize=(9, max(4, 0.6 * len(sorted_df))))
    colors = ["#2ECC71" if g < 0.1 else "#F39C12" if g < 0.2 else "#E74C3C" for g in sorted_df["Gini_Coefficient"]]
    ax.barh(sorted_df["Tokenizer"], sorted_df["Gini_Coefficient"], color=colors, edgecolor="black", alpha=0.9)
    ax.set_xlabel("Gini Coefficient (lower = more equal across languages)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Language Tokenization Inequality (Gini)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, (tok, g) in enumerate(zip(sorted_df["Tokenizer"], sorted_df["Gini_Coefficient"])):
        ax.text(g + 0.002, i, f"{g:.3f}", va="center", fontsize=9)
    save_figure(fig, output_dir, "fig5_gini.png", dpi)


# ----------------------------------------------------------------------
# Printed summary
# ----------------------------------------------------------------------
def print_experiment_summary(
    n_rows: int,
    token_counts_df: pd.DataFrame,
    overhead_df: pd.DataFrame,
    gini_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 78)
    print("MULTI-TOKENIZER ANALYSIS - EXPERIMENT SUMMARY")
    print("=" * 78)
    print(f"Valid aligned rows analyzed: {n_rows}")
    print(
        "Token-counting method: content tokens only "
        "(tiktoken plain encode(); HF add_special_tokens=False)"
    )
    print("\nAggregate Tokenization Overhead / Parity Ratio (target tokens / English tokens):")
    for tokenizer_name in sorted(overhead_df["Tokenizer"].unique()):
        sub = overhead_df[overhead_df["Tokenizer"] == tokenizer_name]
        parts = [
            f"{row['Target_Language']}={row['Aggregate_Token_Ratio']:.2f}x"
            for _, row in sub.iterrows()
        ]
        print(f"  {tokenizer_name}: " + ", ".join(parts))
    print(
        "\nGini coefficient (raw token-cost inequality, NOT an efficiency score -- "
        "see fertility for per-language efficiency) across "
        "(English, Hindi, Romanized_Hindi, Telugu) mean tokens/sentence:"
    )
    for _, row in gini_df.sort_values("Gini_Coefficient").iterrows():
        print(f"  {row['Tokenizer']}: {row['Gini_Coefficient']:.4f}")
    print("=" * 78)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-tokenizer cross-lingual tokenization efficiency/fairness analysis."
    )
    parser.add_argument("--input", default=CONFIG["input_file"], help="Path to multilingual_dataset_*.csv")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"], help="Directory for output CSVs/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    CONFIG["output_dir"] = args.output_dir

    logger.info("=" * 70)
    logger.info("MULTI-TOKENIZER ANALYSIS - START")
    logger.info("=" * 70)

    registry = TokenizerRegistry()
    available = registry.get_available()
    if not available:
        logger.error("No tokenizers loaded. Install tiktoken and/or transformers.")
        sys.exit(1)
    logger.info("Loaded %d tokenizers", len(available))
    for name in available:
        print(f"  - {name}")

    df = load_data(args.input)

    token_counts = compute_token_counts(df, registry)
    if not token_counts:
        logger.error("No tokenizer produced enough valid tokenizations. Aborting.")
        sys.exit(1)

    token_counts_df = summarize_token_counts(token_counts)
    fertility_df = compute_fertility(df, registry, token_counts)
    overhead_df = compute_overhead_ratios(token_counts)
    parity_df = compute_parity(overhead_df)
    strr_df = compute_strr(df, registry)
    gini_df = compute_gini(token_counts_df)
    summary_df = build_summary_table(token_counts_df, overhead_df, strr_df, gini_df)

    out_dir = args.output_dir
    token_counts_df.to_csv(os.path.join(out_dir, "tokenizer_language_token_counts.csv"), index=False)
    fertility_df.to_csv(os.path.join(out_dir, "tokenizer_fertility.csv"), index=False)
    overhead_df.to_csv(os.path.join(out_dir, "tokenizer_overhead.csv"), index=False)
    parity_df.to_csv(os.path.join(out_dir, "tokenizer_parity.csv"), index=False)
    strr_df.to_csv(os.path.join(out_dir, "tokenizer_strr.csv"), index=False)
    gini_df.to_csv(os.path.join(out_dir, "tokenizer_gini.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "tokenizer_summary.csv"), index=False)
    logger.info("Saved all CSV outputs to %s", out_dir)

    fig_token_heatmap(token_counts_df, out_dir, CONFIG["dpi"])
    fig_fertility(fertility_df, out_dir, CONFIG["dpi"])
    fig_overhead(overhead_df, out_dir, CONFIG["dpi"])
    fig_strr(strr_df, out_dir, CONFIG["dpi"])
    fig_gini(gini_df, out_dir, CONFIG["dpi"])

    print_experiment_summary(len(df), token_counts_df, overhead_df, gini_df)
    logger.info("Analysis complete. Outputs saved to %s", out_dir)


if __name__ == "__main__":
    main()
