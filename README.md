# Multilingual Tokenization Bias and Fairness

A comparative study of tokenization efficiency, cross-language token-cost disparity, and tokenization parity across **English, Hindi, Romanized Hindi, and Telugu** using multiple tokenizer families.

---

## Overview

Large language models process text through tokenizers that convert natural-language input into discrete tokens. The number and structure of tokens produced for the same underlying content can vary substantially across languages and tokenizer vocabularies.

This project investigates whether different tokenizer designs produce unequal tokenization costs across languages, with a particular focus on Indian languages and Romanized Hindi.

The study compares:

- **English**
- **Hindi**
- **Romanized Hindi**
- **Telugu**

across five tokenizer configurations:

- GPT-3 (`p50k_base`)
- GPT-4 / GPT-3.5 (`cl100k_base`)
- IndicBERT
- XLM-RoBERTa
- mBERT

The central research question is:

> **How do different tokenizer designs affect tokenization cost and cross-language tokenization parity for English, Hindi, Romanized Hindi, and Telugu?**

---

## Research Motivation

Tokenization is an important intermediate stage in language-model processing. A sentence that conveys the same information can require substantially different numbers of tokens depending on the language and tokenizer vocabulary.

Higher token counts can represent greater token fragmentation and higher token-processing overhead. Consequently, systematic differences in token counts across languages can create disparities in how efficiently different languages are represented by the same tokenizer.

This project examines these disparities quantitatively rather than evaluating language-model quality itself.

The study specifically investigates:

1. Differences in token counts across languages.
2. Token fragmentation through fertility.
3. Relative token cost compared with English.
4. Word-level preservation through Single-Token Retention Rate (STRR).
5. Cross-language inequality using the Gini coefficient.
6. Whether Romanization changes the tokenization cost of Hindi.
7. Differences between English-oriented and multilingual tokenizer configurations.

---

## Dataset

The final experimental dataset contains **1,000 aligned multilingual records**.

Each record contains four language representations:

| Field | Description |
|---|---|
| `English` | Original English sentence |
| `Hindi` | Hindi representation |
| `Romanized_Hindi` | Romanized Hindi representation |
| `Telugu` | Telugu representation |

The dataset is stored at:

```text
generated_dataset/multilingual_dataset_1000.csv
