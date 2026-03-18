# TinyStories Persona Patching (Pirate Style)

This repository contains our final project for an LLM course. We study whether a small instruction-tuned TinyStories chat model can be patched into a stable pirate persona using targeted continued fine-tuning, and whether that style persists under different prompting conditions.

Our project has three main parts:

1. Stress testing the model with three prompt sets
2. Scoring style with cosine similarity and a BERT classifier
3. Patching the model with pirate-style data and comparing before vs. after

---

## Project Goals

The goals of this project are to:

1. Train or use a TinyStories instruction-tuned chat model as a baseline
2. Design prompt stress tests that probe whether pirate style holds under different interaction settings
3. Patch the baseline model with a targeted pirate-style dataset
4. Compare the baseline and patched models using automatic style metrics
5. Analyze whether patching improves prompt-conditioned behavior or mainly shifts the model’s overall style prior

---

## Motivation

Persona patching is a practical test of controllable generation in small language models. If a targeted patch works well, the model should adopt a desired style when appropriate and remain robust even when prompts become longer, multi-turn, or distracting.

We chose a pirate-speaking persona because it is easy to recognize by eye and also easy to evaluate with automatic metrics. This makes it a useful setting for studying whether fine-tuning changes:

- the model’s global stylistic tendencies
- its ability to condition style on prompt context

---

## Main Result

Patching strongly increases pirate-style scores and also tends to make responses longer. However, the scores are very similar across all three prompt stress-test sets, suggesting that the patch mainly changes the model’s global style prior more than its prompt-conditioned instruction following.

---

## Repository Structure

- `datasets/`
  - `prompt_sets/`
    - `single_prompts.json`
    - `multi_turn_conversations.json`
    - `distractor_prompts.json`
  - `combined_pirate_dataset.json`
  - `pirate_refs.txt`
- `prompt_tests.py`
- `pirate_score_json.py`
- `cosine_style_score.py`
- `analyze_cosine.py`
- `analyze_scores.py`
- `plot_cosine.py`
- `train_pirate_bert.py`
- `score_bert.py`
- `plot_bert.py`
- `make_pirate_patch.py`
- `extract_pirate_refs.py`
- course-provided training/model scripts

---

## What Is In This Repository

### Datasets

This project uses three prompt stress-test sets and two pirate-style reference datasets.

- `datasets/prompt_sets/single_prompts.json`  
  30 single-turn prompts for testing pirate style in simple one-shot interactions.

- `datasets/prompt_sets/multi_turn_conversations.json`  
  30 multi-turn conversations for testing whether pirate style persists across dialogue context.

- `datasets/prompt_sets/distractor_prompts.json`  
  30 distractor or conflicting prompts for testing whether pirate style survives prompts that weaken or redirect the persona.

- `datasets/pirate_refs.txt`  
  Pirate reference sentences used as style anchors for embedding cosine similarity.

- `datasets/combined_pirate_dataset.json`  
  Labeled examples used to train the BERT classifier that predicts whether text is pirate style.

### Main Scripts

- `prompt_tests.py`  
  Runs the model on all prompt sets and saves generated responses.

- `pirate_score_json.py`  
  A lightweight keyword-based pirate scoring script used as a quick sanity check.

- `cosine_style_score.py`  
  Computes embedding cosine similarity between generated responses and pirate reference texts.

- `analyze_cosine.py`  
  Summarizes cosine similarity scores by prompt category.

- `analyze_scores.py`  
  Inspects and aggregates results from scored JSON or CSV outputs.

- `plot_cosine.py`  
  Produces plots for cosine-based results.

- `train_pirate_bert.py`  
  Trains a BERT classifier to estimate whether a text is written in pirate style.

- `score_bert.py`  
  Uses the trained BERT classifier to score generated outputs.

- `plot_bert.py`  
  Visualizes BERT classifier scores.

- `make_pirate_patch.py`  
  Builds a pirate patch dataset for continued fine-tuning.

- `extract_pirate_refs.py`  
  Updates or extracts pirate reference text and related style data.

---

## Environment Setup

We provide a `pyproject.toml` file for reproducible installation.

### Option A: Poetry

Run `poetry install`, then run `poetry shell`.

### Option B: pip

Install the main dependencies with `pip install torch datasets transformers sentence-transformers scikit-learn pandas matplotlib`.

### Reproducibility Notes

For stronger reproducibility, package versions should be pinned in `pyproject.toml`, `requirements.txt`, or `environment.yml`.

Model checkpoint files are not committed to GitHub because `.pth` files are too large for the repository.

Expected local checkpoint paths:

- Baseline chat model: `tinystories_chat_model/best_model.pth`
- Patched chat model: `tinystories_chat_model_patched/best_model.pth`

---

## How To Use This Repository

There are two common ways to use this repo:

1. Run the full evaluation pipeline
2. Run only one part, such as scoring or plotting

A normal workflow is:

1. Generate outputs with the model
2. Score those outputs
3. Summarize the scores
4. Plot the results
5. Patch the model
6. Re-run the same evaluation and compare baseline vs patched

---

## Quick Start

If you want the main end-to-end workflow, run these commands in order:

- `python prompt_tests.py`
- `python cosine_style_score.py`
- `python analyze_cosine.py`
- `python plot_cosine.py`
- `python train_pirate_bert.py`
- `python score_bert.py`
- `python plot_bert.py`

What this does:

- `prompt_tests.py` generates model responses for all stress-test prompts
- `cosine_style_score.py` measures how close those responses are to pirate reference text
- `analyze_cosine.py` summarizes the cosine results
- `plot_cosine.py` visualizes the cosine results
- `train_pirate_bert.py` trains the style classifier
- `score_bert.py` applies the classifier to generated outputs
- `plot_bert.py` visualizes classifier-based results

---

## Detailed Step-by-Step Reproduction

### Step 1: Run the prompt stress tests

Run `python prompt_tests.py`.

Purpose:

This script loads the selected model checkpoint and runs it on the three prompt sets:

- single-turn prompts
- multi-turn conversations
- distractor prompts

What to check before running:

- Make sure the checkpoint path inside `prompt_tests.py` points to the correct model
- Decide whether you are running the baseline or the patched model

What it produces:

- `results_all_tests.json`

This file stores the raw generated responses and is the starting point for later scoring scripts.

### Step 2: Compute cosine similarity style scores

Run `python cosine_style_score.py`.

Purpose:

This script compares generated responses to pirate reference text in `datasets/pirate_refs.txt` using embedding cosine similarity.

Interpretation:

Higher cosine similarity means the generated response is closer in style to the pirate reference text.

Typical outputs:

- `results_all_tests_scored.csv` for per-example cosine scores
- `results_all_tests_summary.csv` for average cosine scores grouped by category

Use this step when you want a semantic style similarity measure rather than a simple keyword count.

### Step 3: Inspect and summarize the cosine results

Run `python analyze_cosine.py` and `python analyze_scores.py`.

Purpose:

These scripts help you inspect the outputs from the scoring step.

Use them to answer questions like:

- Which prompt category has the highest average pirate score?
- Are baseline and patched results clearly separated?
- Are there categories where the patch has less effect?
- Are there outliers or unusual examples?

This stage is useful before plotting because it gives a cleaner summary of what changed.

### Step 4: Plot cosine-based results

Run `python plot_cosine.py`.

Purpose:

This script turns the cosine summary data into figures.

Use this step when you want visuals for reports, presentations, the final paper, or quick comparison between prompt categories.

The exact file names and save locations depend on how the plotting script is configured.

### Step 5: Train the BERT style classifier

Run `python train_pirate_bert.py`.

Purpose:

This script trains a binary text classifier on `datasets/combined_pirate_dataset.json`.

The classifier learns to estimate whether a piece of text is pirate style.

Why this matters:

Cosine similarity measures closeness to pirate reference text, but the BERT classifier gives a second, different style signal: `P(pirate_style | text)`.

This helps avoid relying on only one metric.

Typical output:

- a saved classifier model directory or checkpoint folder

The exact output folder depends on the script configuration.

### Step 6: Score generated outputs with the BERT classifier

Run `python score_bert.py`.

Purpose:

This script loads the trained BERT classifier and applies it to generated responses.

Interpretation:

For each generated response, the model outputs a pirate-style probability. Higher values mean the classifier is more confident that the text is pirate style.

Use this step to compare:

- baseline vs patched generations
- different prompt categories
- consistency across test conditions

### Step 7: Plot BERT classifier results

Run `python plot_bert.py`.

Purpose:

This script visualizes classifier-based style results.

Typical use cases:

- compare probability distributions
- show category-level differences
- include plots in a report or presentation

### Step 8: Create the pirate patch dataset

Run `python make_pirate_patch.py`.

Purpose:

This script builds a targeted pirate-style patch dataset for continued fine-tuning.

Typical output:

- `pirate_patch.jsonl`

The actual file name may vary depending on the script.

Use this step before continuing training from the baseline checkpoint.

### Step 9: Fine-tune the chat model with the pirate patch

Using the course-provided training script, continue training from the baseline checkpoint with `python train_tinystories_chat_model.py --pretrained_model_path tinystories_chat_model/best_model.pth --dataset_path pirate_patch.jsonl --output_dir tinystories_chat_model_patched`.

Purpose:

This creates the patched version of the chat model.

After this step:

- switch your checkpoint path to the patched model
- re-run Steps 1 through 7
- compare patched results against baseline results

This before-vs-after comparison is the core experimental result of the project.



