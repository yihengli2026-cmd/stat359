# Style Shift via Pirate Patch Fine-Tuning

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

### Step 1: Train Base TinyStories Model

- `python train_bpe_tokenizer_hf.py`
- `python train_tinystories_model.py`
- `python train_tinystories_chat_model.py`

Output: 
- `tinystories_model/`
- `tinystories_chat_model/` 
- `bpe_tokenizer_tinystories.pkl`

### Step 2: Make the pirate patch and Fine-tune the pirate chat model

- `python .\scripts\make_pirate_patch.py` 

Output:
`pirate_patch_6000.jsonl`

- `python train_tinystories_chat_model.py \
  --patch_dataset pirate_patch_6000.jsonl \
  --output_dir tinystories_chat_model_pirate_6000_balanced`

Output: `tinystories_chat_model_pirate_6000_balanced/`

### Step 3: Generate Model Outputs (Evaluation)

- `python prompt_tests.py --model_path tinystories_chat_model/final_model.pth --output_path results_base.json`
- `python prompt_tests.py --model_path tinystories_chat_model_pirate_6000_balanced/final_model.pth --output_path results_pirate.json`
  
Output:
- `results_base.json`
- `results_pirate.json`

### Step 4: Build Pirate Reference Texts
`python extract_pirate_refs.py`

Output: `pirate_refs.txt`

### Step 5: Score Outputs (Style + Cosine)
- Style Scoring
  
  - `python score_results.py --input results_base.json --out_csv base_scored.csv`
  - `python score_results.py --input results_pirate.json --out_csv pirate_scored.csv`

   Outputs:
  - `results_base_scored.json`

  - ` results_pirate_scored.json`

- Prepare cosine inputs:
  - `python -c "import pandas as pd; df=pd.read_csv('base_scored.csv'); df['output']=df['response']; df.to_csv('base_for_cosine.csv', index=False)"`
  - `python -c "import pandas as pd; df=pd.read_csv('pirate_scored.csv'); df['output']=df['response']; df.to_csv('pirate_for_cosine.csv', index=False)"`

### Step 6: Compute, Analyze, and Plot Cosine Similarity
- Compute Cosine Similarity
  - ` python cosine_style_score.py --input_csv base_for_cosine.csv --ref_file pirate_refs.txt --out_csv base_cosine.csv --device cuda`
  - `python cosine_style_score.py --input_csv pirate_for_cosine.csv --ref_file pirate_refs.txt --out_csv pirate_cosine.csv --device cuda`
- Analyze Results
  - `python analyze_scores.py --baseline base_scored.csv --patched pirate_scored.csv`
  - `python analyze_cosine.py --baseline base_cosine.csv --patched pirate_cosine.csv`
- Plot results
  - `python plot_cosine.py --baseline base_cosine.csv --pirate pirate_cosine.csv --out_prefix cosine`


### Step 7: Train the BERT style classifier

Run `python train_pirate_bert.py`.

Purpose: This script trains a binary text classifier on `datasets/combined_pirate_dataset.json`.

Main output: 

Bert Classifier Model:
- `best_model.pt`
- `tokenizer_config.json`
- `tokenizer.json`

### Step 8: Compute Bert Probability Score/Plot BERT classifier results

-Run `python score_bert.py`.

  - Purpose: This script loads the trained BERT classifier and applies it to generated responses.

  - Output: `bert_base_scored.json` and `bert_pirate_scored.json`

-Run `python plot_bert.py`.

 - Purpose: This script visualizes classifier-based style results.
 - Output: 
   - `bert_bar_clean.png`
  - `bert_box.png`
  - `bert_hist.pg`





---

## Output Management

The main outputs generated by this project are:

- `results_base.json`  
  Raw generations from stress tests of Based Model

- `results_pirate.json`
  Raw generations from stress tests of Patched Model

-Scoring outputs
 - Cos scoring outputs
   `out_csv base_cosine.csv` and `out_csv pirate_cosine.csv`
 - BERT scoring outputs  
   `bert_base_scored.json` and `bert_pirate_scored.json`

- Plot files  
  Produced by `plot_cosine.py` and `plot_bert.py`

- Model folders  
  Produced by `train_pirate_bert.py` and patched model training

Depending on the script configuration, files may be saved in the project root or in dedicated results/model folders.

---

## Individual Contributions

Replace these placeholders with your actual names and contributions.

- **[Yiheng Li]**: proposed the project topic; designed the stress-test prompts and templates; trained the BERT classifier; analyzed results; wrote the limitations section
- **[Yu Wang]**: Constructed the pirate-style patch dataset, performed model fine-tuning, and implemented cosine similarity evaluation to measure stylistic alignment.

---

## References

- Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.
- Bochen0909. (n.d.). tinystories-conversations (Hugging Face dataset). Retrieved March 2026, from https://huggingface.co/datasets/bochen0909/tinystories-conversations
- GPT007. (n.d.). Pirate speak (Hugging Face dataset). Retrieved March 2026, from https://huggingface.co/datasets/GPT007/Pirate%20speak
- KafeisM. (n.d.). pirate-speak-dataset (Hugging Face dataset). Retrieved March 2026, from https://huggingface.co/datasets/KafeisM/pirate-speak-dataset

### Datasets

- `bochen0909/tinystories-conversations`
- `KafeisM/pirate-speak-dataset`
- `GPT007/Pirate speak`


