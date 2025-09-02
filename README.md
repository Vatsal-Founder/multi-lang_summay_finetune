

> **Multilingual abstractive summarizer** fine‑tuned from **`google/mt5-base`**. Trained on custom data with `text → summary` pairs; intended for short/medium summaries across multiple languages.

<p align="center">
  <a href="https://huggingface.co/vatsal18/multi-lang_summay"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue" alt="HF model" /></a>
  <a href="#quickstart"><img src="https://img.shields.io/badge/Transformers-≥4.41-ff69b4" alt="Transformers" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License" /></a>
</p>

---

## Contents

* [Overview](#overview)
* [Quickstart](#quickstart)
* [Demo (Streamlit)](#demo-streamlit)
* [Results](#results)
* [Training Details](#training-details)
* [Data & Features](#data--features)
* [Inference Tips](#inference-tips)
* [Limitations & Safety](#limitations--safety)
* [License](#license)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

---

## Overview

**`vatsal18/multi-lang_summay`** is a sequence‑to‑sequence model for **multilingual abstractive summarization**, fine‑tuned from `google/mt5-base`. It works best on news/articles and dialog‑like text and can handle multiple languages (coverage inherited from mT5).

**Task**: `text` → `summary`

**Intended use**: generate concise summaries (a few sentences) suitable for previews, notes, or headline‑style abstracts.

---

## Quickstart

> Requirements: `transformers`, `torch`, `sentencepiece`, `safetensors`

```bash
pip install -U transformers sentencepiece safetensors torch
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

repo_id = "vatsal18/multi-lang_summay"
device = ("cuda" if torch.cuda.is_available() else
          "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
          "cpu")

tok = AutoTokenizer.from_pretrained(repo_id)
mdl = AutoModelForSeq2SeqLM.from_pretrained(repo_id).to(device).eval()

text = """Paste an article / dialogue here (any supported language)…"""
enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
with torch.no_grad():
    out = mdl.generate(
        **enc,
        max_new_tokens=128, min_new_tokens=32,
        num_beams=5, no_repeat_ngram_size=3,
        length_penalty=1.0, do_sample=False,
    )
summary = tok.decode(out[0], skip_special_tokens=True)
print(summary)
```

**Transformers pipeline**

```python
from transformers import pipeline
summarizer = pipeline("summarization", model=repo_id, tokenizer=repo_id)
res = summarizer("Your text here", max_new_tokens=128, num_beams=5, do_sample=False)[0]
print(res["summary_text"])
```

---

## Demo (Streamlit)

Minimal local demo UI:

```bash
pip install streamlit transformers sentencepiece safetensors torch
```

```python
# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

repo_id = "vatsal18/multi-lang_summay"

tok = AutoTokenizer.from_pretrained(repo_id)
mdl = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

st.title("📝 Multilingual Summarizer")
text = st.text_area("Paste text", height=200)
length = st.selectbox("Length", ["short","medium","long"], index=1)
max_new = {"short":64, "medium":128, "long":256}[length]

if st.button("Summarize") and text.strip():
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
              "cpu")
    mdl.to(device).eval()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = mdl.generate(**enc, max_new_tokens=max_new, min_new_tokens=max(16, max_new//4),
                           num_beams=5, no_repeat_ngram_size=3, length_penalty=1.0, do_sample=False)
    st.success(tok.decode(out[0], skip_special_tokens=True))
```

Run: `streamlit run app.py`

---

## Results

Evaluation on a small slice of the test set (`n=10`):

| Metric     |  Score |
| ---------- | -----: |
| ROUGE‑1    | 0.2236 |
| ROUGE‑2    | 0.0543 |
| ROUGE‑L    | 0.1363 |
| ROUGE‑Lsum | 0.1364 |

> Note: scores on only 10 items are noisy. For a stable estimate, please evaluate on the full test split and per‑language subsets.

---

## Training Details

* **Base model**: `google/mt5-base`
* **Task**: abstractive summarization (`text → summary`)
* **Optimizer**: `adamw_torch` (or `adafactor` on memory‑constrained devices)
* **Batching**: `per_device_train_batch_size=1` with `gradient_accumulation_steps` to reach effective batch ≥ 16
* **Sequence lengths**: input up to 1024 tokens; targets up to 128 tokens
* **Memory savers**: gradient checkpointing (`model.config.use_cache = False`)

**Example `TrainingArguments`**

```python
from transformers import TrainingArguments
trainer_args = TrainingArguments(
    output_dir="model_mtf-final",
    num_train_epochs=2,                  
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,       
    eval_strategy="steps",          
    eval_steps=450,                       
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    logging_steps=25,
    warmup_steps=200,                    
    weight_decay=0.01,
    optim="adafactor",                    
    dataloader_pin_memory=False,     
    gradient_checkpointing=True,          
    report_to="none"
)
# when using gradient checkpointing:
# model.config.use_cache = False
```

---

## Data & Features

The fine‑tuning dataset used columns like:

```
['id', 'url', 'title', 'summary', 'text', 'lang', 'input_ids', 'attention_mask', 'labels']
```

Training consumed `text` (input) and `summary` (target). Pre‑tokenized columns were used internally; for inference you only need raw `text`.

---

## Inference Tips

* Prefer `max_new_tokens` / `min_new_tokens` for clear length control.
* Useful decoding settings: `num_beams=5–8`, `no_repeat_ngram_size=3`, `length_penalty≈1.0`.
* For very long articles, truncate to `max_length=1024` tokens.
* Apple Silicon users: set device to `mps`; avoid `fp16=True` in `TrainingArguments` (CUDA‑only).

---



## Limitations & Safety

* **Factuality**: like most abstractive models, it can hallucinate. Verify facts for critical use.
* **Language variability**: quality differs by language/domain; consider light domain adaptation for best results.
* **Bias**: may reflect biases present in pretraining and fine‑tuning data.

---

## License

Apache‑2.0 (see `LICENSE`). 

---

## Citation

If you use this model, please cite the repository and the base model:

```bibtex
@misc{vatsal18_multilang_summay,
  title  = {Multilingual Abstractive Summarizer (mT5-base fine-tune)},
  author = {Vatsal},
  year   = {2025},
  url    = {https://huggingface.co/vatsal18/multi-lang_summay}
}
```

---

## Acknowledgements

* Base model: **[google/mt5-base](https://huggingface.co/google/mt5-base)**
* Libraries: 🤗 Transformers, Datasets, Evaluate, PyTorch
