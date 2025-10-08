# CAST: Multistage Feedback-Driven Causal Discovery from Textual Data with Large Language Models

> **Anonymous Submission for Double-Blind Review — WWW 2026**  
> Implementation accompanying the paper:  
> *“Multistage Feedback-Driven Causal Discovery from Textual Data with Large Language Models (CAST)”*

---

## 🌐 Overview
**CAST** is a multistage framework that discovers **causal factors and structures** directly from textual data using large language models (LLMs).  

---

## ⚙️ Setup
Python = 3.12  
Install dependencies:
```bash
pip install -r requirements.txt
```

CAST supports local or API-based LLMs (e.g., DeepSeek, Qwen, GPT):
```bash
ollama pull deepseek-r1:8b
```

---

## 🚀 Run
Execute the main pipeline:
```bash
python CAST_AG_LLM_mian.py
```

---

## 📊 Datasets
Included benchmark datasets:
- `Apple_Gastronome_AG7_v20240513.xlsx` — synthetic product reviews  
- `Lung_Diseaser_Causal_Dataset.xlsx` — synthetic medical texts  
- `watermelon_data.xlsx` — real-world user reviews  

Each contains a `Review` column and a numeric `score` column.

---


## ⚖️ License
```
Copyright (c) 2025 Anonymous Authors
For double-blind review only.
Licensed under the Apache License, Version 2.0.
```

---
