🛡️ Hybrid AI Fake News Detector (BERT + RAG)
Developed by: Dhanvith Shetty | Internship: Festiva Moments | College: VCET

## 🖥️ Application Demo

![App Demo](assets/App%20demo.png)

![App Demo 1](assets/App%20demo1.png)


📌 Problem Statement
Static NLP models like BERT are limited by their Training Data Cutoff. A model trained in 2017 cannot verify news about the 2026 T20 World Cup or current 2026 market crashes. This "Temporal Drift" leads to false negatives where the AI flags real, modern news as "Fake" simply because it hasn't seen the facts before.

🚀 The Approach
This project implements a Hybrid Verification Pipeline:

Linguistic Layer: A fine-tuned DistilBERT model analyzes the "fingerprint" of the text (sensationalism, bias, formal vs. informal).

Factual Layer (RAG): If BERT is unsure or flags a modern event, the system uses Retrieval-Augmented Generation via the Tavily Search API to cross-reference live 2026 news databases.

Smart Verdict: A decision-logic layer that prioritizes real-time evidence over stylistic patterns.

🤖 Model Used
Base Model: distilbert-base-uncased

Dataset: ISOT Fake News Dataset (44,000+ articles)

Fine-tuning: 3 Epochs on NVIDIA RTX 2050 (Local GPU)

API: Tavily AI (Real-time search)

📊 Metrics
Accuracy: 99.98%

F1-Score: 1.00

Precision/Recall: 1.00

💡 Key Improvements
Beyond Classification: Integrated a live search API to fix the 2017 knowledge cutoff.

Hardware Acceleration: Configured local CUDA environments for high-speed inference.

Interactive UI: Built a Streamlit dashboard with side-by-side analysis and factual evidence expanders.

🧠 Key Learnings
Data Bias: I learned that high accuracy on old datasets (ISOT) doesn't guarantee real-world performance because language and facts evolve.

The RAG Advantage: Hybrid models are significantly more reliable for news detection than standalone classifiers.
### 🛠️ Installation & Usage
1. **Environment:** Setup a Python 3.10+ environment.
2. **Dependencies:** Install via `pip install -r requirements.txt`.
3. **API Key:** Obtain a key from [Tavily](https://tavily.com/) and add it to `src/app.py`.
4. **Execution:** Launch the interface using `python -m streamlit run src/app.py`.

### 📂 Repository Structure
* `src/`: Core application logic (Streamlit + BERT Inference).
* `notebook/`: Google Colab/Jupyter training pipeline.
* `fine_tuned_bert_fake_news/`: Saved model weights and tokenizer configuration.
* `assets/`: Performance graphs and UI screenshots.
