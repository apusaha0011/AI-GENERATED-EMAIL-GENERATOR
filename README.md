# AI-GENERATED-EMAIL-GENERATOR 📧

A production-quality AI-powered email generation and evaluation system built as an AI Engineer candidate assessment.  
Generates professional business emails using advanced prompt engineering, then evaluates output quality using 3 custom metrics across two LLMs.

---

## 📋 Key Deliverables

This repository uses a single authoritative README (this file). All task deliverables, usage instructions, and summaries are consolidated here.

Key artifacts in the repo:

- **Evaluation outputs:** `data/results/` (JSON/CSV reports and `analysis_summary.md`)
- **Prompt template:** `prompts/email_prompt_template.txt`
- **Streamlit UI:** `streamlit_app.py`
- **Source code:** `src/` (generator, evaluator, models)
- **Tests:** `tests/` (pytest)

---

## Table of Contents

1. [Quick Start](#quick-start)  
2. [Project Structure](#project-structure)  
3. [Advanced Prompt Engineering](#advanced-prompt-engineering)  
4. [Custom Evaluation Metrics](#custom-evaluation-metrics)  
5. [Model Comparison](#model-comparison)  
6. [Running the Evaluation](#running-the-evaluation)  
7. [Output Files](#output-files)  
8. [Running Unit Tests](#running-unit-tests)  
9. [Dependencies](#dependencies)

---

## Project Structure

```
email gen/
├── README.md                          # This file (single-authoritative documentation)
├── src/
│   ├── __init__.py
│   ├── models.py                      # Unified OpenAI + xAI Grok client wrappers
│   ├── generator.py                   # Email generation with advanced prompting
│   └── evaluator.py                   # 3 custom metric implementations
├── data/
│   ├── test_scenarios.json            # 10 test scenarios with human reference emails
│   └── results/                       # Auto-generated evaluation reports
│       ├── evaluation_report_<timestamp>.json
│       ├── evaluation_report_<timestamp>.csv
│       ├── evaluation_report_latest.json
│       ├── evaluation_report_latest.csv
│       └── analysis_summary.md        # Comparative model analysis (one-page summary)
├── prompts/
│   └── email_prompt_template.txt      # Fully documented master prompt template
├── tests/
│   └── test_generator.py              # Unit tests (pytest)
├── streamlit_app.py                   # Streamlit UI for interactive email generation
├── run_evaluation.py                  # Main pipeline entrypoint
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
└── venv/                              # Virtual environment (created locally)
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "email gen"
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
copy .env.example .env       # Windows
# or
cp .env.example .env         # macOS/Linux
```

Edit `.env` and add your keys:

```env
OPENAI_API_KEY=sk-your-openai-key-here
XAI_API_KEY=xai-your-grok-key-here
```

> **Where to get keys:**  
> - OpenAI: https://platform.openai.com/api-keys  
> - xAI (Grok): https://console.x.ai/

### 5. Run the evaluation

```bash
python run_evaluation.py
```

### 6. Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed by Streamlit and enter your Intent, Tone, and Key Facts.

---

## Advanced Prompt Engineering

**Technique: Chain-of-Thought (CoT) + Few-Shot Examples + Role-Playing**

The master prompt template (`prompts/email_prompt_template.txt`) combines three techniques:

| Technique | How It's Applied | Why It Matters |
|-----------|-----------------|----------------|
| **Role-Playing** | System prompt assigns the model a persona: *"senior professional communications expert with 15+ years of experience"* | Grounds the model in the correct behavioral frame and vocabulary register |
| **Few-Shot Examples** | Two complete worked examples are included (demo follow-up email, SLA escalation), each showing the reasoning steps and the final email | Demonstrates the exact output format and reasoning pattern expected |
| **Chain-of-Thought** | Before writing the email, the model must reason through 5 explicit steps: Intent Analysis → Fact Mapping → Tone Calibration → Structure Plan → Draft | Forces deliberate reasoning before output; reduces fact omissions and tone mismatches |

### Prompt Structure

```
SYSTEM PROMPT
  └── Role definition (Role-Playing)
  └── Task description
  └── 5-step reasoning protocol
  └── Output format specification (<EMAIL>...</EMAIL> tags)

FEW-SHOT EXAMPLE 1  (warm follow-up)
  └── Input: intent + facts + tone
  └── Full [STEP 1–5] reasoning
  └── Final email in tags

FEW-SHOT EXAMPLE 2  (urgent escalation)
  └── Input: intent + facts + tone
  └── Full [STEP 1–5] reasoning
  └── Final email in tags

LIVE TASK  (filled at runtime)
  └── {intent}, {key_facts_formatted}, {tone}
```

---

## Custom Evaluation Metrics

### Metric 1 — Fact Recall Score *(Automated / NLP)*

**Definition:** Measures what fraction of the provided key facts are faithfully reproduced in the generated email.

**Logic:**
1. Each key fact is tokenized into content-bearing words (nouns, verbs, adjectives, adverbs).
2. All words are **lemmatized** using NLTK's `WordNetLemmatizer` with POS-aware tagging.  
   This means morphological variants are treated as identical:
   - `request` / `requesting` / `requested` → all map to `"request"`
   - `meeting` / `meetings` / `met` → all map to `"meet"`
   - `urgent` / `urgently` → both map to `"urgent"`
3. A fact is **recalled** if ≥ 50% of its lemmatized content-words appear in the email's lemma set.
4. **Score** = recalled facts / total facts ∈ [0.0, 1.0]

**Why this threshold?** Business emails often rephrase facts rather than quoting them verbatim. A 50% keyword overlap is a robust signal that the fact's substance was included, without penalizing natural paraphrasing.

---

### Metric 2 — Tone Adherence Score *(LLM-as-a-Judge)*

**Definition:** Measures how accurately the generated email matches the intended communication tone.

**Logic:**
- A `gpt-4o-mini` judge is given the generated email and the target tone label.
- The judge applies a strict 10-point rubric:

| Score | Criterion |
|-------|-----------|
| 9–10 | Tone is perfectly consistent throughout; vocabulary and register are ideal |
| 7–8 | Mostly correct with 1–2 minor lapses |
| 5–6 | Tone elements present but inconsistent or mixed |
| 3–4 | Partially wrong; specific phrases contradict the intended tone |
| 1–2 | Entirely wrong tone |

- Raw score (1–10) is normalized to [0.0, 1.0] via `(score - 1) / 9`.
- Judge temperature = 0.0 for deterministic, reproducible scoring.

---

### Metric 3 — Fluency & Professionalism Score *(LLM-as-a-Judge)*

**Definition:** Measures grammar quality, structural completeness, and professional presentation.

**Logic:**
- The same `gpt-4o-mini` judge rates three sub-dimensions:

| Sub-dimension | Max Points | Criteria |
|---------------|:-----------:|---------|
| Grammar & Spelling | 4 | 4=flawless; 3=one minor error; 2=a few errors; 1=impedes fluency; 0=severe |
| Email Structure | 3 | 3=subject+greeting+body+closing; 2=missing one; 1=missing two; 0=none |
| Clarity & Conciseness | 3 | 3=crystal clear, no filler; 2=mostly clear; 1=vague; 0=confusing |

- **Total** = sum of sub-scores (max 10), normalized to [0.0, 1.0].

---

### Composite Score

```
composite_score = (M1 + M2 + M3) / 3
```

All three metrics are equally weighted for the final composite.

---

## Model Comparison

| Model | Provider | API Endpoint |
|-------|----------|-------------|
| `gpt-4o` | OpenAI | `https://api.openai.com/v1` |
| `grok-3` | xAI | `https://api.x.ai/v1` |

Both models:
- Receive the **identical** prompt (same template, same few-shot examples)
- Are evaluated on the **same 10 scenarios** with the **same 3 metrics**
- Use the same judge (`gpt-4o-mini`) for Metrics 2 and 3

The analysis summary (`data/results/analysis_summary.md`) answers:
1. Which model performed better across the 3 metrics?
2. What was the biggest failure mode of the lower-performing model?
3. Which model is recommended for production, with metric-backed justification?

---

## Running the Evaluation

```bash
# Full evaluation (both models, all 10 scenarios, all 3 metrics)
python run_evaluation.py
```

Expected runtime: ~5–10 minutes (20 generation calls + 60 judge calls, with rate-limit delays).

**Console output:**
- Live progress bars per model
- Colour-coded per-scenario score table (green ≥ 0.75, yellow ≥ 0.5, red < 0.5)
- Average scores per model


## Output Files

After running `run_evaluation.py`, the following files are created in `data/results/`:

| File | Description |
|------|-------------|
| `evaluation_report_<timestamp>.json` | Full results including generated emails, per-fact recall details, judge reasoning |
| `evaluation_report_<timestamp>.csv` | Flat tabular summary — one row per (model × scenario) |
| `evaluation_report_latest.json` | Always points to the most recent run |
| `evaluation_report_latest.csv` | Always points to the most recent run |
| `analysis_summary.md` | Auto-generated comparative analysis with metric table and production recommendation |

### CSV Schema

| Column | Description |
|--------|-------------|
| `model_name` | Human-readable model label |
| `scenario_id` | Scenario number (1–10) |
| `intent` | Email intent string |
| `tone` | Target tone |
| `m1_fact_recall` | Metric 1 score [0.0, 1.0] |
| `m2_tone_adherence_raw` | Metric 2 raw judge score (1–10) |
| `m2_tone_adherence_norm` | Metric 2 normalized [0.0, 1.0] |
| `m3_fluency_raw` | Metric 3 raw score (0–10) |
| `m3_fluency_norm` | Metric 3 normalized [0.0, 1.0] |
| `composite_score` | Average of M1 + M2 + M3 |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | ≥1.35.0 | OpenAI SDK (also used for Grok via OpenAI-compatible API) |
| `python-dotenv` | ≥1.0.0 | `.env` file loading |
| `nltk` | ≥3.8.1 | Tokenization, POS tagging, lemmatization for Metric 1 |
| `pandas` | ≥2.2.0 | Data manipulation utilities |
| `tabulate` | ≥0.9.0 | Console table formatting |
| `tqdm` | ≥4.66.0 | Progress bars |
| `colorama` | ≥0.4.6 | Cross-platform terminal colour output |
| `pytest` | (dev) | Unit test runner |

Install all with:

```bash
pip install -r requirements.txt
```

---

## Notes

- The **LLM judge** for Metrics 2 and 3 always uses `gpt-4o-mini` regardless of which model is being evaluated. This ensures a fair, consistent evaluation baseline.
- NLTK corpora (punkt, wordnet, stopwords, averaged_perceptron_tagger) are **auto-downloaded** on first run.
- If an API key for one model is missing, that model is skipped and the pipeline continues with the available model(s).
