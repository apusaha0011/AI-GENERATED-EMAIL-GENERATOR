# Comparative Model Analysis Summary

_Generated: 2026-06-18 01:46:58_

---

## Metric Definitions

| Metric | Definition | Method | Scale |
|--------|-----------|--------|-------|
| **M1 — Fact Recall** | Fraction of key facts whose lemmatized keywords appear in the generated email (≥50% keyword overlap per fact counts as recalled). Morphological variants (e.g., request/requesting/requested) are treated as the same lemma. | NLTK lemmatization + set overlap | 0.0 – 1.0 |
| **M2 — Tone Adherence** | How accurately the generated email matches the requested tone, rated by GPT-4o-mini on a 1–10 rubric (9–10: perfect match; 1–2: entirely wrong tone). | LLM-as-a-Judge | 0.0 – 1.0 |
| **M3 — Fluency & Professionalism** | Grammar & Spelling (0–4) + Email Structure (0–3) + Clarity & Conciseness (0–3), rated by GPT-4o-mini, summed to 10. | LLM-as-a-Judge | 0.0 – 1.0 |

---

## Score Summary

| Model | M1 Fact Recall | M2 Tone Adherence | M3 Fluency | Composite |
|-------|:--------------:|:-----------------:|:----------:|:---------:|
| OpenAI GPT-4o | 0.9800 | 0.9000 | 1.0000 | **0.9600** |
| xAI Grok-3 | 0.9550 | 0.8778 | 1.0000 | **0.9443** |

---

## Analysis

### 1. Which model performed better?

**OpenAI GPT-4o** achieved a higher composite score (0.9600) across all 10 evaluation scenarios.
It outperformed **xAI Grok-3** (0.9443) on Fact Recall by +0.0250, Tone Adherence by +0.0222, and Fluency by +0.0000.

### 2. Biggest failure mode of xAI Grok-3

The weakest dimension for **xAI Grok-3** was **Tone Adherence** (avg: 0.8778). 

### 3. Production Recommendation

Based on the evaluation data, **OpenAI GPT-4o** is recommended for production use.

**Justification:**
- Higher M1 Fact Recall ensures that business-critical information is faithfully included in generated emails, reducing the risk of omissions that could cause miscommunication.
- Higher M2 Tone Adherence means the model can reliably shift register (formal, urgent, empathetic) on demand — a key requirement for a production email assistant serving diverse communication scenarios.
- Higher M3 Fluency & Professionalism confirms that outputs are consistently well-structured and polished, requiring minimal human editing before sending.

---
_Report generated automatically by the Email Generation Assistant evaluation pipeline._