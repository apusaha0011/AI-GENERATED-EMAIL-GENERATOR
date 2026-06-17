"""
src/evaluator.py
================
Three custom evaluation metrics for the Email Generation Assistant.

METRIC 1 — Fact Recall Score  (Automated / NLP)
------------------------------------------------
Measures what fraction of the provided key facts are "recalled" in the generated email.

Logic:
  - Each key fact is tokenized into content-bearing words (nouns, verbs, adjectives, adverbs).
  - Words are reduced to their base lemma using NLTK WordNetLemmatizer, so morphological
    variants are treated as equivalent:
      request / requesting / requested → "request"
      meeting / meetings / met         → "meet"
      urgent / urgently                → "urgent"
  - A fact is considered RECALLED if ≥ 50% of its lemmatized keywords appear in the
    lemmatized token set of the generated email.
  - Score = (number of recalled facts) / (total facts), range [0.0, 1.0].

METRIC 2 — Tone Adherence Score  (LLM-as-a-Judge)
--------------------------------------------------
Measures how accurately the generated email matches the intended tone.

Logic:
  - A GPT-4o-mini judge receives the generated email and the target tone label.
  - The judge is given a strict 1–10 rubric:
      9–10: Tone is perfectly consistent throughout, vocabulary and register are ideal.
      7–8:  Mostly correct tone with minor lapses.
      5–6:  Detectable tone elements but inconsistent or mixed signals.
      3–4:  Tone is partially wrong or contradicted by specific phrases.
      1–2:  Wrong tone entirely.
  - Raw score (1–10) is normalised to [0.0, 1.0].

METRIC 3 — Fluency & Professionalism Score  (LLM-as-a-Judge)
-------------------------------------------------------------
Measures grammar quality, structural completeness, and professional presentation.

Logic:
  - The same GPT-4o-mini judge rates the email on three sub-dimensions:
      Grammar & Spelling  (0–4): Penalises errors, awkward phrasing, run-ons.
      Email Structure     (0–3): Subject line, greeting, body paragraphs, closing.
      Clarity & Conciseness (0–3): No filler, actionable sentences, clear message.
  - Total = sum of sub-scores (0–10), normalised to [0.0, 1.0].
"""

from __future__ import annotations

import json
import os
import re
import string
from typing import Optional

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# NLTK bootstrap — download required corpora once
# ---------------------------------------------------------------------------

def _ensure_nltk_data() -> None:
    """Download required NLTK corpora if not already present."""
    packages = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
        ("corpora/wordnet",           "wordnet"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, package in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _penn_to_wordnet(tag: str) -> str:
    """Convert a Penn Treebank POS tag to a WordNet POS constant."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def _lemmatize_text(text: str) -> set[str]:
    """
    Tokenize `text`, remove punctuation and stop-words, POS-tag remaining
    tokens, then lemmatize each to its base form.

    Returns:
        A set of lemmatized content-word strings (lowercase).
    """
    # Lowercase and strip punctuation
    text_clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text_clean)

    # Keep content tokens that are alphabetic OR contain digits or '@' or '-' (identifiers)
    def _is_content_token(tok: str) -> bool:
        if tok in _STOP_WORDS:
            return False
        if tok.isalpha():
            return True
        # Allow tokens that include digits (e.g., INV-2024-0087), '@' (emails), or hyphens
        if any(ch.isdigit() for ch in tok) or "@" in tok or "-" in tok:
            return True
        return False

    content_tokens = [t for t in tokens if _is_content_token(t)]

    # POS-tag alphabetic tokens for accurate lemmatization; keep non-alpha tokens as-is
    alpha_tokens = [t for t in content_tokens if t.isalpha()]
    tagged = pos_tag(alpha_tokens)
    alpha_lemmas = {
        _lemmatizer.lemmatize(word, _penn_to_wordnet(tag))
        for word, tag in tagged
    }

    # Non-alpha tokens (emails, IDs, numbers) are preserved in lower-case form
    non_alpha_lemmas = {t.lower() for t in content_tokens if not t.isalpha()}

    lemmas = alpha_lemmas | non_alpha_lemmas
    return lemmas


def _get_judge_client() -> OpenAI:
    """Return an OpenAI client used for LLM-as-a-Judge calls (always GPT-4o-mini)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. LLM-judge metrics require OpenAI.")
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# METRIC 1 — Fact Recall Score
# ---------------------------------------------------------------------------

def metric_fact_recall(generated_email: str, key_facts: list[str]) -> dict:
    """
    Compute the Fact Recall Score.

    Each key fact is lemmatized. A fact is "recalled" if at least 50% of its
    lemmatized content-words appear in the lemmatized word set of the email.

    Args:
        generated_email: The email text produced by the model.
        key_facts: List of fact strings from the input scenario.

    Returns:
        dict with keys:
          score (float 0–1), recalled_count (int), total_facts (int),
          per_fact (list of dicts with 'fact', 'recalled', 'overlap_ratio').
    """
    email_lemmas = _lemmatize_text(generated_email)
    per_fact = []
    recalled_count = 0

    for fact in key_facts:
        fact_lemmas = _lemmatize_text(fact)
        if not fact_lemmas:
            # Empty fact — skip
            per_fact.append({"fact": fact, "recalled": True, "overlap_ratio": 1.0})
            recalled_count += 1
            continue

        overlap = fact_lemmas & email_lemmas
        overlap_ratio = len(overlap) / len(fact_lemmas)
        recalled = overlap_ratio >= 0.5

        if recalled:
            recalled_count += 1

        per_fact.append({
            "fact": fact,
            "recalled": recalled,
            "overlap_ratio": round(overlap_ratio, 4),
        })

    total = len(key_facts)
    score = recalled_count / total if total > 0 else 0.0

    return {
        "score": round(score, 4),
        "recalled_count": recalled_count,
        "total_facts": total,
        "per_fact": per_fact,
    }


# ---------------------------------------------------------------------------
# METRIC 2 — Tone Adherence Score
# ---------------------------------------------------------------------------

_TONE_JUDGE_SYSTEM = """You are an expert communications evaluator. 
Your task is to assess how accurately an email matches a specified tone.

Rate the email on a scale of 1–10 using this rubric:
  9–10: Tone is perfectly consistent throughout; vocabulary, register, and sentence 
        structure are ideal for the specified tone. No contradicting signals.
  7–8:  Mostly correct tone with only minor lapses (1-2 phrases feel off).
  5–6:  Tone elements are detectable but inconsistent or mixed with another tone.
  3–4:  The tone is partially wrong; specific phrases actively contradict it.
  1–2:  The tone is entirely wrong or the opposite of what was requested.

Respond ONLY with a valid JSON object in this exact format (no markdown, no extra text):
{"score": <integer 1-10>, "reasoning": "<one sentence explanation>"}"""


def metric_tone_adherence(
    generated_email: str,
    target_tone: str,
    judge_client: Optional[OpenAI] = None,
) -> dict:
    """
    Compute the Tone Adherence Score using GPT-4o-mini as a judge.

    Args:
        generated_email: The email text produced by the model.
        target_tone: The intended tone string (e.g., "formal", "urgent").
        judge_client: Optional pre-built OpenAI client; created if not supplied.

    Returns:
        dict with keys:
          score (float 0–1), raw_score (int 1–10), reasoning (str).
    """
    client = judge_client or _get_judge_client()

    user_message = (
        f"TARGET TONE: {target_tone}\n\n"
        f"EMAIL:\n{generated_email}\n\n"
        "Rate how well this email matches the target tone using the rubric provided."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,   # deterministic judge
        max_tokens=150,
        messages=[
            {"role": "system", "content": _TONE_JUDGE_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )

    raw_text = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        result = json.loads(raw_text)
        raw_score = int(result["score"])
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: attempt regex extraction
        match = re.search(r'"score"\s*:\s*(\d+)', raw_text)
        raw_score = int(match.group(1)) if match else 5
        reasoning = raw_text

    raw_score = max(1, min(10, raw_score))  # clamp to [1, 10]
    normalized = round((raw_score - 1) / 9, 4)   # map [1,10] → [0,1]

    return {
        "score": normalized,
        "raw_score": raw_score,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# METRIC 3 — Fluency & Professionalism Score
# ---------------------------------------------------------------------------

_FLUENCY_JUDGE_SYSTEM = """You are an expert business writing evaluator.
Assess the email on three dimensions and return integer sub-scores:

1. Grammar & Spelling (0–4):
   4 = Flawless. 3 = One minor error. 2 = A few errors but readable. 
   1 = Noticeable errors impeding fluency. 0 = Severely error-ridden.

2. Email Structure (0–3):
   3 = Has subject line, proper greeting, well-organised body paragraphs, and closing.
   2 = Missing one structural element. 1 = Missing two elements. 0 = No structure.

3. Clarity & Conciseness (0–3):
   3 = Crystal clear, no filler, every sentence has purpose.
   2 = Mostly clear, minor redundancy. 1 = Vague or padded. 0 = Confusing/incoherent.

Respond ONLY with a valid JSON object (no markdown, no extra text):
{"grammar": <0-4>, "structure": <0-3>, "clarity": <0-3>, "reasoning": "<one sentence>"}"""


def metric_fluency_professionalism(
    generated_email: str,
    judge_client: Optional[OpenAI] = None,
) -> dict:
    """
    Compute the Fluency & Professionalism Score using GPT-4o-mini as a judge.

    Args:
        generated_email: The email text produced by the model.
        judge_client: Optional pre-built OpenAI client; created if not supplied.

    Returns:
        dict with keys:
          score (float 0–1), raw_score (int 0–10),
          grammar (int), structure (int), clarity (int), reasoning (str).
    """
    client = judge_client or _get_judge_client()

    user_message = f"EMAIL:\n{generated_email}\n\nEvaluate this email on grammar, structure, and clarity."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=200,
        messages=[
            {"role": "system", "content": _FLUENCY_JUDGE_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )

    raw_text = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw_text)
        grammar   = int(result.get("grammar",   2))
        structure = int(result.get("structure", 2))
        clarity   = int(result.get("clarity",   2))
        reasoning = result.get("reasoning", "")
    except (json.JSONDecodeError, KeyError, ValueError):
        grammar = structure = clarity = 2
        reasoning = raw_text

    # Clamp sub-scores to valid ranges
    grammar   = max(0, min(4, grammar))
    structure = max(0, min(3, structure))
    clarity   = max(0, min(3, clarity))

    raw_score = grammar + structure + clarity        # max = 10
    normalized = round(raw_score / 10, 4)            # map [0,10] → [0,1]

    return {
        "score": normalized,
        "raw_score": raw_score,
        "grammar": grammar,
        "structure": structure,
        "clarity": clarity,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------

def evaluate_email(
    generated_email: str,
    key_facts: list[str],
    target_tone: str,
    judge_client: Optional[OpenAI] = None,
) -> dict:
    """
    Run all three metrics on a single generated email and return a composite result.

    Args:
        generated_email: Email text from the model.
        key_facts: List of required facts.
        target_tone: Intended tone.
        judge_client: Shared OpenAI client for LLM-judge calls.

    Returns:
        dict containing results from all three metrics + composite_score (average).
    """
    m1 = metric_fact_recall(generated_email, key_facts)
    m2 = metric_tone_adherence(generated_email, target_tone, judge_client)
    m3 = metric_fluency_professionalism(generated_email, judge_client)

    composite = round((m1["score"] + m2["score"] + m3["score"]) / 3, 4)

    return {
        "metric_1_fact_recall":          m1,
        "metric_2_tone_adherence":       m2,
        "metric_3_fluency_professionalism": m3,
        "composite_score":               composite,
    }
