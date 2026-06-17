"""
tests/test_generator.py
=======================
Unit tests for the email generation and evaluation pipeline.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generator import extract_email_body, _build_prompts
from src.evaluator import (
    _lemmatize_text,
    metric_fact_recall,
)


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------

class TestExtractEmailBody:
    """Tests for the <EMAIL>...</EMAIL> extraction logic."""

    def test_extracts_content_within_tags(self):
        raw = "Some reasoning...\n<EMAIL>\nSubject: Hello\n\nDear User,\nBody.\n</EMAIL>"
        result = extract_email_body(raw)
        assert "Subject: Hello" in result
        assert "Dear User" in result

    def test_case_insensitive_tags(self):
        raw = "<email>Subject: Test\n\nContent</email>"
        result = extract_email_body(raw)
        assert "Subject: Test" in result

    def test_fallback_when_no_tags(self):
        raw = "Subject: No tags here\n\nJust the email."
        result = extract_email_body(raw)
        assert result == raw.strip()

    def test_strips_surrounding_whitespace(self):
        raw = "<EMAIL>   \n  Subject: Trim me  \n   </EMAIL>"
        result = extract_email_body(raw)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_empty_email_tags(self):
        raw = "<EMAIL></EMAIL>"
        result = extract_email_body(raw)
        assert result == ""


class TestBuildPrompts:
    """Tests for prompt construction from template."""

    def test_returns_two_strings(self):
        system, user = _build_prompts("Test intent", ["Fact A", "Fact B"], "formal")
        assert isinstance(system, str)
        assert isinstance(user, str)

    def test_intent_in_user_prompt(self):
        _, user = _build_prompts("Schedule a meeting", ["Room 101"], "casual")
        assert "Schedule a meeting" in user

    def test_tone_in_user_prompt(self):
        _, user = _build_prompts("Follow up", ["Detail 1"], "urgent and assertive")
        assert "urgent and assertive" in user

    def test_key_facts_formatted_as_bullets(self):
        facts = ["Fact one", "Fact two", "Fact three"]
        _, user = _build_prompts("Notify", facts, "formal")
        for fact in facts:
            assert f"- {fact}" in user

    def test_system_prompt_contains_role(self):
        system, _ = _build_prompts("Intent", ["Fact"], "tone")
        assert "senior professional communications expert" in system

    def test_system_prompt_contains_few_shot_examples(self):
        system, _ = _build_prompts("Intent", ["Fact"], "tone")
        assert "FEW-SHOT EXAMPLE 1" in system
        assert "FEW-SHOT EXAMPLE 2" in system


# ---------------------------------------------------------------------------
# Evaluator — Metric 1 (Fact Recall) tests
# ---------------------------------------------------------------------------

class TestLemmatizeText:
    """Tests for the NLTK lemmatization helper."""

    def test_returns_set(self):
        result = _lemmatize_text("The quick brown fox jumps")
        assert isinstance(result, set)

    def test_removes_stopwords(self):
        result = _lemmatize_text("the and is are were")
        # All are stop-words; result should be empty or contain only content words
        stop_words_in_result = result & {"the", "and", "is", "are", "were"}
        assert len(stop_words_in_result) == 0

    def test_lemmatizes_verb_forms(self):
        """request / requesting / requested should all lemmatize to 'request'."""
        lemmas_base   = _lemmatize_text("request")
        lemmas_ing    = _lemmatize_text("requesting")
        lemmas_ed     = _lemmatize_text("requested")
        assert lemmas_base & lemmas_ing, "request and requesting should share a lemma"
        assert lemmas_base & lemmas_ed,  "request and requested should share a lemma"

    def test_lemmatizes_noun_plurals(self):
        """meetings → meet (verb) or meeting should reduce to a common lemma."""
        single = _lemmatize_text("meeting")
        plural = _lemmatize_text("meetings")
        assert single & plural, "meeting and meetings should share a lemma"

    def test_case_insensitive(self):
        lower = _lemmatize_text("invoice payment overdue")
        upper = _lemmatize_text("INVOICE PAYMENT OVERDUE")
        assert lower == upper


class TestMetricFactRecall:
    """Tests for Metric 1 — Fact Recall Score."""

    def test_perfect_recall(self):
        email = "The invoice #INV-001 for $500 is due on June 30th. Payment is required."
        facts = ["Invoice #INV-001 is due on June 30th", "Amount is $500"]
        result = metric_fact_recall(email, facts)
        assert result["score"] == 1.0
        assert result["recalled_count"] == 2

    def test_zero_recall(self):
        email = "Dear Customer, thank you for your interest."
        facts = ["Invoice due on July 15th for $10,000", "Contact: John Smith"]
        result = metric_fact_recall(email, facts)
        assert result["score"] == 0.0

    def test_partial_recall(self):
        email = "Your invoice is overdue."
        facts = ["Invoice #INV-999", "Amount $2,500", "Due date June 1st", "Contact: Alice"]
        result = metric_fact_recall(email, facts)
        assert 0.0 < result["score"] < 1.0

    def test_morphological_variants_recalled(self):
        """
        Fact uses 'requested' — email uses 'request'.
        Both should lemmatize to the same root, so the fact is recalled.
        """
        email = "We request that you review the proposal and provide your feedback."
        facts = ["Client requested a proposal review"]
        result = metric_fact_recall(email, facts)
        assert result["recalled_count"] == 1, (
            "Morphological variants (request/requested) should be treated as the same lemma"
        )

    def test_empty_facts_list(self):
        result = metric_fact_recall("Some email content here.", [])
        assert result["score"] == 0.0
        assert result["total_facts"] == 0

    def test_per_fact_structure(self):
        email = "The meeting is scheduled for Monday at 9 AM."
        facts = ["Meeting on Monday", "Budget is $50,000"]
        result = metric_fact_recall(email, facts)
        assert len(result["per_fact"]) == 2
        for item in result["per_fact"]:
            assert "fact" in item
            assert "recalled" in item
            assert "overlap_ratio" in item

    def test_score_range(self):
        email = "Some email text with various content about the project timeline."
        facts = ["Project timeline", "Budget allocation", "Stakeholder meeting"]
        result = metric_fact_recall(email, facts)
        assert 0.0 <= result["score"] <= 1.0

    def test_high_overlap_threshold(self):
        """
        A fact with 3 keywords where only 1 appears in the email (33% overlap)
        should NOT be recalled (threshold is 50%).
        """
        email = "The project is ongoing."       # only 'project' might match
        facts = ["project deadline budget overdue"]  # 4 content words, email has ~1
        result = metric_fact_recall(email, facts)
        # This tests the 50% threshold — 1/4 = 0.25 < 0.5, so not recalled
        assert result["per_fact"][0]["recalled"] is False


# ---------------------------------------------------------------------------
# Integration smoke test (no API calls)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Lightweight integration tests that do not make real API calls."""

    def test_full_prompt_contains_step_markers(self):
        system, user = _build_prompts(
            intent="Request a meeting",
            key_facts=["Available on Tuesday", "Prefers morning slots"],
            tone="casual",
        )
        full = system + user
        for step in ["STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"]:
            assert step in full, f"Prompt should contain {step}"

    def test_email_extraction_from_realistic_response(self):
        realistic = """
[STEP 1 — INTENT ANALYSIS]: The email aims to confirm a meeting.
[STEP 2 — FACT MAPPING]: Date → opening. Location → body.
[STEP 3 — TONE CALIBRATION]: Casual → first names, contractions.
[STEP 4 — STRUCTURE PLAN]: Short and friendly.
[STEP 5 — DRAFT]: See below.

<EMAIL>
Subject: Quick Catch-Up — Tuesday at 10 AM?

Hi Alex,

Just wanted to confirm our meeting for Tuesday morning. Looking forward to it!

Best,
Jordan
</EMAIL>
"""
        result = extract_email_body(realistic)
        assert "Subject: Quick Catch-Up" in result
        assert "Hi Alex" in result
        assert "<EMAIL>" not in result
        assert "</EMAIL>" not in result
