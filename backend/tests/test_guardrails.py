import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.guardrails import check_query, append_disclaimer_if_needed, DISCLAIMER


def test_emergency_pattern_blocks_generation():
    result = check_query("I have severe chest pain and can't breathe")
    assert result.flagged is True
    assert result.block_generation is True
    assert result.prepend_message is not None


def test_personal_diagnosis_flagged_but_not_blocked():
    result = check_query("Do I have diabetes based on these symptoms?")
    assert result.flagged is True
    assert result.block_generation is False


def test_general_query_not_flagged():
    result = check_query("What are the symptoms of measles?")
    assert result.flagged is False
    assert result.block_generation is False


def test_disclaimer_appended_when_flagged():
    result = check_query("Should I take my medication dosage differently?")
    answer = append_disclaimer_if_needed("Some general info.", result)
    assert DISCLAIMER in answer


def test_disclaimer_not_appended_when_not_flagged():
    result = check_query("What is the WHO definition of a pandemic?")
    answer = append_disclaimer_if_needed("Some general info.", result)
    assert DISCLAIMER not in answer
