import pytest
from backend.services.copilot_service import copilot_service

def test_copilot_model_validation_query():
    ans = copilot_service.answer_query("How was the model validated and what is its R2 on unseen years?")
    assert 'answer' in ans
    assert 'evidence' in ans
    assert ans['records_analyzed'] > 0
    assert '0.78' in ans['answer'] or 'R²' in ans['answer'] or 'out-of-time' in ans['answer'].lower()

def test_copilot_drift_query():
    ans = copilot_service.answer_query("Is the model experiencing any feature drift?")
    assert 'answer' in ans
    assert 'drift' in ans['answer'].lower() or 'psi' in ans['answer'].lower()

def test_copilot_data_quality_query():
    ans = copilot_service.answer_query("What is the data quality score of the agricultural panel?")
    assert 'answer' in ans
    assert 'quality' in ans['answer'].lower() or '100' in ans['answer']

def test_copilot_model_registry_query():
    ans = copilot_service.answer_query("What are the active registered models and versions?")
    assert 'answer' in ans
    assert 'registered' in ans['answer'].lower() or 'pipeline' in ans['answer'].lower()
