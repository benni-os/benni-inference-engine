"""Tests for Skill 81 Ceiling Extractor."""
from bie.skill81.ceiling_extractor import CeilingExtractor

def test_extract_passthrough():
    extractor = CeilingExtractor()
    messages = [{"role": "user", "content": "Hello"}]
    result = extractor.extract(messages, model_profile={}, task_type="general")
    assert isinstance(result, list)
    assert len(result) >= 1

def test_task_type_map():
    extractor = CeilingExtractor()
    assert "DELTA" in extractor.TASK_TYPE_MAP
    assert "BRAVO" in extractor.TASK_TYPE_MAP
    assert "CHARLIE" in extractor.TASK_TYPE_MAP
