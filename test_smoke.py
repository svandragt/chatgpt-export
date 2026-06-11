"""Smoke tests for export_chat module."""
import importlib.util
import sys
import os


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_chat",
        os.path.join(os.path.dirname(__file__), "export_chat.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_chat"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_module_importable():
    """The main script can be imported without errors."""
    mod = _load_module()
    assert mod is not None


def test_extract_text_parts_string():
    """_extract_text_parts handles plain string content."""
    mod = _load_module()
    assert mod._extract_text_parts("hello") == "hello"


def test_extract_text_parts_none():
    """_extract_text_parts handles None gracefully."""
    mod = _load_module()
    assert mod._extract_text_parts(None) == ""
