from pathlib import Path

from scripts.create_sample_character import create_sample_character


def pytest_sessionstart(session):
    create_sample_character(Path("character_input"))
