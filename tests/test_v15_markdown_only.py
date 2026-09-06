from pathlib import Path

from backend.core.config import settings


def test_markdown_is_the_only_public_upload_type():
    assert settings.ALLOWED_FILE_EXTENSIONS == [".md"]
    assert Path("notes.md").suffix.lower() in settings.ALLOWED_FILE_EXTENSIONS
    for filename in ("notes.pdf", "notes.docx", "notes.txt", "notes.png"):
        assert Path(filename).suffix.lower() not in settings.ALLOWED_FILE_EXTENSIONS
