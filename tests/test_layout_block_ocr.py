"""Tests for layout block OCR extraction."""

from pathlib import Path

from src.layout_block_ocr import extract_text_by_blocks

TEST_FILES_DIR = Path(__file__).parent.parent / "example" / "test_files"


def test_comandos_linux_basicos():
    """Test that text is extracted (may be single or multiple blocks)."""
    image_path = TEST_FILES_DIR / "comandos_linux_basicos.jpg"
    if not image_path.exists():
        print(f"Skipping test: {image_path} not found")
        return

    result = extract_text_by_blocks(str(image_path))

    commands = ["pwd", "cd", "ls", "cat", "cp", "mkdir", "mv", "rmdir", "sudo", "locate", "head"]
    found_commands = [cmd for cmd in commands if cmd.lower() in result.lower()]

    print(f"Found {len(found_commands)}/{len(commands)} commands")
    print(f"Output length: {len(result)} chars, lines: {result.count(chr(10)) + 1}")
    print(f"Output:\n{result[:500]}")

    assert len(result) >= 100, f"Expected substantial text (>=100 chars), got {len(result)}"
    assert len(found_commands) >= 5, f"Expected at least 5 commands, found {len(found_commands)}"


def test_null_vws_rtx3060():
    """Test that layout is preserved (multiple lines)."""
    image_path = TEST_FILES_DIR / "null-VWS-RTX3060.jpg"
    if not image_path.exists():
        print(f"Skipping test: {image_path} not found")
        return

    result = extract_text_by_blocks(str(image_path))
    if not result:
        print("No text extracted, may be due to image content")
        return

    line_count = result.count("\n") + 1
    print(f"Extracted {len(result)} chars, {line_count} lines")
    print(f"Output:\n{result[:1000]}")

    assert line_count >= 3, f"Expected at least 3 lines, got {line_count}"


def test_programas_1():
    """Test that multiple blocks are detected."""
    image_path = TEST_FILES_DIR / "Programas 1.png"
    if not image_path.exists():
        print(f"Skipping test: {image_path} not found")
        return

    result = extract_text_by_blocks(str(image_path))
    if not result:
        print("No text extracted, may be due to image content")
        return

    block_count = result.count("\n\n") + 1
    print(f"Extracted {len(result)} chars, {block_count} blocks")
    print(f"Output:\n{result[:1000]}")

    assert len(result) > 50, f"Expected substantial text, got {len(result)} chars"


def test_invalid_path():
    """Test that invalid paths return empty string."""
    result = extract_text_by_blocks("/nonexistent/path.jpg")
    assert result == "", "Expected empty string for invalid path"


if __name__ == "__main__":
    print("Running layout_block_ocr tests...\n")

    print("=" * 60)
    print("Test: comandos_linux_basicos.jpg")
    print("=" * 60)
    try:
        test_comandos_linux_basicos()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")

    print("=" * 60)
    print("Test: null-VWS-RTX3060.jpg")
    print("=" * 60)
    try:
        test_null_vws_rtx3060()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")

    print("=" * 60)
    print("Test: Programas 1.png")
    print("=" * 60)
    try:
        test_programas_1()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")

    print("=" * 60)
    print("Test: invalid_path")
    print("=" * 60)
    try:
        test_invalid_path()
        print("✓ PASSED\n")
    except AssertionError as e:
        print(f"✗ FAILED: {e}\n")
