#!/usr/bin/env python
"""
OCR File Converter - Main entry point.

This demonstrates how to use the extracted OCR and preprocessing pipeline
to process files (images, PDFs, Office documents) with OCR capability.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from src.preprocessor import FilePreprocessor
from src.normalizer import TextNormalizer, TextSerializer
from src.errors import OCRError
from src.logging import get_logger

logger = get_logger(__name__)


def process_file(
    file_path: str,
    ocr_url: str = "http://localhost:8000",
    enable_ocr: bool = True,
    output_format: str = "txt",
    output_dir: Optional[str] = None,
    layout_anchors: Optional[list[str]] = None,
) -> Optional[Path]:
    """
    Process a single file: extract text, normalize, optionally serialize.

    Args:
        file_path: Path to file to process (image, PDF, Office doc, etc.)
        ocr_url: URL of OCR tool
        enable_ocr: Enable OCR for images/scanned PDFs
        output_format: Output format (txt, json, jsonl)
        output_dir: Output directory (default: /tmp)

    Returns:
        Path to processed file, or None if processing failed
    """
    input_path = Path(file_path)
    if not input_path.exists():
        logger.error("File not found: %s", file_path)
        return None

    logger.info("Processing file: %s (format=%s)", file_path, output_format)

    preprocessor = FilePreprocessor(
        ocr_url=ocr_url,
        enable_ocr=enable_ocr,
        enable_office=True,
        enable_archive=True,
    )

    status = preprocessor.get_status()
    logger.info("Preprocessor status: %s", status)

    result_path = preprocessor.preprocess(input_path)
    if not result_path:
        logger.error("Preprocessing failed for: %s", file_path)
        return None

    # Read extracted text
    try:
        text = result_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error("Failed to read %s: %s", result_path, exc)
        return None

    # Normalize text
    normalizer = TextNormalizer()
    normalized_text = normalizer.normalize(text, source_path=input_path, layout_anchors=layout_anchors)
    logger.info("Normalized text: %d → %d characters", len(text), len(normalized_text))

    # Save in requested format
    serializer = TextSerializer()
    output_stem = input_path.stem
    save_dir = Path(output_dir) if output_dir else result_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        output_file = save_dir / f"{output_stem}.json"
        serializer.save_json(normalized_text, output_file, source=str(input_path))
    elif output_format == "jsonl":
        output_file = save_dir / f"{output_stem}.jsonl"
        serializer.save_jsonl(normalized_text, output_file, source=str(input_path))
    else:  # txt (default)
        output_file = save_dir / f"{output_stem}_normalized.txt"
        output_file.write_text(normalized_text, encoding="utf-8")
        logger.info("Saved normalized text to %s", output_file)

    logger.info("Processing successful: %s -> %s", file_path, output_file)
    return output_file


def process_directory(
    directory: str,
    ocr_url: str = "http://localhost:8000",
    enable_ocr: bool = True,
    output_format: str = "txt",
    output_dir: Optional[str] = None,
    layout_anchors: Optional[list[str]] = None,
) -> dict:
    """
    Process all files in a directory recursively.

    Args:
        directory: Path to directory containing files
        ocr_url: URL of OCR tool
        enable_ocr: Enable OCR processing
        output_format: Output format (txt, json, jsonl)
        output_dir: Optional output directory (default: same as input)

    Returns:
        Dictionary with processing results
    """
    input_dir = Path(directory)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Directory not found: %s", directory)
        return {"success": False, "error": f"Directory not found: {directory}"}

    out_dir = Path(output_dir) if output_dir else input_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    supported_extensions = {
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
        ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".txt", ".md", ".csv", ".log",
    }

    results = {
        "total": 0,
        "processed": 0,
        "failed": 0,
        "files": [],
    }

    logger.info("Processing directory: %s", input_dir)

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            logger.debug("Skipping unsupported file: %s", file_path)
            continue

        results["total"] += 1
        logger.info("Processing [%d/%s] %s", results["processed"] + 1, "?", file_path.name)

        try:
            output_path = process_file(
                str(file_path),
                ocr_url=ocr_url,
                enable_ocr=enable_ocr,
                output_format=output_format,
                output_dir=output_dir,
                layout_anchors=layout_anchors,
            )

            if output_path:
                results["processed"] += 1
                results["files"].append({
                    "input": str(file_path),
                    "output": str(output_path),
                    "status": "success",
                })
            else:
                results["failed"] += 1
                results["files"].append({
                    "input": str(file_path),
                    "status": "failed",
                    "error": "Preprocessing returned None",
                })
        except OCRError as exc:
            results["failed"] += 1
            logger.error("OCR error processing %s: %s", file_path, exc)
            results["files"].append({
                "input": str(file_path),
                "status": "failed",
                "error": str(exc),
            })
        except Exception as exc:
            results["failed"] += 1
            logger.error("Unexpected error processing %s: %s", file_path, exc)
            results["files"].append({
                "input": str(file_path),
                "status": "failed",
                "error": f"Unexpected error: {exc}",
            })

    results["success"] = results["failed"] == 0
    logger.info(
        "Directory processing complete: %d processed, %d failed",
        results["processed"],
        results["failed"],
    )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OCR File Converter - Process documents with OCR, normalization, and serialization"
    )
    parser.add_argument(
        "path",
        help="Path to file or directory to process",
    )
    parser.add_argument(
        "--ocr-url",
        default="http://localhost:8000",
        help="URL of OCR tool service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR processing",
    )
    parser.add_argument(
        "--format",
        choices=["txt", "json", "jsonl"],
        default="txt",
        help="Output format: txt (default), json, or jsonl (line-delimited JSON)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--layout-anchors",
        default="",
        help="Comma-separated anchor labels to rebuild infographic-like OCR blocks",
    )

    args = parser.parse_args()

    layout_anchors = [item.strip() for item in args.layout_anchors.split(",") if item.strip()]

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    input_path = Path(args.path)

    # Check if it's a directory or file
    if input_path.is_dir():
        results = process_directory(
            args.path,
            ocr_url=args.ocr_url,
            enable_ocr=not args.no_ocr,
            output_format=args.format,
            output_dir=args.output,
            layout_anchors=layout_anchors or None,
        )

        print(f"\n✓ Directory processing complete!")
        print(f"  Total files: {results['total']}")
        print(f"  Processed: {results['processed']}")
        print(f"  Failed: {results['failed']}")

        if results["files"]:
            print("\n  Details:")
            for file_info in results["files"][:10]:  # Show first 10
                status = "✓" if file_info["status"] == "success" else "✗"
                print(f"    {status} {Path(file_info['input']).name}")

        return 0 if results["success"] else 1

    elif input_path.is_file():
        result_path = process_file(
            args.path,
            ocr_url=args.ocr_url,
            enable_ocr=not args.no_ocr,
            output_format=args.format,
        )

        if result_path:
            print(f"\n✓ Processing complete!")
            print(f"  Input:  {args.path}")
            print(f"  Output: {result_path}")
            return 0
        else:
            print(f"\n✗ Processing failed for {args.path}", file=sys.stderr)
            return 1

    else:
        print(f"\n✗ Path not found: {args.path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
