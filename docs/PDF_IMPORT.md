# PDF Import Feature

This document describes the generic PDF import feature that uses PaddleOCR-VL for layout parsing and text extraction.

## Overview

The PDF import feature allows importing any PDF document as a blog post. The system:

1. Downloads the PDF from a given URL
2. Uses PaddleOCR-VL API for layout parsing and text extraction
3. Converts the extracted content to Markdown
4. Downloads and saves all images
5. Optionally translates the content using DeepSeek
6. Saves the result as a blog post with proper frontmatter

## Configuration

### Environment Variables

Add these to your `.env.local` file:

```env
# Required - PaddleOCR-VL API token
PADDLEOCR_VL_TOKEN=your_token_here

# Optional - API URL (defaults to PaddleOCR-VL endpoint)
PADDLEOCR_VL_API_URL=https://xbe1mb28fa0dz7kb.aistudio-app.com/layout-parsing

# Optional - Maximum PDF size in MB (default: 50)
PDF_MAX_MB=50

# Optional - Enable translation (0 = disabled, 1 = enabled)
MARKDOWN_TRANSLATE_ENABLED=1
MARKDOWN_TRANSLATE_PROVIDER=deepseek

# Optional - DeepSeek API key (required if translation is enabled)
DEEPSEEK_API_KEY=sk-your-api-key-here
```

### GitHub Actions

The import workflow has been updated to support PDF imports. When using the workflow:

1. Set `PADDLEOCR_VL_TOKEN` in GitHub Secrets
2. Provide a PDF URL in the workflow input
3. The system will automatically detect it's a PDF and use the appropriate adapter

## Usage

### Command Line

```bash
# Import a PDF without translation
npm run import:content -- --url https://example.com/paper.pdf

# Import a PDF with translation enabled
MARKDOWN_TRANSLATE_ENABLED=1 npm run import:content -- --url https://example.com/paper.pdf

# Import from blocked domains (e.g., arXiv) using --forcePdf flag
npm run import:content -- --url https://arxiv.org/pdf/2306.00978 --forcePdf

# Note: Without --forcePdf, arXiv URLs are blocked by default
# The --forcePdf flag forces the use of the generic PDF importer
```

### GitHub Actions Workflow

1. Go to Actions → Import Content
2. Click "Run workflow"
3. Enter the PDF URL (e.g., `https://example.com/document.pdf`)
4. For blocked domains like arXiv, check "Force PDF import mode"
5. Configure options:
   - Enable translation if desired
   - Choose translation provider (deepseek recommended)
6. Run the workflow

## Features

### PDF Download & Validation

- Follows redirects automatically
- Retries on failure (3 attempts with exponential backoff)
- Validates file is actually a PDF (checks magic bytes `%PDF-`)
- Enforces size limits (default 50MB max)
- Validates minimum size (50KB) to catch incomplete downloads

### OCR Processing

- Uses PaddleOCR-VL layout parsing API
- Extracts text with proper formatting
- Identifies and maps images
- Maintains document structure (headings, lists, tables)

### Markdown Processing

- Fixes unclosed code fences
- Normalizes list indentation
- Removes excessive blank lines
- Validates content quality (minimum 20 effective lines)
- Ensures MDX compatibility

### Image Handling

- Downloads all images referenced in the OCR result
- Stores images in `/images/pdf/{slug}/` directory
- Updates markdown to reference local image paths
- Supports multiple image formats (PNG, JPG, GIF, WebP)
- Validates image format using magic bytes
- Prevents path traversal attacks

### Optional Translation

- Integrates with existing translation system
- Uses DeepSeek for translation
- Preserves code blocks, URLs, and technical terms
- Maintains markdown structure

### Security

- Never logs tokens or sensitive data
- Validates all file paths to prevent directory traversal
- Enforces file size limits
- Validates file types

## Error Handling

The system provides clear error messages for common issues:

- **Missing token**: Clear message indicating `PADDLEOCR_VL_TOKEN` is required
- **Download failures**: HTTP status codes and error details
- **Invalid PDF**: Detected when file doesn't start with `%PDF-`
- **File too large**: Shows actual size vs. limit
- **Insufficient content**: When OCR returns less than 20 effective lines
- **OCR API errors**: Includes HTTP status and error message

## Testing

The feature includes comprehensive unit tests:

```bash
# Run PDF adapter tests
npm test -- tests/unit/pdf-vl-adapter.test.ts

# Run all tests
npm test
```

Tests cover:

- URL detection (PDF vs. non-PDF)
- Successful PDF import flow
- Error cases (missing token, download failures, invalid PDFs)
- Content quality validation
- Image download and processing

## Architecture

### Files

- `scripts/import/adapters/pdf_vl.ts` - Main adapter implementation
- `scripts/import/adapters/pdf_vl_utils.ts` - PDF download and validation
- `scripts/import/adapters/pdf_vl_ocr.ts` - PaddleOCR-VL API client
- `scripts/import/adapters/pdf_vl_markdown.ts` - Markdown processing and image handling
- `tests/unit/pdf-vl-adapter.test.ts` - Comprehensive unit tests

### Flow

1. **Detection**: Adapter checks if URL ends with `.pdf`
2. **Download**: PDF is downloaded with validation and retries
3. **Validation**: Check file is valid PDF and meets size requirements
4. **OCR**: Send to PaddleOCR-VL API for layout parsing
5. **Processing**: Clean up markdown, validate content quality
6. **Images**: Download all images and update references
7. **Translation** (optional): Translate content while preserving structure
8. **Output**: Generate blog post with proper frontmatter

## Limitations

- Maximum PDF size: 50MB (configurable)
- Minimum content requirement: 20 effective lines
- Only PDF files are supported (no other document formats)
- Requires internet connection for OCR API
- Requires PaddleOCR-VL API token

## Troubleshooting

### "PADDLEOCR_VL_TOKEN environment variable is required"

Solution: Add `PADDLEOCR_VL_TOKEN` to your `.env.local` file or GitHub Secrets.

### "File too small" or "Not a valid PDF file"

The downloaded file is not a valid PDF. Check:

- The URL actually points to a PDF
- The server is accessible
- No authentication is required

### "Insufficient content quality"

The OCR returned less than 20 effective lines of content. This may indicate:

- The PDF is a scanned image (OCR may struggle)
- The PDF is mostly images with little text
- The OCR service had an issue

Solution: Verify the PDF contains text (not just scanned images).

### "Failed to download PDF after 3 attempts"

Network or server issues. Check:

- The URL is accessible
- Your network connection is stable
- The server is responding

## Future Enhancements

Potential improvements:

- Support for other document formats (DOCX, PPTX, etc.)
- Batch PDF processing
- Custom OCR provider support
- Enhanced image optimization
- Better handling of complex layouts (multi-column, etc.)
