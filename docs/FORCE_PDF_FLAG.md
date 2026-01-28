# Force PDF Import Flag Documentation

This document describes the `--forcePdf` flag implementation for the content-import pipeline.

## Overview

The `--forcePdf` flag allows users to bypass domain-specific blockers (like arXiv) and force any URL to be processed as a generic PDF using the PaddleOCR-VL adapter.

## Problem Statement

Previously, arXiv URLs (e.g., `https://arxiv.org/pdf/2306.00978`) were blocked with the message "arXiv import is no longer supported". This prevented users from importing arXiv papers even though they are PDFs that could be processed by the generic PDF importer.

## Solution

A new `--forcePdf` flag has been added that:

1. Bypasses arXiv domain blocking
2. Forces the use of the generic PDF adapter (PaddleOCR-VL)
3. Allows any PDF URL to be imported, regardless of domain

## Usage

### Command Line

```bash
# Default behavior - arXiv URLs will be blocked
npm run import:content -- --url https://arxiv.org/pdf/2306.00978
# Error: arXiv import is no longer supported

# With --forcePdf flag - arXiv URL will be imported as PDF
npm run import:content -- --url https://arxiv.org/pdf/2306.00978 --forcePdf

# Also accepts --force-pdf
npm run import:content -- --url https://arxiv.org/pdf/2306.00978 --force-pdf
```

### Environment Variable

```bash
export FORCE_PDF=true
npm run import:content -- --url https://arxiv.org/pdf/2306.00978
```

### GitHub Actions Workflow

The `import-content.yml` workflow now includes a `force_pdf` boolean input:

1. Go to Actions → Import Content
2. Click "Run workflow"
3. Fill in the URL field with your arXiv PDF URL
4. Check the "Force PDF import mode" checkbox
5. Run the workflow

## Implementation Details

### Files Modified

1. **scripts/content-import.ts**
   - Added `forcePdf` field to `ImportArgs` type
   - Updated `parseArgs()` to read `--forcePdf` flag from CLI args or `FORCE_PDF` env var
   - Modified arXiv URL check to skip when `forcePdf=true`
   - Updated adapter resolution to force PDF adapter when `forcePdf=true`

2. **scripts/utils/errors.ts** (NEW)
   - Created `serializeError()` utility function
   - Ensures all errors have consistent structure with message/name/stack/cause
   - Prevents `error={}` empty object logs

3. **.github/workflows/import-content.yml**
   - Added `force_pdf` workflow input (boolean, default: false)
   - Pass flag to CLI as `--forcePdf` when enabled

### Error Serialization

All `logger.error()` calls now use `serializeError()` to ensure proper error formatting:

```typescript
// Before (could produce error={})
logger.error('Content import failed', { error });

// After (always has structure)
logger.error('Content import failed', { error: serializeError(error) });
```

Error structure always includes:

- `message`: Error message string
- `name`: Error type name
- `stack`: Stack trace (if available)
- `cause`: Nested cause error (if present)

### Adapter Selection Priority

With `--forcePdf=true`:

1. Skip domain blocklists (arXiv, ar5iv)
2. Force PDF adapter regardless of URL pattern
3. Process URL as generic PDF

Without flag (default behavior):

1. Check domain blocklists (arXiv → blocked)
2. Use normal adapter resolution by URL pattern
3. PDF adapter only for `.pdf` URLs

## Testing

### Unit Tests

New test file: `tests/unit/force-pdf-flag.test.ts`

Covers:

- Default arXiv blocking (forcePdf=false)
- arXiv import with forcePdf=true
- Error serialization with message/name/stack
- Proper adapter selection

### Running Tests

```bash
# Run specific test
npm run test -- tests/unit/force-pdf-flag.test.ts

# Run all tests
npm run test

# Run full CI suite
npm run check
npm run test
npm run test:e2e
npm run lint
```

## Examples

### Example 1: Import arXiv Paper

```bash
# This will fail with "arXiv not supported" error
npm run import:content -- --url https://arxiv.org/pdf/2306.00978

# This will import successfully using PDF adapter
npm run import:content -- --url https://arxiv.org/pdf/2306.00978 --forcePdf
```

### Example 2: Combined with Other Flags

```bash
npm run import:content -- \
  --url https://arxiv.org/pdf/2306.00978 \
  --forcePdf \
  --use-first-image-as-cover \
  --allow-overwrite
```

### Example 3: Environment Variable Setup

```bash
# .env.local
PADDLEOCR_VL_TOKEN=your-token-here
FORCE_PDF=true

# Run import
npm run import:content -- --url https://arxiv.org/pdf/2306.00978
```

## Requirements

- `PADDLEOCR_VL_TOKEN` environment variable must be set for PDF import
- PDF must be accessible via HTTP/HTTPS
- PDF must be text-based (not scanned images)
- Content must have at least 20 effective lines

## Error Handling

All errors are now properly serialized with:

- Clear error messages
- Stack traces for debugging
- Sensitive information redacted (tokens, API keys)
- Nested error causes preserved

Example error output:

```json
{
  "message": "Failed to download PDF",
  "name": "NetworkError",
  "stack": "NetworkError: Failed to download PDF\n    at ...",
  "cause": {
    "message": "Connection timeout",
    "name": "TimeoutError"
  }
}
```

## Security Considerations

- Error serialization automatically redacts sensitive values (tokens, API keys)
- Sensitive URL parameters are masked in logs
- No secrets are logged to console or files

## Backward Compatibility

- Default behavior unchanged: arXiv URLs are still blocked
- Existing imports continue to work as before
- All CLI flags and environment variables are optional
- No breaking changes to existing functionality

## Future Enhancements

Potential future improvements:

- Support for more PDF sources
- Batch import of multiple PDFs
- Progress indicators for long PDF processing
- PDF quality validation before import
