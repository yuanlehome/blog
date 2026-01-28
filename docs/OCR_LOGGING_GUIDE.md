# OCR-VL Logging Guide

This guide explains how to use the structured logs from the OCR-VL import pipeline to diagnose failures.

## Log Structure

All OCR-VL logs follow a structured JSON format with these key correlation fields:

- `ocrJobId`: Unique identifier for each OCR job (e.g., `ocr-a1b2c3d4`)
- `runId`: Overall import run identifier (if available from parent logger)
- `event`: Span markers (`span.start`, `span.end`, `ocr-request-prepare`, `ocr-request-success`, `ocr-request-fail`)
- `errorCode`: Classification of errors for quick diagnosis

## Example: Successful OCR Import

### 1. OCR Job Start

```json
{
  "ts": "2026-01-28T18:00:00.123Z",
  "level": "info",
  "msg": "OCR-VL job starting",
  "event": "span.start",
  "span": "ocr-vl-job",
  "adapter": "pdf_vl",
  "stage": "ocr",
  "ocrJobId": "ocr-a1b2c3d4",
  "sourceUrl": "https://example.com/paper.pdf",
  "adapterId": "pdf_vl",
  "adapterName": "PDF VL (Generic PDF Import)",
  "pdfBytes": 1024000,
  "ocrProvider": "paddle-ocr-vl",
  "ocrMode": "layout-parsing-markdown",
  "startTs": "2026-01-28T18:00:00.123Z"
}
```

### 2. OCR Request Preparation

```json
{
  "ts": "2026-01-28T18:00:00.234Z",
  "level": "info",
  "msg": "OCR request starting",
  "module": "pdf_vl_ocr",
  "event": "ocr-request-prepare",
  "ocrJobId": "ocr-a1b2c3d4",
  "attempt": 1,
  "endpointHost": "api.paddlepaddle.org.cn",
  "endpointPath": "/paddleocr/layout-parsing",
  "timeoutMs": 90000,
  "retryPolicy": {
    "maxAttempts": 4,
    "backoff": "exponential"
  },
  "requestPayloadMeta": {
    "sendPdfAs": "base64",
    "payloadBytesApprox": 1365333,
    "pdfSizeKb": 1000
  },
  "networkMeta": {
    "overrideIpState": "missing",
    "proxyPresent": {
      "http": false,
      "https": false,
      "noProxy": false
    },
    "userAgent": "blog-content-import/1.0",
    "dnsStrategy": "undiciDispatcher"
  }
}
```

### 3. OCR Request Success

```json
{
  "ts": "2026-01-28T18:00:15.567Z",
  "level": "info",
  "msg": "OCR request succeeded",
  "module": "pdf_vl_ocr",
  "event": "ocr-request-success",
  "ocrJobId": "ocr-a1b2c3d4",
  "attempt": 1,
  "statusCode": 200,
  "requestId": null,
  "elapsedMs": 15333,
  "responseMeta": {
    "markdownChars": 52341,
    "imagesCount": 12,
    "outputImagesCount": 0
  }
}
```

### 4. OCR Job End

```json
{
  "ts": "2026-01-28T18:00:20.789Z",
  "level": "info",
  "msg": "OCR-VL job completed",
  "event": "span.end",
  "span": "ocr-vl-job",
  "adapter": "pdf_vl",
  "stage": "ocr",
  "ocrJobId": "ocr-a1b2c3d4",
  "endTs": "2026-01-28T18:00:20.789Z",
  "durationMs": 20666,
  "resultSummary": {
    "pagesProcessed": 1,
    "imagesCount": 12,
    "markdownChars": 52341,
    "retries": 0,
    "finalStatus": "success"
  }
}
```

## Example: Failed OCR Import (Invalid IP)

This example shows the "Invalid IP address: undefined" error and how to diagnose it.

### 1. OCR Job Start

```json
{
  "ts": "2026-01-28T18:10:00.123Z",
  "level": "info",
  "msg": "OCR-VL job starting",
  "event": "span.start",
  "span": "ocr-vl-job",
  "ocrJobId": "ocr-f1e2d3c4",
  "sourceUrl": "https://example.com/paper.pdf",
  "pdfBytes": 1024000,
  "ocrProvider": "paddle-ocr-vl",
  "startTs": "2026-01-28T18:10:00.123Z"
}
```

### 2. OCR Request Preparation

```json
{
  "ts": "2026-01-28T18:10:00.234Z",
  "level": "info",
  "msg": "OCR request starting",
  "module": "pdf_vl_ocr",
  "event": "ocr-request-prepare",
  "ocrJobId": "ocr-f1e2d3c4",
  "attempt": 1,
  "endpointHost": "api.paddlepaddle.org.cn",
  "endpointPath": "/paddleocr/layout-parsing",
  "networkMeta": {
    "overrideIpState": "invalid",
    "proxyPresent": { "http": false, "https": false, "noProxy": false },
    "dnsStrategy": "undiciDispatcher"
  }
}
```

**🔍 CLUE #1**: `overrideIpState: "invalid"` indicates that an IP override was configured but invalid.

### 3. OCR Request Failure

```json
{
  "ts": "2026-01-28T18:10:00.345Z",
  "level": "error",
  "msg": "OCR request failed",
  "module": "pdf_vl_ocr",
  "event": "ocr-request-fail",
  "ocrJobId": "ocr-f1e2d3c4",
  "attempt": 1,
  "statusCode": 0,
  "errorCode": "OCR_NET_INVALID_IP",
  "errName": "TypeError",
  "errMessage": "fetch failed",
  "undiciCauseName": "TypeError",
  "undiciCauseCode": "ERR_INVALID_IP_ADDRESS",
  "causeSummary": "TypeError: ERR_INVALID_IP_ADDRESS",
  "endpointHost": "api.paddlepaddle.org.cn",
  "elapsedMs": 111,
  "shouldRetry": false,
  "networkMeta": {
    "overrideIpState": "invalid",
    "proxyPresent": { "http": false, "https": false, "noProxy": false }
  }
}
```

**🔍 CLUE #2**: `errorCode: "OCR_NET_INVALID_IP"` tells us this is an invalid IP error.
**🔍 CLUE #3**: `undiciCauseCode: "ERR_INVALID_IP_ADDRESS"` confirms the underlying cause.
**🔍 CLUE #4**: `shouldRetry: false` indicates this is not a retryable error.

### 4. Final Error

```json
{
  "ts": "2026-01-28T18:10:00.456Z",
  "level": "error",
  "msg": "OCR API call failed permanently",
  "module": "pdf_vl_ocr",
  "stage": "final_error",
  "ocrJobId": "ocr-f1e2d3c4",
  "attempt": 1,
  "maxAttempts": 4,
  "statusCode": 0,
  "errorCode": "OCR_NET_INVALID_IP",
  "error": {
    "name": "OcrApiError",
    "message": "Failed to call PaddleOCR-VL API: fetch failed",
    "cause": {
      "name": "TypeError",
      "message": "Invalid IP address: undefined",
      "code": "ERR_INVALID_IP_ADDRESS"
    }
  }
}
```

**🔍 CLUE #5**: Full error details show "Invalid IP address: undefined" in the cause chain.

## Diagnosing "Invalid IP address: undefined"

When you see this error, follow these steps:

### Step 1: Find the ocrJobId

Look for the `ocrJobId` in any error log (e.g., `ocr-f1e2d3c4`).

### Step 2: Search for OCR request preparation

Search logs for: `ocrJobId: "ocr-f1e2d3c4" AND event: "ocr-request-prepare"`

### Step 3: Check networkMeta fields

Look at these fields in the request preparation log:

- `overrideIpState`: Should be "missing" for normal operation, "enabled" if IP override is valid
- If `overrideIpState: "invalid"`, check environment variables:
  - `PADDLE_OCR_VL_API_IP`
  - `PADDLE_OCR_VL_IP`
  - `PDF_OCR_API_IP`
  - `PADDLEOCR_VL_IP`

### Step 4: Check errorCode

In the failure logs, look for:

- `OCR_NET_INVALID_IP`: Invalid IP address configuration
- `OCR_NET_DNS_FAIL`: DNS resolution failed (ENOTFOUND)
- `OCR_NET_TIMEOUT`: Request timeout
- `OCR_NET_CONNECTION`: Connection failed
- `OCR_HTTP_NON_2XX`: API returned HTTP error

### Step 5: Solution

For "Invalid IP address: undefined":

1. Check if any IP override environment variables are set to `undefined` or empty string
2. Either provide a valid IP address or remove the environment variable completely
3. For GitHub Actions, ensure secrets are properly configured and not undefined

## Error Codes Reference

| Error Code                | Meaning             | Typical Cause                                                   |
| ------------------------- | ------------------- | --------------------------------------------------------------- |
| `OCR_NET_INVALID_IP`      | Invalid IP address  | Environment variable set to `undefined`, empty, or malformed IP |
| `OCR_NET_DNS_FAIL`        | DNS lookup failed   | Domain not found (ENOTFOUND) or DNS timeout (EAI_AGAIN)         |
| `OCR_NET_TIMEOUT`         | Request timeout     | Network slow, server not responding, or abort signal            |
| `OCR_NET_TLS`             | TLS/SSL error       | Certificate issues, TLS handshake failed                        |
| `OCR_NET_CONNECTION`      | Connection failed   | ECONNREFUSED, ECONNRESET, host unreachable                      |
| `OCR_HTTP_NON_2XX`        | HTTP error status   | 4xx client errors, 5xx server errors                            |
| `OCR_RESPONSE_PARSE_FAIL` | JSON parse failed   | Invalid response format                                         |
| `OCR_RESULT_EMPTY`        | Empty result        | API returned no markdown/images                                 |
| `OCR_PDF_FETCH_FAIL`      | PDF download failed | Network error downloading PDF                                   |
| `OCR_PDF_PARSE_FAIL`      | PDF parse failed    | Invalid PDF format                                              |
| `OCR_UNKNOWN`             | Unknown error       | Uncategorized error                                             |

## Log Correlation

To trace a complete OCR job:

1. Find all logs with the same `ocrJobId`
2. Look for these events in order:
   - `event: "span.start"` with `span: "ocr-vl-job"` → Job start
   - `event: "ocr-request-prepare"` → Request preparation (one per attempt)
   - `event: "ocr-request-success"` or `"ocr-request-fail"` → Request result
   - `event: "span.end"` with `span: "ocr-vl-job"` → Job end

3. If retries occurred, you'll see multiple request cycles with incrementing `attempt` numbers

## Security Note

The logging system is designed to never expose sensitive information:

- ❌ Tokens are never logged
- ❌ Full proxy URLs are never logged
- ❌ Full request/response bodies are never logged
- ❌ Complete PDF binary data is never logged
- ✅ Only metadata (sizes, counts, presence flags) is logged
- ✅ Only host/path components of URLs are logged (no query params with tokens)
- ✅ Error snippets are truncated to 2KB max
