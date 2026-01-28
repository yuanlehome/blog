/**
 * PaddleOCR-VL API Client
 *
 * Handles communication with PaddleOCR-VL layout parsing API
 */

import type { Logger } from '../../logger/types.js';

/**
 * OCR result structure
 */
export interface PaddleOcrVlResult {
  markdown: string;
  images: Record<string, string>; // img_path -> img_url
  outputImages?: string[];
}

/**
 * PaddleOCR-VL API response structure
 */
interface PaddleOcrVlResponse {
  result?: {
    layoutParsingResults?: Array<{
      markdown?: {
        text?: string;
        images?: Record<string, string>;
      };
      outputImages?: string[];
    }>;
  };
  error?: string;
  message?: string;
}

/**
 * Custom error for OCR API failures
 */
export class OcrApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public responseBody?: string,
    public requestId?: string,
  ) {
    super(message);
    this.name = 'OcrApiError';
  }
}

/**
 * Custom error for OCR response parsing failures
 */
export class OcrParseError extends Error {
  constructor(
    message: string,
    public responseBody?: string,
  ) {
    super(message);
    this.name = 'OcrParseError';
  }
}

/**
 * Call PaddleOCR-VL API to parse PDF
 */
export async function callPaddleOcrVl(
  pdfBuffer: Buffer,
  apiUrl: string,
  token: string,
  logger?: Logger,
): Promise<PaddleOcrVlResult> {
  // Convert PDF to base64
  const base64Pdf = pdfBuffer.toString('base64');

  logger?.debug('Preparing OCR request', {
    module: 'pdf_vl',
    stage: 'ocr',
    base64Length: base64Pdf.length,
  });

  // Prepare request payload
  const payload = {
    file: base64Pdf,
    fileType: 0, // 0 = PDF, 1 = image
    useDocOrientationClassify: false,
    useDocUnwarping: false,
    useChartRecognition: false,
  };

  let response: Response;
  let responseText: string = '';

  try {
    // Make API request
    response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        Authorization: `token ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    responseText = await response.text();

    logger?.debug('OCR API response received', {
      module: 'pdf_vl',
      stage: 'ocr',
      statusCode: response.status,
      responseLength: responseText.length,
      // Don't log full response or token
    });
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : String(error);
    throw new OcrApiError(`Failed to call PaddleOCR-VL API: ${errorMsg}`, 0, errorMsg);
  }

  // Check HTTP status
  if (!response.ok) {
    // Sanitize response body for logging (truncate if too long)
    const bodySample =
      responseText.length > 500 ? responseText.slice(0, 500) + '...' : responseText;

    throw new OcrApiError(
      `PaddleOCR-VL API returned error: HTTP ${response.status} ${response.statusText}`,
      response.status,
      bodySample,
    );
  }

  // Parse response
  let parsedResponse: PaddleOcrVlResponse;
  try {
    parsedResponse = JSON.parse(responseText);
  } catch (error) {
    const bodySample =
      responseText.length > 500 ? responseText.slice(0, 500) + '...' : responseText;
    throw new OcrParseError(
      `Failed to parse PaddleOCR-VL API response as JSON: ${error instanceof Error ? error.message : String(error)}`,
      bodySample,
    );
  }

  // Check for API-level errors
  if (parsedResponse.error) {
    throw new OcrApiError(
      `PaddleOCR-VL API error: ${parsedResponse.error}`,
      response.status,
      parsedResponse.message,
    );
  }

  // Extract result
  const layoutResults = parsedResponse.result?.layoutParsingResults;
  if (!layoutResults || layoutResults.length === 0) {
    throw new OcrParseError(
      'PaddleOCR-VL API returned empty layoutParsingResults',
      JSON.stringify(parsedResponse).slice(0, 500),
    );
  }

  const firstResult = layoutResults[0];
  const markdown = firstResult.markdown?.text;
  const images = firstResult.markdown?.images;

  if (!markdown) {
    throw new OcrParseError(
      'PaddleOCR-VL API did not return markdown text',
      JSON.stringify(firstResult).slice(0, 500),
    );
  }

  logger?.info('OCR parsing successful', {
    module: 'pdf_vl',
    stage: 'ocr',
    markdownLength: markdown.length,
    imageCount: images ? Object.keys(images).length : 0,
  });

  return {
    markdown,
    images: images || {},
    outputImages: firstResult.outputImages,
  };
}
