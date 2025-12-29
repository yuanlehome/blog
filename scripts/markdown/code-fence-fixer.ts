/**
 * Code Fence Language Detection Module
 *
 * Infers programming languages for code blocks without language annotations.
 * Uses heuristics based on shebang, keywords, and file patterns.
 */

export interface LanguagePattern {
  language: string;
  patterns: {
    shebang?: RegExp[];
    keywords?: RegExp[];
    filePatterns?: RegExp[];
    structurePatterns?: RegExp[];
  };
  priority?: number; // Higher priority checked first
}

/**
 * Language detection patterns
 */
const LANGUAGE_PATTERNS: LanguagePattern[] = [
  // Python
  {
    language: 'python',
    priority: 10,
    patterns: {
      shebang: [/^#!\s*\/.*python[23]?/i],
      keywords: [
        /\bdef\s+\w+\s*\(/,
        /\bclass\s+\w+/,
        /\bimport\s+\w+/,
        /\bfrom\s+\w+\s+import/,
        /\b(elif|async|await)\b/,
        /__init__/,
      ],
    },
  },
  // JavaScript
  {
    language: 'javascript',
    priority: 9,
    patterns: {
      shebang: [/^#!\s*\/.*node/i],
      keywords: [
        /\bfunction\s+\w+\s*\(/,
        /\bconst\s+\w+\s*=/,
        /\blet\s+\w+\s*=/,
        /\bvar\s+\w+\s*=/,
        /\b(async|await|Promise)\b/,
        /=>\s*{/,
        /console\.log/,
        /require\s*\(/,
      ],
    },
  },
  // TypeScript
  {
    language: 'typescript',
    priority: 10,
    patterns: {
      keywords: [
        /:\s*(string|number|boolean|any|void|unknown|never)\b/,
        /\binterface\s+\w+/,
        /\btype\s+\w+\s*=/,
        /\benum\s+\w+/,
        /<\w+>/,
        /\bReadonly</,
        /\bPartial</,
      ],
    },
  },
  // Bash/Shell
  {
    language: 'bash',
    priority: 10,
    patterns: {
      shebang: [/^#!\s*\/(bin\/)?bash/i, /^#!\s*\/(bin\/)?sh/i],
      keywords: [
        /\b(echo|cd|ls|mkdir|rm|cp|mv|grep|sed|awk|cat|chmod|chown)\s/,
        /\$\{?\w+\}?/,
        /\bif\s+\[\[/,
        /\bthen\b/,
        /\bfi\b/,
        /\bdo\b.*\bdone\b/,
      ],
    },
  },
  // Go
  {
    language: 'go',
    priority: 9,
    patterns: {
      keywords: [
        /^package\s+\w+/m,
        /\bfunc\s+\w+\s*\(/,
        /\bfunc\s+\(\w+\s+\*?\w+\)\s+\w+/,
        /\b(defer|goroutine|chan|select)\b/,
        /\btype\s+\w+\s+struct/,
        /\binterface\s*{/,
        /fmt\.\w+/,
      ],
    },
  },
  // Rust
  {
    language: 'rust',
    priority: 9,
    patterns: {
      keywords: [
        /\bfn\s+\w+\s*\(/,
        /\blet\s+mut\b/,
        /\b(impl|trait|pub|mod|use)\b/,
        /\bstruct\s+\w+/,
        /\benum\s+\w+/,
        /::\w+/,
        /&mut\b/,
        /\bSome\(|None\b/,
        /\bOk\(|Err\(/,
      ],
    },
  },
  // C++
  {
    language: 'cpp',
    priority: 8,
    patterns: {
      keywords: [
        /^#include\s*[<"]/m,
        /\bstd::/,
        /\b(namespace|template|typename|class|public|private|protected)\b/,
        /\b(vector|string|map|set|unique_ptr|shared_ptr)<\w+>/,
        /\bcout\s*<<|cin\s*>>/,
        /::\w+/,
      ],
    },
  },
  // C
  {
    language: 'c',
    priority: 7,
    patterns: {
      keywords: [
        /^#include\s*[<"]/m,
        /\b(int|char|float|double|void|long|short|unsigned)\s+\w+\s*[=;(]/,
        /\bprintf\s*\(/,
        /\bscanf\s*\(/,
        /\bmalloc\s*\(/,
        /\bfree\s*\(/,
      ],
    },
  },
  // Java
  {
    language: 'java',
    priority: 8,
    patterns: {
      keywords: [
        /\bpublic\s+(class|interface|enum)\s+\w+/,
        /\bprivate\s+\w+\s+\w+/,
        /\bSystem\.out\.print/,
        /\bpublic\s+static\s+void\s+main/,
        /\b(extends|implements|abstract|final)\b/,
        /\bnew\s+\w+\s*\(/,
        /\@\w+/,
      ],
    },
  },
  // YAML (including GitHub Actions)
  {
    language: 'yaml',
    priority: 10,
    patterns: {
      keywords: [
        /^name:\s*.+$/m,
        /^on:\s*$/m,
        /^\s{2,}(runs?-on|steps|uses|with|env|if):/m,
      ],
      structurePatterns: [
        /^\w+:\s*$/m, // Top-level keys
        /^\s{2,}-\s+\w+:/m, // List items with keys
        /^\s{2,}\w+:\s+[^#\n]+$/m, // Indented key-value pairs
      ],
    },
  },
  // JSON
  {
    language: 'json',
    priority: 9,
    patterns: {
      structurePatterns: [
        /^\s*\{/,
        /^\s*\[/,
        /"\w+":\s*[{\[\"]/, // JSON key-value
      ],
    },
  },
  // Dockerfile
  {
    language: 'dockerfile',
    priority: 10,
    patterns: {
      keywords: [
        /^FROM\s+/m,
        /^RUN\s+/m,
        /^COPY\s+/m,
        /^ADD\s+/m,
        /^WORKDIR\s+/m,
        /^ENV\s+/m,
        /^EXPOSE\s+/m,
        /^CMD\s+/m,
        /^ENTRYPOINT\s+/m,
      ],
    },
  },
  // SQL
  {
    language: 'sql',
    priority: 8,
    patterns: {
      keywords: [
        /\b(SELECT|FROM|WHERE|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TABLE|DATABASE)\b/i,
        /\bJOIN\b/i,
        /\bGROUP BY\b/i,
        /\bORDER BY\b/i,
      ],
    },
  },
  // HTML
  {
    language: 'html',
    priority: 7,
    patterns: {
      structurePatterns: [
        /<(!DOCTYPE|html|head|body|div|span|a|p|h[1-6]|ul|ol|li|table|tr|td)/i,
        /<\/\w+>/,
      ],
    },
  },
  // CSS
  {
    language: 'css',
    priority: 7,
    patterns: {
      keywords: [
        /[.#]?\w+\s*\{/,
        /:\s*[^;]+;/,
        /\b(color|background|font|margin|padding|border|display|position)\s*:/,
      ],
    },
  },
];

/**
 * Detect language from code content using patterns
 */
export function detectCodeLanguage(code: string): string {
  const trimmedCode = code.trim();
  if (!trimmedCode) {
    return 'text';
  }

  // Try to detect from first line (shebang)
  const firstLine = trimmedCode.split('\n')[0];
  for (const pattern of LANGUAGE_PATTERNS) {
    if (pattern.patterns.shebang) {
      for (const shebang of pattern.patterns.shebang) {
        if (shebang.test(firstLine)) {
          return pattern.language;
        }
      }
    }
  }

  // Sort patterns by priority (higher first)
  const sortedPatterns = [...LANGUAGE_PATTERNS].sort(
    (a, b) => (b.priority || 0) - (a.priority || 0),
  );

  // Check keywords and structure patterns
  const scores: Record<string, number> = {};

  for (const pattern of sortedPatterns) {
    let score = 0;

    // Check keywords
    if (pattern.patterns.keywords) {
      for (const keyword of pattern.patterns.keywords) {
        if (keyword.test(code)) {
          score += 1;
        }
      }
    }

    // Check structure patterns
    if (pattern.patterns.structurePatterns) {
      for (const structPattern of pattern.patterns.structurePatterns) {
        if (structPattern.test(code)) {
          score += 1;
        }
      }
    }

    if (score > 0) {
      scores[pattern.language] = score;
    }
  }

  // Find language with highest score
  const entries = Object.entries(scores);
  if (entries.length === 0) {
    return 'text';
  }

  entries.sort((a, b) => b[1] - a[1]);
  return entries[0][0];
}

/**
 * Check if a code block appears to be valid GitHub Actions workflow YAML
 */
export function isGitHubActionsWorkflow(code: string): boolean {
  const hasName = /^name:\s*.+$/m.test(code);
  const hasOn = /^on:\s*$/m.test(code) || /^on:\s+\[/m.test(code);
  const hasJobs = /^jobs:\s*$/m.test(code);
  const hasSteps = /^\s{2,}steps:/m.test(code);

  return (hasName || hasOn) && (hasJobs || hasSteps);
}
