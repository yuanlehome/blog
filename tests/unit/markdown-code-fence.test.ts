/**
 * Tests for Code Fence Language Detection
 */

import { describe, it, expect } from 'vitest';
import {
  detectCodeLanguage,
  isGitHubActionsWorkflow,
} from '../../scripts/markdown/code-fence-fixer';

describe('Code Fence Fixer', () => {
  describe('detectCodeLanguage', () => {
    it('should detect Python from shebang', () => {
      const code = `#!/usr/bin/env python3
print("Hello World")`;
      expect(detectCodeLanguage(code)).toBe('python');
    });

    it('should detect Python from keywords', () => {
      const code = `def hello_world():
    import sys
    print("Hello")
    return True`;
      expect(detectCodeLanguage(code)).toBe('python');
    });

    it('should detect Bash from shebang', () => {
      const code = `#!/bin/bash
echo "Hello World"
cd /tmp`;
      expect(detectCodeLanguage(code)).toBe('bash');
    });

    it('should detect Bash from keywords', () => {
      const code = `if [[ -f "$file" ]]; then
  echo "File exists"
  rm $file
fi`;
      expect(detectCodeLanguage(code)).toBe('bash');
    });

    it('should detect C++ from includes and namespace', () => {
      const code = `#include <iostream>
using namespace std;

int main() {
    vector<int> numbers;
    cout << "Hello" << endl;
    return 0;
}`;
      expect(detectCodeLanguage(code)).toBe('cpp');
    });

    it('should detect YAML from structure', () => {
      const code = `name: my-app
version: 1.0.0
dependencies:
  - package1
  - package2`;
      expect(detectCodeLanguage(code)).toBe('yaml');
    });

    it('should detect JSON from structure', () => {
      const code = `{
  "name": "my-app",
  "version": "1.0.0",
  "dependencies": {
    "package1": "^1.0.0"
  }
}`;
      expect(detectCodeLanguage(code)).toBe('json');
    });

    it('should detect Dockerfile from keywords', () => {
      const code = `FROM node:18
WORKDIR /app
COPY package.json .
RUN npm install
CMD ["npm", "start"]`;
      expect(detectCodeLanguage(code)).toBe('dockerfile');
    });

    it('should return text for unrecognizable code', () => {
      const code = `This is just plain text
without any recognizable patterns
123 456 789`;
      expect(detectCodeLanguage(code)).toBe('text');
    });

    it('should return text for empty code', () => {
      expect(detectCodeLanguage('')).toBe('text');
      expect(detectCodeLanguage('   ')).toBe('text');
    });
  });

  describe('isGitHubActionsWorkflow', () => {
    it('should detect GitHub Actions workflow', () => {
      const code = `name: CI
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm test`;

      expect(isGitHubActionsWorkflow(code)).toBe(true);
    });

    it('should detect workflow without explicit name', () => {
      const code = `on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3`;

      expect(isGitHubActionsWorkflow(code)).toBe(true);
    });

    it('should return false for regular YAML', () => {
      const code = `version: 1.0
config:
  database: postgres
  port: 5432`;

      expect(isGitHubActionsWorkflow(code)).toBe(false);
    });

    it('should return false for non-workflow content', () => {
      const code = `This is not a workflow file`;
      expect(isGitHubActionsWorkflow(code)).toBe(false);
    });
  });
});
