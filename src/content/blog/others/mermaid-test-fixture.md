---
title: Mermaid Test Fixture
date: 2024-12-01
tags: [testing, mermaid]
status: draft
---

## Flowchart Example

```mermaid title="Simple Flowchart"
graph TD
  A[Start] --> B{Decision}
  B -->|Yes| C[Do it]
  B -->|No| D[Skip]
  C --> E[End]
  D --> E
```

Some text after the mermaid diagram.

## Sequence Diagram

```mermaid
sequenceDiagram
  Alice->>Bob: Hello Bob!
  Bob-->>Alice: Hi Alice!
```
