---
title: 'Mermaid 渲染链路验收样例（时序图）'
date: 2026-03-01
tags: [mermaid, rendering, validation]
status: draft
source: original
---

```mermaid title="Long sequence diagram validation" sequenceMode=loose scale=1.2 width=1200 fontSize=14 wrap=true
sequenceDiagram
    participant WebClientWithVeryLongName as Web Client With Very Long Display Name
    participant ApiGatewayCoreService as API Gateway Core Service Layer
    participant OrderOrchestrationEngine as Order Orchestration Engine Component
    participant SettlementAndRiskSystem as Settlement & Risk Control Subsystem

    WebClientWithVeryLongName->>ApiGatewayCoreService: Submit order payload with extensive labels and metadata for anti-fraud review
    ApiGatewayCoreService->>OrderOrchestrationEngine: Validate and dispatch workflow request with correlation id and tracing context
    Note over OrderOrchestrationEngine,SettlementAndRiskSystem: A long note to verify wrapping and readability in both light and dark theme outputs
    OrderOrchestrationEngine->>SettlementAndRiskSystem: Execute settlement pre-check and reserve account balance for downstream confirmation
    SettlementAndRiskSystem-->>WebClientWithVeryLongName: Respond with structured status including human readable diagnostics
```
