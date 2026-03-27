# Next Steps and Recommendations

This document outlines recommendations for scaling, risk mitigation, and stakeholder engagement for the JSON Schema Constraint Enrichment project.

---

## Table of Contents

1. [Current State Summary](#current-state-summary)
2. [Recommended Next Steps](#recommended-next-steps)
3. [Scaling Considerations](#scaling-considerations)
4. [Risk Assessment](#risk-assessment)
5. [Stakeholder Engagement Plan](#stakeholder-engagement-plan)
6. [Roadmap](#roadmap)

---

## Current State Summary

### What's Working

| Component | Status | Performance |
|-----------|--------|-------------|
| Schema Enrichment Pipeline | ✅ Complete | Fully functional |
| Data Dictionary Parsing | ✅ Complete | CSV support |
| LLM Interpretation (OpenRouter) | ✅ Complete | ~500-2000ms latency |
| T5 Local Inference | ✅ Complete | ~50-100ms latency, 83% success rate |
| Hybrid Interpreter (T5 + LLM fallback) | ✅ Complete | Cost-optimized |
| Schema Validation | ✅ Complete | JSON Schema Draft 7 |
| Evaluation Metrics | ✅ Complete | 5 custom metrics |
| Training Data Collection | ✅ Complete | Production logging |
| Test Coverage | ✅ Complete | 155 tests passing |

### Known Limitations

- T5 model struggles with email format rules (skipped test)
- GPU requires CUDA capability 7.0+ (falls back to CPU)
- Single-domain adapters (financial, healthcare)

---

## Recommended Next Steps

### Phase 1: Immediate (1-2 weeks)

#### 1.1 Improve T5 Model Coverage

**Priority:** High

The T5 model currently achieves 83% success rate. Target: 95%+

**Actions:**
- [ ] Add 50+ email/format training examples
- [ ] Add phone number pattern examples
- [ ] Add "must be positive" → `exclusiveMinimum: 0` examples
- [ ] Retrain adapter in Colab
- [ ] Re-enable skipped test

**Training data gaps identified:**
```
"Must be a valid email address" → {"format": "email"}
"Phone must be 10 digits" → {"pattern": "^\\d{10}$"}
"Income must be positive" → {"exclusiveMinimum": 0}
```

#### 1.2 Add Healthcare Adapter

**Priority:** Medium

**Actions:**
- [ ] Train healthcare adapter using `training_data/healthcare_domain.json`
- [ ] Test with `examples/healthcare_schema.json`
- [ ] Validate against medical coding standards (ICD-10, CPT)

#### 1.3 Documentation Review

**Priority:** Medium

**Actions:**
- [ ] Technical review of `docs/deployment.md`
- [ ] Add API documentation (if exposing as service)
- [ ] Create user guide for non-technical stakeholders

---

### Phase 2: Short-term (1-2 months)

#### 2.1 Multi-Domain Adapter

**Priority:** High

Train a single adapter that handles multiple domains.

**Benefits:**
- Simpler deployment (one model)
- Cross-domain learning
- Easier maintenance

**Approach:**
```python
# Combine training data
combined = {
    "domain": "multi",
    "examples": financial_examples + healthcare_examples + retail_examples
}
```

#### 2.2 API Service Layer

**Priority:** Medium

Expose enrichment as a REST API for integration.

**Proposed endpoints:**
```
POST /api/v1/enrich
  Body: { schema: {...}, dictionary: [...] }
  Response: { enriched_schema: {...}, stats: {...} }

GET /api/v1/health
  Response: { status: "ok", model_loaded: true }
```

**Technology options:**
- FastAPI (recommended for async + auto-docs)
- Flask (simpler, synchronous)

#### 2.3 CI/CD Pipeline

**Priority:** Medium

**Actions:**
- [ ] GitHub Actions workflow for tests on PR
- [ ] Automated model evaluation on training data changes
- [ ] Docker image build and push

---

### Phase 3: Medium-term (3-6 months)

#### 3.1 Active Learning Pipeline

**Priority:** High

Automatically improve the model from production usage.

**Workflow:**
```
Production Run → Collect LLM Fallbacks → Human Review → Add to Training → Retrain
```

**Components:**
- [x] Training data collection (implemented)
- [ ] Review UI for approving examples
- [ ] Automated retraining trigger
- [ ] A/B testing for new models

#### 3.2 Enterprise Features

**Priority:** Medium

- [ ] Multi-tenant support
- [ ] Audit logging
- [ ] Role-based access control
- [ ] SSO integration

#### 3.3 Additional Input Formats

**Priority:** Low

- [ ] Excel data dictionaries (.xlsx)
- [ ] Database metadata (JDBC introspection)
- [ ] OpenAPI/Swagger schemas
- [ ] Protobuf definitions

---

## Scaling Considerations

### Horizontal Scaling

| Component | Scaling Strategy |
|-----------|------------------|
| API Service | Kubernetes pods with HPA |
| T5 Inference | GPU nodes or CPU autoscaling |
| LLM Fallback | Rate limiting per API key |

### Performance Optimization

| Current | Target | Action |
|---------|--------|--------|
| T5 CPU: 50-100ms | <30ms | Batch inference, ONNX conversion |
| LLM: 500-2000ms | <500ms | Caching common rules |
| Memory: 2GB | <1GB | Quantized T5 model (int8) |

### Cost Optimization

| Scenario | Monthly Cost Estimate |
|----------|----------------------|
| 100% LLM (10k rules/day) | $50-200 |
| 80% T5 + 20% LLM | $10-40 |
| 95% T5 + 5% LLM | $2-10 |

**Recommendation:** Target 95% T5 success rate to minimize API costs.

### Infrastructure Options

| Option | Pros | Cons |
|--------|------|------|
| **Cloud VM** | Simple, full control | Manual scaling |
| **Kubernetes** | Auto-scaling, resilient | Complex setup |
| **Serverless (Lambda)** | Pay-per-use | Cold start latency, model loading |
| **Edge (on-prem)** | Data privacy, low latency | Hardware management |

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| T5 model produces invalid constraints | Medium | High | Schema validation catches errors; LLM fallback |
| LLM API downtime | Low | High | Local T5 as primary; cache common rules |
| Model drift over time | Medium | Medium | Regular evaluation against test set |
| GPU compatibility issues | Medium | Low | CPU fallback implemented |
| Training data quality | Medium | High | Human review of collected examples |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API key exposure | Low | High | Environment variables, secrets management |
| Large schema processing | Medium | Medium | Chunking, timeout handling |
| Dependency vulnerabilities | Medium | Medium | Regular `pip audit`, dependabot |

### Business Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Low adoption | Medium | High | Stakeholder demos, pilot program |
| Incorrect constraints in production | Low | Critical | Validation + human review workflow |
| Regulatory compliance (healthcare/finance) | Medium | High | Audit logging, explainability |

### Risk Matrix

```
        │ Low Impact │ Med Impact │ High Impact │
────────┼────────────┼────────────┼─────────────┤
High    │            │            │             │
Likely  │            │            │             │
────────┼────────────┼────────────┼─────────────┤
Medium  │ GPU compat │ Model drift│ T5 invalid  │
Likely  │            │ Large files│ Data quality│
────────┼────────────┼────────────┼─────────────┤
Low     │            │            │ LLM downtime│
Likely  │            │            │ API key leak│
────────┴────────────┴────────────┴─────────────┘
```

---

## Stakeholder Engagement Plan

### Stakeholder Map

| Stakeholder | Interest | Influence | Engagement Strategy |
|-------------|----------|-----------|---------------------|
| **Data Engineers** | High | High | Technical demos, hands-on training |
| **Data Architects** | High | High | Architecture review, standards alignment |
| **Business Analysts** | Medium | Medium | Show business rule preservation |
| **Compliance/Legal** | Medium | High | Audit capabilities, explainability |
| **IT Operations** | Medium | Medium | Deployment docs, runbooks |
| **Management** | Low | High | ROI metrics, executive summary |

### Engagement Activities

#### Week 1-2: Awareness

- [ ] Send project overview email to stakeholders
- [ ] Schedule 30-min demo for data engineering team
- [ ] Share GitHub repo with technical stakeholders

#### Week 3-4: Pilot Program

- [ ] Identify 2-3 pilot use cases
- [ ] Work with data engineers to run on real schemas
- [ ] Collect feedback on accuracy and usability

#### Month 2: Validation

- [ ] Present pilot results to stakeholders
- [ ] Address concerns and feature requests
- [ ] Get sign-off for production deployment

#### Month 3+: Rollout

- [ ] Training sessions for end users
- [ ] Office hours for questions
- [ ] Monthly metrics review with stakeholders

### Communication Plan

| Audience | Format | Frequency | Content |
|----------|--------|-----------|---------|
| Technical team | Slack/Teams | Daily | Progress, blockers |
| Data architects | Meeting | Weekly | Design decisions |
| Management | Email report | Bi-weekly | Metrics, milestones |
| All stakeholders | Newsletter | Monthly | Features, adoption |

### Success Metrics for Stakeholders

| Metric | Target | Measurement |
|--------|--------|-------------|
| Schema enrichment accuracy | >95% | Evaluation suite |
| Time saved per schema | 2-4 hours | Before/after comparison |
| Adoption rate | 80% of new schemas | Usage tracking |
| User satisfaction | >4/5 rating | Survey |

---

## Roadmap

### Q1: Foundation

```
Week 1-2    Week 3-4    Week 5-6    Week 7-8
────────────────────────────────────────────
[Improve T5 model coverage          ]
[Train healthcare adapter   ]
        [Documentation review       ]
                [Pilot program      ]
                        [Feedback collection]
```

### Q2: Scale

```
Month 1         Month 2         Month 3
────────────────────────────────────────
[Multi-domain adapter       ]
[API service development    ]
        [CI/CD pipeline     ]
                [Production rollout ]
```

### Q3-Q4: Enterprise

```
Month 1-2       Month 3-4       Month 5-6
────────────────────────────────────────
[Active learning pipeline           ]
[Enterprise features        ]
                [Additional formats ]
                        [Scale & optimize]
```

---

## Appendix: Quick Wins

Actions that can be completed in <1 day with high impact:

1. **Add 20 email format examples** to training data
2. **Create `.gitignore`** for generated files
3. **Add GitHub Actions** for running tests on PR
4. **Write 1-page executive summary** for management
5. **Record 5-min demo video** for async sharing

---

## Appendix: Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-03-16 | Use T5 with LLM fallback | Cost optimization, latency | LLM-only, T5-only |
| 2026-03-16 | Custom evaluation metrics | Task-specific needs | DeepEval library |
| 2026-03-16 | Skip format test | Training data gap | Fix in model |

---

## Contact

For questions about this document or the project:

- **Repository:** https://github.com/carlhuxley/constraints_2_json
- **Issues:** https://github.com/carlhuxley/constraints_2_json/issues
