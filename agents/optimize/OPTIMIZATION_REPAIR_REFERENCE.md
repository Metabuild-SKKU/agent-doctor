# Optimization Repair Reference

> Audience: coding agents and maintainers implementing the Optimize pipeline.
>
> Snapshot: 2026-07-23, branch `feature/optimization-label-coverage`.
>
> Read the repository `AGENTS.md` and this directory's `AGENTS.md` first. This
> file is a code-audit snapshot and implementation checklist; it does not
> override either contract. When this file and code disagree, verify the code
> and tests and update this file in the same change.

## 1. Objective

Make every diagnostic outcome reach one honest terminal state:

1. a confirmed, executable prescription is applied and evaluated;
2. an unsupported or inconclusive diagnosis remains non-actionable with a
   precise reason; or
3. a data/evaluation defect is reported as manual work.

"All labels are handled" does **not** mean that the three D-group labels become
automatic config changes. `corpus_gap`, `corpus_gap_partial_hop`, and
`bad_gold_answer` must remain manual.

### Explicit ownership boundary: reranker

Reranker implementation, tuning, and reranker-specific diagnostics are owned by
another team and are excluded from this workstream. Do not implement or refactor
that subsystem from this backlog unless the user explicitly brings it back into
scope. Keep its current gaps visible only as external integration dependencies.

## 2. Non-negotiable contracts

- Keep `def run(state: AgentDoctorState) -> AgentDoctorState` and return the
  same state object on every path.
- Do not modify `graph.py` as part of an Optimize-only repair.
- Eval owns diagnosis and evidence. Optimize must not re-diagnose a finding.
- Apply one causal prescription/config axis at a time, then run the existing
  Index/Eval validation loop.
- Use canonical parameter paths inside Optimize and map them to state only in
  the config-mapping layer.
- A patch is not "applied" unless all of the following are true:
  - its value is concrete and constraint-valid;
  - the mapper produced a non-empty expected diff;
  - the real Eval/Index/RAG/Serve consumer reads the mapped value;
  - the value survives the handoff to the final serving process.
- `status="applied"` means "changed and awaiting/undergoing validation", not
  "quality improved".
- Preserve fallback behavior, but surface whether a fallback made a requested
  feature a no-op.

## 3. Verified current execution path

```text
DiagnosticReport.findings
  -> planner.plan()
     -> ready + confirmed findings only
     -> label ranking
     -> PrescriptionCandidate/search_space
  -> optimizer.run()
     -> backend/path/capability/constraint filtering
     -> ConfigPatch
  -> config_mapper.apply_config_patch()
     -> state.index_config
  -> history pending item
  -> Index -> Eval
  -> history.judge()
     -> keep or rollback + blacklist
```

The orchestration, history, rollback, blacklist, internal single-axis sweep,
chunk pre-screening, and RAGBuilder adapter are real implementations. Several
older Markdown sections still describe them as missing; do not use those stale
statements as code truth.

## 4. Severity-ordered defect inventory

### P0-1. Hybrid prescription is blocked and has a latent value inversion

Evidence:

- `rules.py` uses `{"use_hybrid": True}`.
- Planner canonicalizes the key and currently produces
  `{"retriever.search_type": [True]}`.
- `optimizer.DEFAULT_CAPABILITIES["hybrid_search"]` is `False`, so the normal
  request is skipped as `unsupported_capability`.
- If that gate is merely changed to `True`,
  `config_mapper.map_canonical_change()` interprets only the literal string
  `"hybrid"` as enabled. Boolean `True` maps to `use_hybrid=False`.
- `AgentDoctorState` defaults to `use_hybrid=True`, while the rule comment
  assumes a dense-only baseline.

Required repair:

1. Express the rule canonically:
   `{"retriever.search_type": "hybrid"}`.
2. Define an allowed value domain such as `["dense", "hybrid"]`.
3. Make reads and writes round-trip:
   `use_hybrid=True <-> "hybrid"` and `False <-> "dense"`.
4. Enable the capability only after the contract test and consumer test pass.
5. Make the diagnostic mode-aware. If the current mode is already hybrid,
   `retrieval_lexical_mismatch` must not prescribe "enable hybrid" again;
   it needs a fusion-weight/candidate-depth prescription or a different cause.

### External dependency. Reranker Optimize integration

Evidence:

- `Retriever` reads `use_reranker` and `reranker_model`.
- `qdrant_store.rerank()` loads and runs a `CrossEncoder`.
- `rules.py` marks `retrieval_low_rank` ready with `enable_reranker`.
- `config_mapper` can read `reranker.enabled` aliases but cannot write them.
- `optimizer` excludes `reranker.enabled` from state-mappable/rules paths and
  sets the reranker capability to `False`.
- Existing tests explicitly assert that reranker mapping is ignored.

This workstream must not implement those fixes. When the owning team delivers
them, accept the integration only if their change provides:

- canonical state mapping and safe defaults;
- truthful requested/applied/fallback execution details;
- pre/post candidate/rank evidence needed by Eval;
- a resolved default model rather than the literal model name `"None"`;
- end-to-end tests proving Optimize, Eval, and restarted Serve use the result.

Until that contract lands, keep reranker prescriptions non-actionable and mark
their reason as `external_dependency`.

### P0-3. `applies_when` is stored but never evaluated

Evidence:

- `rules.py` uses `topic_cluster` conditions for
  `retrieval_semantic_mismatch`.
- `PrescriptionCandidate.applies_when` exists.
- Planner copies the dictionary into candidates.
- No planner function compares it with `Finding.metadata`.
- Optimizer explicitly treats condition evaluation as planner responsibility.

Impact:

- Mutually exclusive semantic-mismatch prescriptions are tried in list order.
- A blocked embedding candidate can fall through to chunk shrinking even when
  the intended signal says the embedding model is the cause.
- `topic_cluster` is not currently produced by Eval, so even a new condition
  evaluator needs a defined missing-signal policy.

Required repair:

1. Define exact matching semantics:
   - scalar expected value;
   - list of accepted values;
   - behavior across multiple findings;
   - behavior when the key is missing.
2. Prefer `any finding matches` for probe-local evidence and require explicit
   aggregation for label-level evidence.
3. Make missing required evidence ineligible rather than silently matching.
4. Add focused planner tests before changing any rule to rely on the feature.

### P0-4. The initial optimization report can describe the wrong prescription

Evidence:

- Optimizer may skip earlier candidates and select a later one.
- Agent stores the actual `result.selected_candidate` in history.
- Reporter derives the displayed prescription, patch, and trade-offs from
  `request.candidates[0]`.
- Agent discards the `ConfigDiff` returned by `apply_config_patch()` and calls
  the reporter without the actual result/diff.

Required repair:

1. Capture the returned `ConfigDiff`.
2. Verify that expected keys changed or were added before setting
   `state.status="applied"`.
3. Pass `OptimizationResult.selected_candidate`, concrete `ConfigPatch`, and
   `ConfigDiff` to the reporter.
4. Report ignored keys and warnings as failures/inconclusive results, not as a
   successful application.

### P0-5. Runtime optimization is not reliably preserved in Serve

Evidence:

- Runtime-only changes set `reindex_required=False`.
- Index then returns early and does not refresh chunk metadata.
- Eval is correct because it passes `state.index_config` explicitly.
- Serve serializes only chunks, not the final config.
- The API rebuilds the retriever with Qdrant connection settings and chunk
  metadata, so it can restore stale hybrid/reranker/top-k values.
- `/search` and `/answer` default to `top_k=3`, overriding an optimized default.
- If the API process is already healthy, Serve does not reload the newly
  written artifact.

Required repair:

1. Persist a versioned serving artifact containing both chunks and final
   retrieval/generation config.
2. Let request `top_k` be optional; when absent, use the optimized retriever
   default.
3. Add an explicit reload/version handshake or safely restart the local API.
4. Test behavior after process restart, not only within the same Python process.
5. Remove the MCP client's unconditional `top_k=3` as well as the FastAPI
   endpoint default.

### P0-6. A successful RAGBuilder result cannot be applied

Evidence:

- `_run_ragbuilder()` returns a validated `best_config` but no `config_patch`.
- Agent applies only a `status="proposed"` result that has a non-null
  `config_patch`.
- `config_mapper.apply_best_config()` exists but is not used by Agent.

Impact:

- Even a manually constructed RAGBuilder request can finish successfully and
  leave `state.index_config` unchanged.
- The adapter is also unreachable in normal planning because Planner selects
  only `rules` or `internal`.

Required repair:

1. Require every proposed backend result to include a concrete `ConfigPatch`.
2. Build that patch from the validated RAGBuilder best config and run the same
   diff/no-op checks used by rules/internal results.
3. Define an explicit backend-selection input; do not silently choose a
   surrogate backend.
4. Supply `input_source` and an evaluation dataset when RAGBuilder is selected.
5. Add a real Agent integration test that proves the returned config changes
   state and is re-evaluated.

### P1-1. Rules and diagnosis taxonomy are not closed

Current counts:

- 25 labels in `LABEL_TO_PRESCRIPTIONS`;
- 17 distinct labels can be emitted by `diagnose.py`;
- one emitted roll-up has no rule: `generation_failure`;
- nine rule labels have no live diagnostic producer.

Rule-only labels:

- `chunking_overchunking`
- `chunking_underchunking`
- `reranker_low_recall`
- `reranker_low_precision`
- `generation_contradiction`
- `generation_misinterpretation`
- `generation_abstention_failure`
- `generation_parametric_overreliance`
- `generation_numerical_error`

`generation_failure` is an unconfirmed low-tier roll-up. It should normally
trigger deeper diagnosis, not a guessed automatic prescription.

The default diagnostic mode makes this mismatch more visible. `DEFAULT_MODE`
is FAST, while Planner discards every unconfirmed finding. In normal FAST
execution the only ready label that is routinely confirmed is
`retrieval_incomplete_enumeration`; low-rank/lexical/semantic need STANDARD,
context causes need FULL, and generation causes need DEEP. A failed report with
no confirmed actionable finding currently falls through to Serve as skipped.

Required repair:

- Define an optimization-ready diagnostic policy. Options include a bounded
  escalation to the required tier or an explicit
  `needs_deeper_diagnosis` terminal state/report.
- Do not auto-apply preliminary findings merely to increase coverage.

Required repair:

- Add a taxonomy contract test with an explicit allowlist for non-actionable
  roll-ups.
- A label may become ready only when its producer, evidence metadata, and
  downstream consumer are all tested.

### P1-2. Readiness is label-wide although support is prescription-specific

Examples:

- `retrieval_incomplete_enumeration` is ready because `top_k` works, while its
  MMR and adaptive-retrieval prescriptions do not.
- `lost_in_the_middle` is entirely draft even though its second prescription
  (`decrease_top_k`) uses a supported path.
- `too_long_context` is ready; unsupported context compression is left for the
  optimizer to skip.

Required repair:

- Add per-prescription readiness/prerequisites, or derive readiness from the
  parameter registry.
- Keep label readiness as "at least one executable prescription", not as the
  status copied onto every candidate.
- Surface unsupported later prescriptions as deferred work rather than
  blacklisting them as failed experiments.

### P1-3. Chunk strategy contract is inconsistent

Evidence:

- Rules use `chunking_strategy="recursive_sentence"`.
- Mapper does not write the path and does not include the real state alias
  `chunk_strategy`.
- Optimizer allows only `recursive_sentence`.
- Index accepts `fixed`, `markdown`, `recursive`, and `markdown_recursive`.

Required repair:

- Canonical path: `chunker.strategy`.
- State key: `chunk_strategy`.
- Allowed values must be taken from the Index strategy registry or a public
  capability function.
- Use `recursive`, not the RAGBuilder-specific name
  `recursive_sentence`, for the internal pipeline.
- Keep the RAGBuilder adapter responsible for external name translation.

### P1-4. Generation prescriptions have no state-to-consumer path

Evidence:

- `AgentDoctorState` has no `generation_config`.
- Eval calls `generate_answer()` without a config.
- Provider functions hardcode temperature zero.
- Prompt construction is fixed.
- Serve persists no generation config.
- Several generation prescriptions change multiple keys, while optimizer
  rejects every multi-axis search space.

Required repair:

1. Add an explicit `generation_config` state contract and ownership.
2. Pass it through Eval, RAG generation, and Serve.
3. Make provider/model/temperature/prompt policy real consumers.
4. Split independent multi-key prescriptions into atomic trials.
5. For settings that must be atomic (for example a verifier plus its type),
   introduce one typed policy value rather than bypassing the single-axis rule.

### P1-5. Validation and rollback can claim confidence without enough evidence

Evidence:

- Missing before/after reports are treated as `keep=True`.
- History floors are marked temporary.
- The floor uses `answer_relevancy`, while Eval publishes
  `response_relevancy`.
- Only metrics present in the report are checked, so a missing guardrail is not
  distinguished from a passing guardrail.
- General history uses any strict score increase; internal search separately
  supports `min_delta`.

Required repair:

- Missing validation data should be `inconclusive` and normally rollback or
  retain pending state, not become a verified keep.
- Share metric aliases and thresholds with Eval or a common policy module.
- Require the expected guardrail metrics for a trial profile.
- Define a consistent `min_delta` policy to avoid retaining stochastic noise.

### P1-6. Lexical evidence is too weak for a causal hybrid prescription

Evidence:

- `_bm25_hits_gold` calls the dependency-light token-overlap
  `keyword_search()`, not a real BM25 implementation.
- Retrieval fallback can silently change dense/hybrid requests into keyword
  search.

Required repair:

- Rename the signal to lexical fallback or implement actual BM25.
- Store retrieval mode, fallback reason, and stage ranks in `EvalRecord`.
- Do not label a vector/hybrid defect when the tested stage did not actually
  run.

Pre/post reranker observability remains an external-team deliverable, not an
implementation task in this workstream.

### P1-7. Retrieval cause precedence makes some ready labels misleading

Evidence:

- The retrieval slot picks the first confirmed cause.
- In STANDARD mode, lexical/semantic signals precede
  `retrieval_missing_gold`; keyword success chooses lexical and keyword failure
  plus corpus membership chooses semantic. Confirmed missing-gold is therefore
  mostly shadowed in the normal fully provisioned path.
- `retrieval_incomplete_enumeration` is confirmed using a broad cardinality
  heuristic and appears before bridge/low-rank/lexical/semantic causes. It can
  claim multi-hop or simple candidate-shortage failures.

Required repair:

- Treat `retrieval_missing_gold` as a fallback/unknown cause or redefine its
  unique evidence.
- Require explicit enumeration question shape and expected cardinality.
- Confirm enumeration via a controlled top-k counterfactual; cardinality alone
  should be preliminary.
- Add precedence tests for enumeration vs bridge, low-rank, semantic, and
  corpus-gap cases.

### P1-8. Target metrics are metadata, not decision criteria

Evidence:

- Rules and requests carry target metrics.
- Planner sets `primary_metric="overall_score"` for every request.
- General history keeps a change on any overall-score increase plus global
  floors; it does not require the target metric to improve.
- Missing target metrics are not inconclusive.
- An internal sweep that ends on the current last candidate may return
  `verified`; graph routing then serves even if the report is still below the
  threshold and other prescriptions remain.

Required repair:

- Use the label's target metric as the primary objective and overall score as a
  global guardrail, or document and test another explicit policy.
- Require target-metric availability and a calibrated `min_delta`.
- After an internal study finishes without meeting the gate, return to the
  normal Planner loop when budget remains instead of serving solely because
  the best candidate was already evaluated.

### P1-9. Fallbacks are evaluated as if the requested stage ran

Current silent fallbacks include:

- model load failure -> deterministic hash embedding;
- Qdrant/setup/search failure -> keyword retrieval;
- reranker load failure -> original order;
- generation/provider failure -> extractive top-context answer;
- per-document Index failure -> partial corpus.

Eval currently loses most of this provenance. A dense/hybrid/model/reranker or
generation trial can therefore be scored and blacklisted even though the
requested component never ran.

Required repair:

- Introduce structured retrieval/generation/index execution attestations.
- Mark a trial `inconclusive` when its target component was not applied or the
  indexed corpus is partial.
- Do not blacklist a prescription for an infrastructure/model-download
  failure.

### P1-10. Parameter types, cost, and attempt history are incomplete

Evidence:

- Numeric constraints accept both integers and floats even where Index requires
  integer top-k/chunk values.
- All rule diagnosis confidences are unset.
- Planner ignores an explicit prescription cost and derives cost only from
  reindex yes/no.
- Candidate priority is always zero.
- Skipped/failed candidates are usually blacklisted or put in `state.error`
  without a full history/report entry.

Required repair:

- Put `int`/`float`/`bool`/enum types in `ParameterSpec` and validate strictly.
- Use explicit cost when present and fallback to reindex cost only when absent.
- Calibrate confidence from Eval signal quality.
- Persist attempted, skipped, inconclusive, and failed results with structured
  reason codes.

### P1-11. Generation/data-error precedence can misclassify failures

Evidence:

- Oracle `bad_gold_answer` is checked before generation causes and becomes
  confirmed from high faithfulness/relevancy alone.
- That can send a faithful but incorrectly reasoned/bound answer to manual
  ground-truth review before checking a generation cause.
- `generation_partial_answer` uses low response relevancy rather than an actual
  completeness test.
- `generation_hop_binding_error` uses a broad multi-hop + high-faithfulness
  condition.
- Correct-but-ungrounded answers can exit through `_no_diagnosis` before a
  parametric-overreliance finding is considered.

Required repair:

- Treat bad-gold as a manual review candidate after automatic generation causes
  have been ruled out.
- Add component coverage for partial answers, hop-level evidence/claim binding,
  numeric/constraint checks, and correct-but-ungrounded additive diagnosis.
- Decide whether generation findings are mutually exclusive; if not, replace a
  single `_pick` result with primary/secondary or additive semantics.

### P1-12. Corpus-gap findings do not yet cover general user questions

Evidence:

- Corpus membership checks gold chunk IDs, not semantic evidence sufficiency.
- Auto/taxonomy probes are re-synchronized to current chunks, so this check is
  strongest at finding stale/missing references.
- User-log questions without trusted ground truth/gold IDs can exit without a
  corpus-gap diagnosis.
- Partial-hop findings do not include concrete `missing_gold_ids` or
  `missing_hops` metadata even though the manual action needs that detail.

Required repair:

- Use trusted document/span evidence for external QA datasets.
- Add hop-level evidence state and missing IDs/hops to Finding metadata.
- Do not claim a general user-log corpus gap without an evidence-sufficiency or
  grounded-abstention evaluation.

### P2-1. Implemented scaffolds that normal pipeline selection cannot reach

| Scaffold | Current state | Required integration |
| --- | --- | --- |
| `propose_only` | schema and reporter branch exist; planner always sets `False` | define state/API/UI input and route contract |
| RAGBuilder | substantial adapter and optimizer dispatch exist; planner only selects `rules` or `internal` | define explicit backend policy and supply input source/eval dataset |
| AutoRAG | backend literal/branch exists; optimizer returns not implemented | either build an adapter or remove it from user-visible choices |
| generation config | public generator accepts a generic config argument | define typed settings and make providers/prompt consume them |
| AspectCritic contradiction | evaluator helper and record field exist | invoke it in live Eval and enable the diagnostic slot |
| query rewrite/decomposition | rules and a diagnostic ablation exist | implement a shared query-planning stage used by Eval and Serve |
| MMR/adaptive retrieval | rule entries exist | implement retrieval policy and observable details |
| context compression/order/filter | diagnostic ablations/rules partly exist | implement the same runtime transformation in RAG and Serve |

`generation_abstention_failure` has a particularly small diagnostic starting
point: Eval already knows whether an answer should not exist and can detect a
non-abstaining answer. A trusted no-answer probe can produce a deterministic
finding before adding LLM-based generation diagnoses.

## 5. Canonical parameter inventory

The target state below is the recommended contract. "Current" describes the
2026-07-23 code, not the desired feature claim.

| Canonical path | State target | Consumer | Reindex | Current state |
| --- | --- | --- | --- | --- |
| `retriever.top_k` | `index_config.top_k` | Eval/Retriever | no | works in Eval; Serve default/persistence wrong |
| `retriever.search_type` | `index_config.use_hybrid` | Retriever | no | consumer exists; optimizer disabled; boolean rule bug |
| `retriever.hybrid_dense_weight` | `index_config.hybrid_dense_weight` | Retriever | no | consumer exists; no Optimize contract |
| `reranker.enabled` | external-team contract | Retriever | no | excluded from this workstream |
| `reranker.model` | external-team contract | Retriever | no | excluded from this workstream |
| `reranker.candidate_count` | external-team contract | Retriever | no | excluded from this workstream |
| `reranker.threshold` | external-team contract | Reranker | no | excluded from this workstream |
| `chunker.chunk_size` | `index_config.chunk_size` | Index | yes | works |
| `chunker.chunk_overlap` | `index_config.chunk_overlap` | Index | yes | works |
| `chunker.strategy` | `index_config.chunk_strategy` | Index | yes | registry exists; names/mapping disagree |
| `embedding.model` | `index_config.embedding_model` | Index/Retriever | yes | consumer and mapper exist; capability disabled |
| `context.compression.enabled` | retrieval/generation policy | RAG | no | read alias only; no implementation |
| `generation.*` | `generation_config` | Generator | no | state/consumer path missing |

Prefer a single `ParameterSpec` registry consumed by mapper and optimizer:

```python
ParameterSpec(
    canonical_path="retriever.search_type",
    target="index_config",
    state_key="use_hybrid",
    value_type=str,
    allowed_values=("dense", "hybrid"),
    capability="hybrid_search",
    reindex_required=False,
)
```

The registry should replace duplicated path/capability/reindex sets where
practical. Rules must reference canonical paths only.

## 6. Label implementation matrix

Legend:

- **partial**: at least one useful path works;
- **blocked**: diagnosis/rule exists, but the intended first path cannot apply;
- **scaffold**: declaration or helper exists without a complete live path;
- **manual**: intentionally not auto-optimized.

| Label | Produced by Eval | Rule status | Effective state | Required work |
| --- | --- | --- | --- | --- |
| `retrieval_low_rank` | yes | ready | external dependency | owning team supplies reranker integration/evidence |
| `retrieval_lexical_mismatch` | yes | ready | blocked | canonical hybrid values, capability, mode-aware signal |
| `retrieval_semantic_mismatch` | yes | ready | partial/misrouted | implement conditions; enable validated embedding candidates; fix strategy |
| `retrieval_missing_gold` | yes | ready | partial | top-k/chunk work; query expansion missing |
| `retrieval_incomplete_enumeration` | yes | ready | partial | top-k works; MMR/adaptive missing |
| `retrieval_missing_bridge_dependency` | yes | draft | scaffold | shared query decomposition and hop config |
| `chunking_context_mismatch` | yes | ready | partial | overlap/size work; strategy contract missing |
| `chunking_overchunking` | no | draft | scaffold | define measurable signal; then reuse chunk-size path |
| `chunking_underchunking` | no | draft | scaffold | define measurable signal; then reuse chunk-size path |
| `reranker_low_recall` | no | draft | external dependency | owning team supplies producer and consumer contract |
| `reranker_low_precision` | no | draft | external dependency | owning team supplies producer and consumer contract |
| `generation_hallucination` | yes | draft | scaffold | generation config and provider/prompt consumers |
| `generation_partial_answer` | yes | draft | scaffold | completeness policy/checklist implementation |
| `generation_contradiction` | no | draft | scaffold | run AspectCritic live; verifier implementation |
| `generation_misinterpretation` | no | draft | declaration only | diagnostic criterion and restatement policy |
| `generation_abstention_failure` | no | draft | declaration only | no-answer probes, abstention policy, citation consumer |
| `generation_parametric_overreliance` | no | draft | declaration only | causal diagnostic and strict-grounding consumer |
| `generation_numerical_error` | no | draft | declaration only | numeric evidence diagnostic and calculation verifier |
| `generation_hop_binding_error` | yes | draft | scaffold | typed hop-evidence answer policy/verifier |
| `too_long_context` | yes | ready | partial | top-k/chunk work; compression missing |
| `lost_in_the_middle` | yes | draft | partial but gated | context ordering missing; top-k path could be prescription-ready |
| `context_noise_interference` | yes | draft | scaffold | filter/MMR/conflict policy consumers |
| `corpus_gap` | yes | manual | correct | keep manual; show missing-data action |
| `corpus_gap_partial_hop` | yes | manual | correct | keep manual; identify missing hop evidence |
| `bad_gold_answer` | yes | manual | correct | keep manual; review/exclude ground truth |

Additional Eval-only outcome:

- `generation_failure`: unconfirmed roll-up used when deeper diagnosis is not
  available. Keep it non-actionable and request/trigger an appropriate deeper
  diagnostic run.

## 7. Implementation sequence

### Phase 0: lock contracts with tests

Add tests before enabling more labels:

1. Rule/parameter contract:
   - every ready prescription uses canonical paths;
   - every path has a `ParameterSpec`;
   - each symbolic value resolves to a concrete valid value;
   - every mapped change round-trips through state reads.
2. Taxonomy closure:
   - emitted labels are ruled, manual, or explicitly allowed roll-ups;
   - ready labels have at least one executable prescription.
3. `applies_when` matching and missing-evidence behavior.
4. Reporter uses the actual selected candidate and actual diff.
5. Missing Eval report never becomes a verified keep.

### Phase 1: finish hybrid runtime optimization

Implement hybrid end to end:

- canonical rules and parameter specs;
- state defaults and mapper;
- optimizer capabilities/constraints;
- truthful dense/hybrid/fallback application metadata;
- Eval retrieval details and dense/hybrid counterfactual tests;
- versioned Serve artifact and reload;
- dense/hybrid integration tests.

Do not call this phase complete if only the same-process Eval changes. A fresh
Serve process must reproduce the chosen config.

### Phase 2: finish diagnosis-driven non-reranker retrieval/chunk labels

- Add under/overchunking signals using gold-span containment, chunk utilization,
  and controlled re-chunk previews.
- Fix chunk strategy names and mapping.
- Define or remove `topic_cluster`; then enable `applies_when`.
- Add per-prescription readiness.
- Rework retrieval-cause precedence and the enumeration confirmation contract.

Reranker-specific labels remain draft/external until the owning team delivers
their end-to-end contract.

### Phase 3: implement advanced retrieval policies

Implement one shared policy at a time:

1. query expansion;
2. bridge decomposition/multi-hop retrieval;
3. MMR;
4. adaptive retrieval;
5. context compression/order/noise filtering.

Each policy must be shared by Eval and Serve. An Eval-only ablation is evidence,
not a production implementation.

### Phase 4: implement generation optimization

- Add `generation_config`.
- Thread it through Eval, RAG, provider calls, serialized serving config, and
  API startup.
- Implement atomic typed policies for grounding, completeness, abstention,
  numerical verification, and hop binding.
- Enable live diagnostics one at a time and promote only the corresponding
  prescriptions.
- Wire generation provenance so extractive fallback is never scored as an LLM
  configuration trial.

### Phase 5: expose optional backends and user choice

- Define how `propose_only` is requested and routed.
- Define a backend-selection policy for `rules`, `internal`, and RAGBuilder.
- Supply RAGBuilder's real input source/eval dataset.
- Keep AutoRAG unavailable until a real adapter and comparison contract exist.
- Make every backend return the same applyable `ConfigPatch` contract.

## 8. Definition of done for one automatic label

A label is complete only when all boxes are true:

- [ ] Eval can produce it from a documented causal signal.
- [ ] `confirmed=True` has a reproducible evidence path.
- [ ] Required metadata keys and aggregation semantics are documented.
- [ ] At least one prescription is individually ready.
- [ ] The prescription uses one canonical config axis.
- [ ] Concrete candidates are grounded or explicitly bounded.
- [ ] Mapper round-trip and capability checks pass.
- [ ] A real Index/RAG/Generator consumer uses the value.
- [ ] Eval measures the changed behavior, including fallback/application state.
- [ ] History can keep or rollback it with valid metrics.
- [ ] Reporter shows the actual selected prescription and actual diff.
- [ ] Serve preserves the chosen value after restart.
- [ ] Unit and `Eval -> Optimize -> Index -> Eval` integration tests pass.

## 9. Verification commands

Use the repository environment on Windows:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest `
  tests\test_planner.py `
  tests\test_optimizer.py `
  tests\test_config_mapper.py `
  tests\test_optimize_agent.py `
  tests\test_internal_adapter.py -q
```

Add targeted suites for retrieval details, serving artifact reload, taxonomy
closure, and every newly promoted in-scope label. Avoid a live model/network
dependency in the default suite; inject deterministic retriever/generator
fakes. Run reranker integration tests only when accepting the owning team's
delivery.
