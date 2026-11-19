# Stage 1 ledger — plan: local Conda environment, DeepSeek adapter and offline validation

- Baseline: AgentD 2ecae868ba26d0ed9ffda51ecf719e5aa7d7d31b; isolated worktree branch codex/deepseek-local.
- Environment: D:/anaconda/envs/agentd-deepseek, Python 3.11.18; CUDA torch 2.6.0 requested. All task caches on D.
- REINVENT source: 3266e54bbe148831796477577d977de8cfbdeb05, verified official archive from preceding stage.
- Pre-flight: shared runtime configuration must propagate to MCP children, generation and retrieval. Local Windows and DSW paths differ; resolve from runtime settings/current interpreter.
- Ruling: use existing local worktree path specified by user; do not change reference clone, DSW, or upload anything.
- Ruling: stage-1 tests must never send authenticated real API calls; use HTTP transport mocks and fail-closed network guards.
- Ruling: original tests construct an agent that demands secrets/local model; update tests to supply simulated dependencies, preserving production behavior.
- Task environment: Conda created; dependency/model downloads in progress.
- Ruling: Conda Python initially exposed the pre-existing Windows user site-packages. Set PYTHONNOUSERSITE=1 in this environment and all verification commands, so dependencies/tests cannot silently use unrelated user packages.
- Model: multilingual-e5-small revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 downloaded, local manifest records SHA-256.
- First full test attempt: native process aborted in FAISS search due to pip FAISS LLVM OpenMP vs torch Intel OpenMP conflict. GPU preflight reproduced it. Do not set KMP_DUPLICATE_LIB_OK; replacing Windows FAISS with conda-forge 1.10.0 OpenBLAS build and preserving numpy 1.26.4.
- First baseline import tests: 3 passed, 1 failed because ChEMBL performs HTTP at module import. Move that import into get_drug_smiles so offline startup is genuinely offline; real ChEMBL calls remain outside stage one.
- FAISS fix verified: conda-forge faiss 1.10.0 cpu_openblas_py311_h44cb508_0 with NumPy 1.26.4 runs a real FAISS search in the same process as torch 2.6.0+cu124 without duplicate-runtime bypasses. Windows distribution metadata is named faiss; setup.py uses platform markers, Linux remains faiss-cpu.
- Config/search/doc tests: 12 passed plus 2 subtests using blocked real transports.
- Regression: invalid generated SMILES were accepted by the upstream pool combiner; observed failing assertion, then added RDKit validation and explicit no-valid-SMILES failure.
- Final review: two P2 gaps (effective model inheritance and PDF source/page context) reproduced by 3 failing regression tests, fixed, then complete suite rerun.
- Final full suite: 31 passed, 2 subtests passed, 7 dependency deprecation warnings, 0 failures (19.66s). Tests block real requests/httpx transports. pip check passed.
- Final real local verification: PowerShell launcher exit 0; RTX3060 CUDA computation, multilingual 384D E5/FAISS retrieval and both generators passed. Reinvent=2 valid SMILES; Mol2Mol=2 valid SMILES. Final artifact reports/offline-20261105-121750/report.json.
- Delivery: LOCAL_SETUP.md, TEST_REPORT.md, environment locks, reports and review-only sanitized diff. No real API requests, no OSS upload, no DSW mutation.
- Ruling: keep the user-requested worktree/branch in place because the installed editable AgentD environment runs from it. Do not merge, publish, delete the worktree, or modify the reference clone.

## Stage 2 — bounded live API preflight

- User requested dedicated configuration/live check, at most one chat, one search and one fetch, no retries, output <=64 tokens. A real key and explicit fee confirmation remain required before any paid request.
- Added 7 regression tests, observed all fail before implementation. Found pinned LangChain rewrites max_tokens to max_completion_tokens; DeepSeek documents max_tokens, so adapter remaps it back. Added a transport-level 429 test proving zero chat retries.
- Focused tests: 8 passed plus 2 subtests, using mock HTTP only. No live API call made.
- Full regression suite after stage-2 changes: 39 passed, 4 subtests passed, 7 dependency warnings; configuration-only CLI produced needs_key with all attempt counts 0. Prepared ignored empty credential file with current-user/SYSTEM ACL; no real secret was written or printed.
- Status: code preparation complete; live validation awaiting private key configuration and explicit fee confirmation. Do not repeat or expand paid requests automatically.
- User then explicitly confirmed private key configuration and this paid test batch. Executed check_deepseek.py --live exactly once, report reports/deepseek-live-approved-20261105.json, exit 0, status passed.
- Actual usage: chat 12 input + 1 output = 13 tokens; Search returned 3 PubMed sources; Fetch retrieved 5964 characters. Attempts chat=1, search=1, fetch=1; no retries. Updated API_CHECK_REPORT.md and stage-one report's subsequent-stage status.

## Stage 3 — local affinity models and DeepPK

- User-approved plan implemented in the existing isolated codex/deepseek-local worktree. Downloads and model snapshots use D Downloads, all installation/download caches use D Temp. Existing uncommitted work preserved.
- Interface preflight: affinity runtime settings must propagate into worker processes; both prediction tools must retain original input row identity; worker exit codes must reflect tool failure. Original model mean pooling and rescaling remain the binding compatibility contract.
- Downloaded fixed ProtT5 and MoLFormer compat-v4 snapshots. Independently rehashed every file and verified official LFS digests. Total snapshot bytes 2,604,035,998. Model cards and custom code retained. Added environment/affinity-models.lock.json.
- Installed sentencepiece 0.2.0, accelerate 0.31.0, psutil 7.2.2. Existing Transformers 4.41.0 and torch 2.6.0+cu124 retained. Updated Windows pip lock and freeze audit. pip check passed.
- Added sequential local-only encoder loading, GPU FP16 ProtT5, CPU FP32 fallback, persistent protein feature cache, fixed per-encoder seeds, CPU FP32 strict BAPULM head, resource measurement and native Windows output names. Input columns and row order are preserved.
- New tests observed failing before implementation. Actual CUDA measurement exposed an uninitialized allocator, reproduced with a regression test and fixed by explicit CUDA initialization.
- Real offline acceptance blocked socket connections and HTTP requests. Official short sequence and UniProt human BCL-2 P10415 (239 residues, source/hash saved) passed. BCL-2 aspirin/caffeine outputs 4.8933992386 and 3.9568099976 pKd. Independent cold repeat difference 0; normal AgentD tool cache hit matched exactly.
- Actual CPU FP32 short-sequence inference passed; simulated CUDA OOM test proved one CPU retry. GPU BCL-2 peak allocated 3,261,243,904 bytes, reserved 3,487,563,776 bytes. Resource reports retain elapsed times and sampled peak RSS.
- DeepPK submitted exactly one two-molecule job admet_1793866815.7091815. First parsing failed because current server field labels differ from the official documentation sample. Ruling: accept both documented and current field labels while preserving original names and confidence fields. Retrieved same job, no resubmission. Real output passed, 2 rows, 197 columns, 73 predictions and 49 probability columns.
- Independent read-only final reviewer found three P2 issues: crash-stale PID lock, misleading DeepPK top-level report state, and reused cold-repeat cache. Added regression coverage for lock recovery and status and reran actual cold acceptance with a unique cache and explicit cache-miss assertion. Used OS-backed filelock for crash release.
- Final full suite: 58 passed, 4 subtests passed, 7 dependency deprecation warnings, 0 failures (19.28s). pip check and git diff --check passed. Real generator regression also passed Reinvent=2 and Mol2Mol=2 valid SMILES, E5/FAISS and CUDA checks passed.
- Delivery: AFFINITY_SETUP.md, model manifests, current dependency locks, prediction CSVs, source provenance, resource reports and test report. Existing worktree remains the runtime source.
- Scope ruling: plan section 5 is a follow-on stage after local acceptance. Prediction-table integration into LLM optimization, Linux packaging, fee-confirmed OSS upload and A10 validation remain next-stage work. Boltz remains a separate future stage. No further paid API request was needed for this stage.

## Stage 4 — bounded real generation and optimization loop

- User requested full generation/optimization and explicitly approved one local batch capped at RMB 1, 20 chat requests, 6 Search and 3 Fetch requests, with a 30-minute batch deadline and no SDK automatic retries.
- Implemented stable candidate IDs, canonical-molecule deduplication, strict affinity/ADMET joins, confidence-preserving optimization input, per-candidate parsing/validity states, unchanged-result reuse, modified-only re-prediction and measured comparison CSVs. Worker exit codes now determine failure. Baseline failures and interrupted runs also retain candidate states.
- Added a process-shared FileLock budget with pre-request conservative reservations, actual token-usage settlement at peak uncached rates and persisted request counts. Bounded chat is non-streaming with output <=2048 tokens and serialized input <=32 KiB. Search result excerpts and count are limited in bounded mode. Configuration-only startup is the default.
- Independent final review found baseline worker/merge failures lacking per-candidate records. Reproduced missing-row failure, fixed it and reran the full suite. Added inter-process budget contention coverage.
- First zero-request startup BCL2_loop_20261105_165808_a1d73b exposed MCP child user-site dependency leakage. Inspected MCP's inherited environment whitelist and actual loaded binary paths, reproduced missing PYTHONNOUSERSITE propagation, fixed both runner transports and verified tests.
- Real paid batch BCL2_loop_20261105_170300_a4fe00 exposed inherited protocol stdin in local CLIs/workers and unsupported sampling parameter tb_logdir. Reproduced both, isolated stdin/stdout, added pinned REINVENT parameter-schema validation and recovered saved checkpoints. Added regression tests for standard streams and config normalization.
- Ruling: recover the same paid batch from persisted extraction/configs/generated molecules rather than reissue successful paid calls. Preserve its original usage ledger and original deadline. Startup attempts and interrupted worker metadata remain inspectable. Recovery never reissued the original 8 successful chat calls.
- Final real run exit 0, summary ok=true. Human BCL-2 P10415, 239 residues; Venetoclax seed verified by formula C45H50ClN7O7S. Both generators produced 2 valid SMILES, yielding 4 distinct candidates. All 4 were structurally modified and re-predicted successfully. Comparison has 4 rows/337 columns, exact candidate alignment and finite affinity/RDKit values.
- Actual calls chat=16, search=1, fetch=0. Prompt tokens=45,291, completion tokens=4,892. Conservative charge upper bound RMB 0.134718, remaining reservation 0. Original paid batch elapsed 1371.621 seconds, below 1800. DeepPK before/after each submitted once, jobs admet_1793870341.199011 and admet_1793870566.3376489 completed.
- Measured affinity increased for 1 candidate and decreased for 3. QED increased for 2 and decreased for 2. Technical closed-loop acceptance passed, with individual outcomes retained in the report.
- Final suite 76 passed, 4 subtests passed, 7 dependency deprecation warnings, 0 failures (30.96s). pip check and git diff --check passed. Exact-secret scan passed across 64 actual-run text artifacts.
- Delivered LOOP_SETUP.md, LOOP_TEST_REPORT.md, current source manifest, existing dependency locks and complete run artifacts. Linux packaging and actual-size/fee-confirmed OSS upload steps are specified in the report as the next stage.
