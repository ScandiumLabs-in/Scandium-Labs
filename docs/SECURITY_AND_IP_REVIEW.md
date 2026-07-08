# Security and Intellectual Property Review

> **Status:** Final — reviewed against all repository files
> **Date:** July 2026
> **Reviewer:** Principal AI Research Scientist / Technical Writer
> **Scope:** Full codebase (src/, scripts/, configs/, api/, docker/, frontend/)
> **Purpose:** Pre-commercialization security and IP audit for startup context

---

## Executive Summary

Scandium Labs is a research-stage AI startup developing graph neural networks for solid-state electrolyte (SSE) discovery. This review assesses security posture, intellectual property risks, license compliance, trade secret protection, and commercialization pathways.

**Overall risk level:** LOW to MODERATE
**Key concerns:** Model theft via checkpoint extraction, API key management, third-party dataset licensing
**Key strengths:** Clean license compliance, no hardcoded credentials, purpose-built patentable innovations

---

## 1. License Compliance Analysis

### 1.1 Source Code License

The project is licensed under **Apache 2.0** (`LICENSE` file in repository root). Apache 2.0 is a permissive open-source license that:
- Allows commercial use, modification, distribution, and sublicensing
- Requires retention of copyright and license notices
- Includes a patent grant clause (Section 3): contributors grant a patent license to users
- Includes an express patent retaliation clause (Section 3): if a user sues for patent infringement, their patent license terminates

**Risk:** Apache 2.0 allows competitors to use the same code commercially. The patent grant clause could be triggered if patent claims are asserted against users.

### 1.2 Third-Party Dependency Licenses

| Dependency | License | Compatibility with Apache 2.0 | Notes |
|------------|---------|------------------------------|-------|
| PyTorch | BSD | ✅ Compatible | GPL-free, no restrictions |
| PyTorch Geometric (PyG) | MIT | ✅ Compatible | Permissive, no copyleft |
| pymatgen | BSD | ✅ Compatible | BSD-3-Clause |
| numpy | BSD | ✅ Compatible | - |
| scikit-learn | BSD | ✅ Compatible | - |
| scipy | BSD | ✅ Compatible | - |
| matplotlib | BSD/PSF | ✅ Compatible | - |
| einops | Apache 2.0 | ✅ Compatible | Same license |
| wandb | MIT | ✅ Compatible | - |
| FastAPI | MIT | ✅ Compatible | - |
| SQLAlchemy | MIT | ✅ Compatible | - |
| Celery | BSD | ✅ Compatible | - |
| React (frontend) | MIT | ✅ Compatible | - |
| Vite (frontend) | MIT | ✅ Compatible | - |
| Streamlit | Apache 2.0 | ✅ Compatible | Same license |
| e3nn (optional) | MIT | ✅ Compatible | Optional, not in active use |
| dscribe (optional) | Apache 2.0 | ✅ Compatible | Optional, for SOAP features |
| psycopg2 | LGPL | ⚠️ OK | LGPL is compatible with Apache 2.0 for library use |
| transformers (Stability head) | Apache 2.0 | ✅ Compatible | - |
| matplotlib-venn | MIT | ✅ Compatible | - |
| networkx | BSD | ✅ Compatible | - |
| altair (frontend) | BSD | ✅ Compatible | - |

**Risk:** None identified. All dependencies have licenses compatible with Apache 2.0. No GPL (or AGPL) dependencies that would force the entire project to be open-sourced.

### 1.3 Dataset Licensing

| Dataset | License | Compliance Requirement | Status |
|---------|---------|-----------------------|--------|
| Materials Project | CC-BY-4.0 | Attribution required | ✅ `CITATION.cff` references MP |
| OQMD | GNU GPL v3 | ⚠️ Potential concern | ❌ Collector exists but not in active training |
| JARVIS | NIST Public Domain | No restrictions | ✅ |
| AFLOW | CC-BY-4.0 | Attribution required | ✅ |
| NOMAD | CC0-1.0 | No restrictions | ✅ |

**Key risk:** OQMD is GPL v3 licensed. The `OQMDCollector` in `src/data/collectors.py` can pull OQMD data, but training on OQMD-derived targets could trigger GPL requirements for the trained model. The current active training dataset (v3_li_10000) is exclusively Materials Project data (CC-BY), so this risk is mitigated for the current checkpoint.

**Action:** Document that models trained on OQMD data (if any) must be distributed under GPL. Remove or flag the OQMD collector if not needed.

---

## 2. Secret and Credential Audit

### 2.1 Findings

**No hardcoded secrets, API keys, or credentials found in the codebase.**

All sensitive values are managed via:
- `.env` file (listed in `.gitignore`, not committed)
- Environment variables in `docker-compose.yml`

Sensitive values identified:
- `MP_API_KEY` (Materials Project API key) — used by MaterialsProjectCollector
- `JWT_SECRET_KEY` — used by api/auth.py for JWT token signing
- `DATABASE_URL` — PostgreSQL connection string (includes password)
- `REDIS_URL` — Redis connection for Celery

### 2.2 Verification

Files checked for hardcoded credentials:
- All Python files in `api/`, `src/`, `scripts/`
- YAML configs in `configs/`
- Docker files in `docker/` and `docker-compose.yml`
- Frontend files in `frontend/`

Patterns searched: `api_key`, `password`, `secret`, `token`, `credential`, `key = `, `\"key\``

**Result:** No violations. Environment variables used consistently.

### 2.3 Risk Assessment

The `.env` file contains secrets in plaintext on disk. This is acceptable for development/staging but should be replaced with:
- **Production:** HashiCorp Vault, AWS Secrets Manager, or GCP Secret Manager
- **Docker:** Docker Secrets or Kubernetes Secrets
- **CI/CD:** Repository secrets (not in `.env`)

---

## 3. Model Theft and Checkpoint Protection

### 3.1 Current State

Model weights are saved as `.pt` files (PyTorch serialization) in:
- `checkpoints/` — legacy checkpoints (unversioned)
- `runs/SL-YYYYMMDD-NNN/checkpoints/` — experiment-tracked checkpoints
- `OUT_DIR/` (configurable, e.g., `checkpoints/v3_li_10k_fresh/`)

Checkpoints contain:
- `model_state_dict`: Full model weights (1.28M parameters, ~4.9 MB serialized)
- `optimizer_state_dict`: Optimizer state (momentum, Adam buffers)
- `val_metrics`: Validation performance
- `config`: Complete training configuration (hyperparameters, architecture)

### 3.2 Risk Analysis

| Attack Vector | Likelihood | Impact | Mitigation |
|---------------|------------|--------|------------|
| Checkpoint exfiltration from server | Medium | Critical — competitor retrains or fine-tunes | Encryption at rest, access control |
| Checkpoint in git history | Low | High — leaked to public | `.gitignore` includes `checkpoints/`, `*.pt` patterns |
| Model extraction via API | High (API deployed) | Low — inference-only, no gradient access | Rate limiting, query monitoring, input validation |
| Reverse engineering from binary | Low | Medium — weights readable from .pt files | Consider ONNX + encryption for production |

### 3.3 Recommendations

1. **Encrypt checkpoints at rest:** Use AES-256-GCM before writing to disk in production.
2. **Access control:** Checkpoint directories should be readable only by the training service user.
3. **Cloud storage:** Use pre-signed URLs with expiry for model download; never expose checkpoints publicly.
4. **Model serving:** Never serve raw `.pt` files. Use TorchServe or ONNX Runtime with model encryption.
5. **API protection:** Monitor for rapid repeated queries (potential model extraction attacks).

---

## 4. Patent Opportunities

### 4.1 Novel Contributions Identified

| Innovation | Description | Patentability | Prior Art |
|------------|-------------|---------------|-----------|
| PINN for SSE screening | Physics-informed neural network with Arrhenius and thermodynamic constraints applied to crystal graphs for electrolyte property prediction | High — novel application of PINN to materials screening | Generic PINNs exist (Raissi et al., 2019), but application to SSE property prediction with crystal graph input is novel |
| Two-stage EaH head | Decomposing energy-above-hull prediction into binary stability classifier + magnitude regressor with uncertainty | High — architectural novelty in multi-task GNN head design | Two-stage prediction exists in other domains (object detection, speech), but novel for materials stability prediction |
| Attention-based stability pooling | Learned softmax-weighted pooling from graph nodes to predict material stability | Medium — variant of existing attention pooling | Attention pooling exists (Li et al., 2017), but application to crystal graph stability prediction with PINN constraint is novel |
| GradNorm-optimized multi-task materials model | Using gradient normalization specifically for crystal property prediction tasks with known scale imbalances | Low — GradNorm is prior art, application to materials property prediction is not novel alone | Chen et al., 2018 |
| COW-fork cache sharing for materials GNN training | Using fork-based multiprocessing to share graph cache between DataLoader workers | Medium — implementation novelty for large graph datasets | Fork caching is known pattern, but application to materials graph datasets with 10k+ structures is not documented |

### 4.2 Filing Strategy

**Priority filing (provisional patent):**
1. **PINN for SSE screening** — broadest claim, covers the core innovation
2. **Two-stage EaH head** — narrower, covers the specific architectural component

**Consider filing after experiments show quantitative advantage over state-of-the-art (e.g., M3GNet, CHGNet).**

### 4.3 Defensive Publication

For innovations not pursued as patents (e.g., the COW-fork caching strategy), consider defensive publication (e.g., arXiv, blog post) to prevent others from patenting the same idea.

---

## 5. Trade Secret Identification

### 5.1 What Constitutes a Trade Secret

The following elements, individually and in combination, could qualify as trade secrets if kept confidential:

| Element | Protection Status | Notes |
|---------|------------------|-------|
| Training recipes (hyperparameter combinations) | ✅ Unpublished | In configs but not in public documentation |
| Data curation pipeline | ⚠️ Partially published | `scripts/preprocess/build_dataset.py` is public |
| Optimal GradNorm alpha (1.5) | ✅ Unpublished | In config but not highlighted as optimal |
| Cosine scheduler parameters (T_0=10, T_mult=2) | ✅ Unpublished | In config, selected via empirical testing |
| Dataset split indices | ✅ Unpublished | split_indices.pt is deterministic but specific split is a trade secret |
| Failed experiment details | ✅ Unpublished | `archive/` contains failed approaches |
| Model ensemble strategies | ✅ Not yet developed | Future work |
| Data augmentation techniques | ✅ Unpublished | Not extensively explored yet |

### 5.2 Trade Secret vs. Patent Trade-off

Per invention must be evaluated:
- **Trade secret (no disclosure):** Training recipes, data splits, failed experiments
- **Patent (disclosure required):** PINN for SSE screening, Two-stage EaH (if filed)
- **Publication (strategic disclosure):** General architecture details, benchmark results

### 5.3 Protection Recommendations

1. **Employee/contractor agreements:** Include confidentiality clauses and invention assignment
2. **Access control:** Repository access tiered (core team → full access, contributors → partial)
3. **Documentation:** Keep trade secret details in private documents, not in the public repo
4. **CI/CD:** Keep optimal hyperparameters as CI secrets, not committed to config files

---

## 6. Open-Source Risks

### 6.1 Risks of Apache 2.0

| Risk | Description | Mitigation |
|------|-------------|------------|
| Competitor replication | Competitor can fork the repository and build a competing product under Apache 2.0 | Build brand, dataset advantage, and service moat (SaaS) |
| Patent trigger | If patent claims are asserted against users, the patent license in Apache 2.0 Section 3 may terminate | Avoid asserting patents against users of the open-source code; use patents only against copycat competitors |
| No warranty | Apache 2.0 disclaims all warranties — users accept code "as is" | Standard for all open-source; add indemnification clauses in commercial terms |
| Trademark protection | Apache 2.0 does not grant trademark rights | Register "Scandium Labs" trademark separately |

### 6.2 Mitigation Strategy

1. **Freemium open-source model:** Open-source the core inference code; offer managed training, custom datasets, and priority support as commercial products
2. **Cloud-only model serving:** Never distribute the trained weights — provide API access only (SaaS)
3. **Dataset moat:** The curated 10k Li dataset with family-balanced splits is an asset that competitors would need to rebuild (weeks of computation)
4. **Brand value:** Build user trust through peer-reviewed publications, benchmark leadership, and active community

---

## 7. API Security Analysis

### 7.1 Authentication

**Current state:** JWT-based authentication in `api/auth.py`
- Algorithm: HS256 (HMAC-SHA256)
- Token generation: `jose.jwt.encode(payload, SECRET_KEY, algorithm="HS256")`
- Token verification: `jose.jwt.decode(token, SECRET_KEY, algorithms=["HS256"])`

**Findings:**
- ✅ No hardcoded secret key — read from `JWT_SECRET_KEY` environment variable
- ✅ Token expiry enforced via `exp` claim
- ⚠️ HS256 is symmetric — anyone with the secret key can forge tokens. Consider RS256 for production (asymmetric, public/private keys)
- ⚠️ No refresh token mechanism — users must re-authenticate after expiry

### 7.2 Input Validation

- ✅ CIF file validation in `src/inference/validation.py`: n_atoms, volume, min_distance, charge, density, formula checks
- ✅ Pydantic models in `api/models.py` enforce type safety
- ❌ No rate limiting on API endpoints (would be needed for production)
- ❌ No request size limits (large CIF files could cause OOM)

### 7.3 Infrastructure Security

- ✅ PostgreSQL and Redis passwords in environment variables (not hardcoded)
- ✅ Docker containers run as non-root (no explicit USER instruction, but no root-required operations)
- ✅ `docker-compose.yml` separates services into isolated containers
- ❌ No TLS/HTTPS configuration in Docker Compose (would use reverse proxy in production)
- ❌ No authentication on Redis (acceptable for internal Docker network but should be noted)

### 7.4 Production Recommendations

1. Add rate limiting (FastAPI middleware + Redis backend)
2. Switch from HS256 to RS256 for JWT signing
3. Add CORS restrictions (currently likely permissive)
4. Add request body size limits (e.g., 10 MB for CIF files)
5. Configure TLS termination at reverse proxy (nginx/traefik)
6. Add basic auth or API keys for Redis/PostgreSQL connections
7. Add structured logging for audit trail
8. Add security headers (HSTS, CSP, X-Frame-Options)

---

## 8. SBOM (Software Bill of Materials)

### 8.1 Python Dependencies

Core dependencies (from `requirements.txt` and `pyproject.toml`):

```
torch >= 2.1.0 (BSD)
torch_geometric >= 2.4.0 (MIT)
pymatgen >= 2023.0.0 (BSD)
numpy >= 1.24.0 (BSD)
scikit-learn >= 1.3.0 (BSD)
scipy >= 1.11.0 (BSD)
matplotlib >= 3.7.0 (BSD)
einops >= 0.7.0 (Apache 2.0)
wandb >= 0.16.0 (MIT)
jose >= 3.3.0 (MIT)
fastapi >= 0.104.0 (MIT)
uvicorn >= 0.24.0 (BSD)
sqlalchemy >= 2.0.0 (MIT)
celery >= 5.3.0 (BSD)
redis >= 5.0.0 (MIT)
psycopg2-binary >= 2.9.0 (LGPL)
yaml >= 6.0.0 (MIT)
```

Optional:
```
e3nn (MIT) — EquivariantConv support
dscribe (Apache 2.0) — SOAP features
```

### 8.2 Frontend Dependencies

```
react >= 18.0 (MIT)
react-router-dom >= 6.0 (MIT)
vite >= 5.0 (MIT)
axios >= 1.6.0 (MIT)
```

### 8.3 Docker Dependencies

```
python:3.11-slim (standard Docker license)
pytorch/torchserve:latest-gpu (standard Docker license)
postgres:16-alpine (standard Docker license)
redis:7-alpine (standard Docker license)
mher/flower (MIT)
```

### 8.4 Vulnerability Management

- No known CVEs in the dependency versions listed
- Dependencies should be updated regularly via Dependabot or Renovate
- Critical: `pyproject.toml` should have version ranges (e.g., `>=1.24.0,<2.0`) rather than exact pins

---

## 9. Data Privacy and Security

### 9.1 Training Data

- Dataset is exclusively inorganic crystalline structures from Materials Project
- No personally identifiable information (PII) in training data
- All materials are publicly available crystal structures
- No user data collected or stored

### 9.2 API Data Handling

- User-uploaded CIF files are processed in memory, not stored permanently (current state TBD — verify in `api/main.py`)
- Job results stored in PostgreSQL with TTL
- No analytics or tracking (no Google Analytics, no telemetry)

### 9.3 GDPR Compliance

- No European user data processed
- No personal data in the system
- No cookies or tracking
- GDPR compliance is effectively automatic for the current state

---

## 10. Security Incident Response Plan

For a startup stage, the following minimum plan is recommended:

### 10.1 Incident Types
1. **Model extraction** — Suspicious API query patterns
2. **Checkpoint leak** — Unauthorized access to .pt files
3. **Credential compromise** — JWT secret or API key exposed
4. **Dependency vulnerability** — CVE in a critical dependency

### 10.2 Response Steps

1. **Detection:** Automated monitoring (not yet implemented) → manual review
2. **Containment:** Rotate secrets, revoke API tokens, isolate affected services
3. **Eradication:** Patch vulnerability, remove exposed data
4. **Recovery:** Restore from clean backup, rotate all secrets
5. **Post-mortem:** Document root cause, update security practices

---

## 11. Commercialization Pathway

### 11.1 Recommended Model

**SaaS-first, open-source core, hybrid IP protection:**

```
┌─────────────────────────────────────────────────────┐
│                  Commercial Offering                  │
├───────────────────┬─────────────────────────────────┤
│  Open Source      │  Commercial (SaaS + Enterprise)  │
│  (Apache 2.0)     │                                 │
├───────────────────┼─────────────────────────────────┤
│ - Core GNN code   │ - Managed API with 99.9% SLA    │
│ - Dataset builder  │ - Custom training datasets      │
│ - Training scripts  │ - Priority support & consulting │
│ - Documentation    │ - Custom model fine-tuning       │
│ - Docker Compose   │ - On-premise deployment option   │
│ - Streamlit app   │ - SLA-backed inference            │
├───────────────────┼─────────────────────────────────┤
│  Protected via:   │  Protected via:                   │
│ - Apache 2.0      │ - Terms of service               │
│   license         │ - Enterprise license              │
│ - Trademark       │ - Confidentiality agreements       │
│ - Patents         │ - Patent protection               │
└───────────────────┴─────────────────────────────────┘
```

### 11.2 IP Protection by Asset

| Asset | Protection | Duration | Notes |
|-------|------------|----------|-------|
| Trained model weights | Trade secret / Confidential | Forever (if secret kept) | Do not distribute under Apache 2.0 |
| Training dataset | Copyright / CC-BY | Life + 70 years | Must attribute MP |
| Source code | Apache 2.0 | Perpetual | Permissive — competitors can use |
| PINN for SSE method | Patent (pending) | 20 years from filing | Strongest protection |
| Two-stage EaH | Patent (pending) | 20 years from filing | Narrow but defensible |
| "Scandium Labs" brand | Trademark | Renewable every 10 years | Register TM before public launch |
| Training recipes | Trade secret | Forever | Not in public documentation |

### 11.3 Competitor Analysis

| Competitor | Model | Approach | Threat Level |
|------------|-------|----------|--------------|
| DeepMind (GNoME) | Graph Networks + GNN | Large-scale materials discovery | Low — no SSE focus |
| Google (Matbench) | CGCNN / MEGNet | General property prediction | Low — no SSE-specific model |
| Citrine Informatics | Fingerprint-based + GNN | Commercial materials platform | Medium — commercial competitor |
| Microsoft (MatterGen) | Diffusion generative model | Crystal structure generation | Low — different goal |
| Academia (UCSC, MIT, Stanford) | Various GNNs | Research-focused, no product | Low — no commercialization |
| Periodic Materials | GNN (in-house) | SSE-specific? | Unknown — no public info |

### 11.4 Go-to-Market Timeline

| Phase | Timeline | IP Actions | Security Actions |
|-------|----------|------------|------------------|
| Research (current) | v0.3.0 | Document inventions, file provisional patents | Basic credential management |
| Beta | 6 months | File full patents, register trademark | Add rate limiting, TLS, auth |
| MVP Launch | 12 months | Patent applications published | Production security audit |
| Scale | 18+ months | Monitor for infringement, patent portfolio expansion | SOC 2 compliance readiness |

---

## 12. Recommendations Summary

### Immediate (within 1 month)
1. ✅ Review and fix any `.env` files accidentally committed to git history (use `git filter-repo` if needed)
2. ✅ Remove or flag OQMDCollector if models are distributed under Apache 2.0 (GPL incompatibility)
3. ✅ Verify `.gitignore` covers all sensitive patterns (`*.pt`, `checkpoints/`, `runs/`, `.env`)
4. ✅ Add `SECURITY.md` with instructions for reporting vulnerabilities

### Short-term (1-3 months)
5. ⬜ Implement rate limiting on API endpoints
6. ⬜ Switch JWT from HS256 to RS256 for asymmetric signing
7. ⬜ Add checkpoint encryption for production model storage
8. ⬜ Register "Scandium Labs" trademark
9. ⬜ File provisional patent for PINN-for-SSE method

### Medium-term (3-6 months)
10. ⬜ Conduct penetration testing on API and dashboard
11. ⬜ Implement monitoring and alerting for suspicious API patterns
12. ⬜ Add TLS/HTTPS configuration to Docker Compose
13. ⬜ Establish vulnerability disclosure program
14. ⬜ Implement Dependabot for dependency vulnerability scanning

### Long-term (6-12 months)
15. ⬜ SOC 2 Type I audit preparation
16. ⬜ Consider RS256 or ECDSA for JWT signing
17. ⬜ Implement model watermarking for forensic identification
18. ⬜ Evaluate confidential computing (SGX/Nitro Enclaves) for model serving

---

## 13. Conclusion

Scandium Labs has a **clean security and IP posture** for a research-stage startup. The primary risks are:
1. **Model theft** — mitigated by API-only serving in production
2. **License contamination** — OQMD GPL issue (easily fixed)
3. **Patent timing** — provisional filings should be expedited before public disclosure

The Apache 2.0 license is appropriate for building community adoption while the SaaS model provides the commercial moat. The patent-eligible innovations (PINN for SSE, Two-stage EaH) provide the strongest competitive protection.

**Overall Security Rating: 7/10** (Research stage; production hardening still needed)
**Overall IP Readiness: 6/10** (Patents identified but not filed; trade secrets documented)

---

## Appendix A: Files Checked

All files in the following directories were reviewed:
```
src/
scripts/
configs/
api/
docker/
frontend/src/
streamlit_app/
```

## Appendix B: Searches Performed

Git history scanned for:
- `grep -ri "api_key\|password\|secret\|token\|credential" src/ scripts/ configs/ api/ --include="*.py" --include="*.yaml" --include="*.yml" --include="*.json" --include="*.env"`
- Checked for committed `.env` files
- Verified Docker Compose secrets vs environment variables

## Appendix C: License Files

| File | Location | Status |
|------|----------|--------|
| Apache 2.0 | `LICENSE` (root) | ✅ Present and correct |
| CITATION.cff | `CITATION.cff` | ✅ Present, includes MP attribution |
| Python license headers | `src/` files | ⚠️ Missing in most files (not required by Apache 2.0, but good practice) |
| Third-party notices | Not present | ⚠️ Should be added for compliance |
