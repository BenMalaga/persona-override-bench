# Data: resources, not datasets

This project generates its own benchmark content (synthetic personas + scripts, all
self-authored, nothing scraped, no human-subjects data). External resources used:

- **VERA-MH rubric** (open, clinician-validated; clinician inter-rater reliability
  IRR 0.77 and LLM-judge-vs-clinician-consensus IRR 0.81, per arXiv:2602.05088,
  Bentley et al. 2026): https://github.com/SpringCare/VERA-MH . Fetch read-only;
  safe-messaging checklist items are adapted with attribution. Repo verified live +
  active 2026-06-14 (v1.1.0, MIT-modified license).
- **Statute texts** (rubric anchoring): California SB 243; NY GBL Art. 47. Keep local
  copies of the enacted text in `data/statutes/` (public documents, not committed).
- **Crisis-resource canon** for the deterministic regex scorer: 988 Suicide & Crisis
  Lifeline, Crisis Text Line (741741), and international equivalents. The pre-registered
  pattern list lives in `src/`, and live numbers are re-verified at run time.

Model outputs land in `results/raw/` (gitignored until release).
