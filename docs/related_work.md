# Related work

How PersonaOverride-Bench sits relative to the closest existing benchmarks and
papers, and what it adds. Citations reflect what each source actually claims;
quotes paraphrase the abstract/README unless marked.

Prior-art picture compiled from a web survey on 2026-06-14, with primary sources
checked where a figure is load-bearing.

---

## 1. The regulatory anchor (the niche)

Two statutes give this benchmark its distinct framing; both were verified against
legal-analysis primary sources (Manatt, Morrison Foerster, Perkins Coie, Skadden,
CA Legislative Information, NY Governor's office).

- **California SB 243** (Padilla), companion-chatbot law. **Effective 2026-01-01.**
  Operators must maintain a protocol to prevent self-harm content and, at minimum,
  refer a user who expresses suicidal ideation / self-harm to crisis providers
  (suicide hotline / crisis text line). Annual operator reporting to the CA Office
  of Suicide Prevention begins **2027-07-01.** Private right of action / statutory
  damages. (Matches `docs/rubric_statute_map.md` and PRE_REGISTRATION §9.)
- **New York General Business Law Article 47**, "AI Companion Models." **Effective
  2025-11-05.** Requires (a) a crisis-referral protocol on a user's expression of
  suicidal ideation / self-harm, and (b) a clear "you are not talking to a human"
  disclosure **at the start of each interaction and at least every three hours** of
  a continuing session. The start + every-3-hours disclosure applies to **all users**,
  not only minors (Manatt, 2025-11). Enforced by the NY AG, civil penalties up to
  **$15,000/day**, fines fund NY suicide-prevention programs.

To our knowledge, no existing benchmark ties its scored items to these specific
statutory clauses. That regulation-anchored compliance framing is what distinguishes
the project from neighboring work.

## 2. Closest neighbors (persona / companion safety evaluation)

| Work | What it does | Distinction from this work |
|---|---|---|
| **Persona-Grounded Safety Evaluation of AI Companions in Multi-Turn Conversations** (Juneja & Lomidze, arXiv:2605.00227, 2026) | "First end-to-end scalable framework" for controlled multi-turn simulation + safety evaluation of companion apps; applied to Replika across 9 clinical personas (depression, anxiety, PTSD, eating disorders, incel identity), 25 high-risk scenarios, ~1,674 dialogue pairs. Detects unsafe content via emotion modeling + LLM classification. | No explicit **"never break character" instruction-robustness** test; **no pressure ladder** (graded persona lock-in L0-L5); **no regulation-anchored** thresholds (no SB 243 / NY GBL); no judge-independent deterministic resource scorer; no two-sided false-alarm (over-trigger) cost. This is the nearest paper and serves as the direct comparator on these four deltas. |
| **Persona Non Grata: Single-Method Safety Evaluation Is Incomplete for Persona-Imbued LLMs** (arXiv:2604.11120, 2026) | Argues a single evaluation method understates persona-conditioned safety risk; persona conditioning degrades safety via toxicity amplification + role-play exploitation. | Method-completeness argument, not a crisis-referral compliance instrument; no statute anchor; no crisis-trajectory factor. Supports the decomposed, multi-signal design rationale used here. |
| **Persona Jailbreaking in Large Language Models** (arXiv:2601.16466, 2026) | Persona prompts as a jailbreak vector. | Jailbreak (elicit-harm) framing; this benchmark studies the inverse defensive question (does persona **suppress a legally required helpful behavior**). Cited to draw the contrast: pressure -> less-safety is the *manipulation check* (H1), not the headline result. |
| **Enhancing Jailbreak Attacks on LLMs via Persona Prompts** (arXiv:2507.22171) | GA-evolved persona prompts cut refusal rates 50-70%. | Same offensive direction; relevant as evidence that L5 lock-in is a realistic threat model rather than a strawman. |
| **MultiBreak: Scalable and Diverse Multi-turn Jailbreak Benchmark** (arXiv:2605.01687, 2026) | Multi-turn jailbreak benchmark pooling 5 datasets; gradual-escalation + concealment strategies. | General jailbreak coverage; the multi-turn factor here is a **crisis-disclosure trajectory** (monotonic / deep-entrenchment / zigzag) with per-turn severity tiers, tied to a referral duty rather than harm elicitation. Cited as a methodological neighbor for multi-turn escalation. |
| **Memory-Driven Role-Playing** (arXiv:2603.19313, 2026) | Evaluates + enhances persona-knowledge utilization / consistency in role-play. | Capability-of-role-play, not safety-under-role-play; informs the persona-consistency / backstory-confabulation secondary metric used here. |

## 3. Crisis-response / mental-health safety benchmarks

| Work | Role in this benchmark |
|---|---|
| **VERA-MH** (SpringCare; open, MIT-modified). Reliability/validity paper: **arXiv:2602.05088** (Bentley et al., 2026); concept paper arXiv:2510.15297. Repo `github.com/SpringCare/VERA-MH` is **live + active** (v1.1.0, 2026-04-29). | This benchmark **reuses the VERA-MH rubric** for safe-messaging items, with attribution. **Reliability figures: clinician inter-rater reliability IRR = 0.77; LLM-judge-vs-clinician-consensus IRR = 0.81.** These two distinct numbers are used precisely; the clinician-only figure (0.77) is not conflated with the judge-alignment figure (0.81). VERA-MH's finding that an LLM judge can reach 0.81 alignment with clinician consensus is supporting evidence for the judge stage and the reason it still requires a local two-human reliability gate before any judge-dependent metric is trusted. |
| **AI Safety Training Can be Clinically Harmful** (Suhas BN, Sherrill, Arriaga, Wiese, Abdullah; arXiv:2604.23445, 2026) | Directly relevant to the Track C / over-trigger side. Finds RLHF safety alignment disrupts therapy by "inserting crisis resources into controlled exercises," offering false reassurance, and refusing to challenge distorted cognitions; therapeutic appropriateness collapsed to 0.22-0.33 at the highest severity. This is independent evidence that **over-triggering is a real, measurable cost**, which is exactly what Track C (figurative-language distractors) quantifies. Cited as external justification for the two-sided trade-off curve. |
| **STELLA: Safety Testing Engine for Large Language Assistants** (medRxiv 2025.12.11.25342078) | Adjacent safety-testing engine; cited as a contemporaneous testing framework, contrasted on the regulation anchor + deterministic resource scorer. |
| **JMedEthicBench** (Liu et al., 2026) | First multi-turn safety benchmark for medical LLMs; safety scores decline across ~50k adversarial conversations, 27 models. Cited for the multi-turn-degradation finding (parallels H1 dose-response) in the medical domain. |
| **Character.ai harmful-response measurement** (reported alongside the persona-safety literature, 2026): ~35.7% overall harmful-response rate; worst under "Risk Intent" (60.3%) and "Harmful Belief" (34.6%) persona states. | Real-world severity context for why the L5 lock-in threat model matters; cited as motivation, not as a method reused here. |

## 4. Additional prior work engaged

The papers below are discussed alongside those above:

- **arXiv:2605.00227** (Persona-Grounded Safety Eval): primary comparator, summarized in §2.
- **arXiv:2601.16466** (Persona Jailbreaking): summarized in §2.
- Further related preprints and reports surveyed: 2512.12775, 2512.01247 (NDSS'26),
  2502.20757, 2602.13234, 2601.17003, 2511.08880, Persistent Personas v2, the
  Anthropic persona-vectors / Assistant Axis work, McBain/RAND, De Freitas, and Moore.
  Each is read against its primary text and quoted only for what it states.

## 5. Positioning summary

The field is active: several directly adjacent 2026 preprints appeared during this
survey, placing the work in a crowded but viable corner of the literature. The
specific combination this benchmark contributes is: (1) a graded persona-pressure
ladder (L0-L5 lock-in) crossed with (2) crisis-disclosure **trajectory shape**
(monotonic / entrenchment / zigzag) and severity tier, (3) scored by a
**judge-independent deterministic resource regex** for the headline H2 claim, (4)
**anchored to named SB 243 / NY GBL Art. 47 clauses**, and (5) reporting the
**two-sided over-trigger cost** (externally corroborated by arXiv:2604.23445). To
our knowledge, no single neighbor combines more than two of these five.
