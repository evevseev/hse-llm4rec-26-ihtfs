# IHTFS - FAR-Trans Recommendation Project

HSE coursework (2026), LLM4Rec: AI-Assisted Recommender Systems.

## Team

- Roman Dmitrievich Zenin
- Evgeny Vasilyevich Evseev

## Research question

Can a frozen open-weight LLM improve nDCG@10 over reciprocal rank fusion (RRF)
when both rank the same candidate pool built from past purchases, content
similarity, and popularity on FAR-Trans?

The first planned comparison uses the starter's 100-user teaching cohort.
The final shared benchmark protocol is pending. The output is a top-10 asset
list; accuracy against future purchases does not measure investment suitability
or profit.

## Current status

Session 1: research question and study plan. A real released record and metric
arithmetic have been checked. The recommendation model has not been implemented
or evaluated. No ranking improvement is claimed.

The plan uses one Qwen2.5-1.5B-Instruct checkpoint, an earlier development window,
and a locked main comparison with two candidate orders. See the study plan and
its review clarifications before implementation.

## Materials

- [Study plan and related work](session1/session1_plan_and_literature.md)
- [Candidate-pool and RRF protocol](session1/session1_candidate_protocol.md)
- [Review clarifications and input contract](session1/audit_addendum.md)
- [Proposed English reranking prompt](session1/prompts/rerank.txt)
- [Actual released record and provenance](session1/session1_data_example.md)
- [Independent source check](session1/eval/verify_evidence.py)
- [Saved verification output](session1/results/independent_verification.json)
- [Reproducibility check](session1/results/reproducibility_check.json)
- [Analytical design checks](session1/results/design_checks.json)

To repeat the source check, download the official data separately and run:

```bash
python3 session1/eval/verify_evidence.py \
  --data-dir /path/to/FAR-Trans \
  --output /tmp/far-trans-verification.json
```

The script uses only the Python standard library. Raw data, private reports,
AI session exports, and course materials are not stored in this repository.
AI assistance and the boundaries of verification are described in the individual
submission.

## Sources

- [FAR-Trans dataset](https://researchdata.gla.ac.uk/1658/), University of Glasgow,
  CC BY 4.0.
- J. Sanz-Cruzado, N. Droukas, and R. McCreadie,
  [FAR-Trans: An Investment Dataset for Financial Asset Recommendation](https://arxiv.org/abs/2407.08692), 2024.
- [Course starter](https://github.com/dryjins/RecSys-LLMs/tree/51337032503c56e3a28ef646a24aa016782e1bd3/project/far-trans).

Initial Session 1 artifacts are also preserved in
[Evgeny's individual coursework repository](https://github.com/evevseev/hse-llm4rec-26/tree/main/week7).
