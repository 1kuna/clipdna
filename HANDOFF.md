# ClipDNA Handoff

Date: 2026-06-22
Continuity: repo-state handoff. No owner thread was recovered for this pass.

## Current State

- Branch: `main`
- Remote/upstream: `origin/main` at `https://github.com/1kuna/clipdna.git`
- Starting state for this handoff pass: clean and aligned with upstream.
- Latest code commit before this handoff: `77689ba Remove cached bytecode and ignore artifacts`.
- Product shape from README: video fingerprinting system using ViSiL embeddings, FAISS, and Chromaprint.

## Last Meaningful Work

Recent history shows a desktop/NAS/GPU stack was added, followed by artifact cleanup:

- `aec73e3 Implement ClipDNA desktop stack and setup wizard`
- `77689ba Remove cached bytecode and ignore artifacts`

The repo is split into `nas/`, `gpu-node/`, `shared/`, `scripts/`, and `tests/`.

## What Is Not Verified In This Pass

No Docker services, GPU worker, API health check, indexing, querying, or tests were run while writing this handoff.

## Resume Steps

1. Read `PROJECT_OUTLINE.md`, `CLIPDNA_IOS_SPEC.md`, and `docs/`.
2. Bring up NAS services from `nas/` with a local `.env`.
3. Smoke `curl http://localhost:${API_PORT}/health`.
4. Only then run indexing/query scripts on a tiny controlled sample.

## Cautions

- Do not assume NAS/GPU storage paths exist on another machine.
- Keep source videos, clips, FAISS indexes, embeddings, and database volumes out of git.
- This handoff is repo-state-only and should be superseded if historical project context is later recovered.
