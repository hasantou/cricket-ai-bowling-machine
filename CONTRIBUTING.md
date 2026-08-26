# Contributing

This is a small, early-stage team project. Lightweight process on purpose —
add more structure only once it's actually needed.

## Workflow

1. Create a branch off `main`: `git checkout -b <name>/<short-description>`
   (e.g. `alex/pose-estimation-baseline`).
2. Commit early and often with clear messages (what changed and why, not
   just what).
3. Open a pull request into `main` for anything beyond a trivial fix, even
   solo — it keeps a reviewable history as the team grows.
4. Keep large files (video, datasets, model weights) out of git entirely —
   see `data/README.md` for where those belong instead.

## Where things live

- **Planning and business docs** → `docs/`
- **Computer vision / ML code** → `cv-pipeline/`
- **Mastery scoring & style-recommendation logic** → `adaptation-engine/`
- **Mechanical / embedded work** (Phase 2+) → `hardware/`

## Issues

Use the templates under `.github/ISSUE_TEMPLATE/` — one for bugs, one for
research/experiment tracking (useful for logging MVP validation sessions:
what was tried, what the coach said, what the AI called).

## Code review basics

- Someone other than the author reviews before merge, once the team is
  more than one person.
- Note any assumption baked into a mastery threshold or scoring rule in the
  PR description — these are exactly the design decisions Phase 1's
  validation is testing.
