# data/

Session recordings, labelled datasets, and model weights go **outside git**
— they're too large and change too often to version usefully in a repo.
This folder is gitignored on purpose (see root `.gitignore`).

Suggested approach once the team is recording real sessions:

- Store raw footage and datasets in cloud storage (e.g. an S3/GCS bucket
  or a shared drive), one folder per capture session, dated.
- Keep a lightweight index here (or in a shared spreadsheet) of what each
  session contains and where it lives, so it's discoverable without being
  committed.
- If small labelled samples are useful to keep in-repo for tests, put them
  under `cv-pipeline/tests/fixtures/` instead — anything under a few MB is
  fine there.
