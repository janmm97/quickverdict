# Contributing to quickverdict

Thanks for helping! Small, focused pull requests are easiest to review.

## Set up

```bash
git clone https://github.com/janmm97/quickverdict
cd quickverdict
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest            # runs on CPU with a tiny random model, no GPU needed
ruff check .
```

## How we work

- **Open an issue first** for anything bigger than a bug fix, so we can agree on the approach.
- Branch from `main`, one topic per branch.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:` …).
- Keep the code small: the smallest change that fully does the job.
- Any accuracy or speed claim must come with the benchmark command and the hardware it ran on.

## Pull request checklist

- [ ] `pytest` and `ruff check .` pass
- [ ] Tests cover the change (they fail if the change is reverted)
- [ ] README / docs updated if behaviour changed
- [ ] `CHANGELOG.md` entry under `[Unreleased]`

## Reviews

A maintainer reviews every PR. CI must be green before merge. We may ask for changes; that's normal.

## Questions

Open a GitHub issue. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
