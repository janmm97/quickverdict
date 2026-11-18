# Security policy

## Supported versions

Only the latest release gets security fixes.

## Reporting a vulnerability

Please report privately via GitHub: Security tab → "Report a vulnerability". Don't open a public issue.
We aim to reply within 7 days and to fix confirmed issues in the next release.

## Deployment notes

quickverdict binds to `127.0.0.1` by default. If you expose it on a network, set `QUICKVERDICT_API_KEY`
and put it behind TLS. Models are loaded with `transformers`; only load model ids you trust.
