# Security

- API keys live in `.env`, which is git-ignored. Never commit real keys.
- Client documents (case files, petitions, CSV exports) are never committed: `*.csv` and `data/private/` are git-ignored. The repository only ships a sample corpus of short summaries of public legislation.
- The app binds to `127.0.0.1` by default. `--share` opens a public ngrok tunnel; use it only for short demos.

To report a vulnerability, open a private security advisory on this repository or contact the maintainer through Upwork.
