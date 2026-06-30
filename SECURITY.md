# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability in Scandium Labs, **do not** open a public issue. Instead, report it privately by emailing **security@scandiumlabs.com** or using GitHub's private vulnerability reporting feature.

Please include:
- A description of the vulnerability
- Steps to reproduce it
- The affected version(s)
- Any potential impact or exploit scenario

You should receive a response within **48 hours**. We will keep you informed as the issue is investigated and resolved.

## Security Practices

- All code is reviewed for security before merging
- Dependencies are scanned for known vulnerabilities (via `pip-audit` or equivalent)
- Secrets and tokens are **never** committed to the repository
- Least-privilege principle is applied to all service accounts and API keys
- HTTPS/TLS is enforced for all network communication in production

## API Key Management

- API keys are loaded exclusively from environment variables or a `.env` file (see `.env.example`)
- `.env` files are listed in `.gitignore` and **must never** be committed
- Keys are rotated periodically and immediately revoked if a leak is suspected
- Use read-only or scoped keys where possible

## Environment Variable Usage

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API authentication |
| `HF_TOKEN` | Hugging Face authentication |
| `DATABASE_URL` | Production database connection string |
| `LOG_LEVEL` | Logging verbosity (default: `INFO`) |

All environment variables are documented in `.env.example`. Never hardcode secrets in source code.

## Data Privacy

- No user data is collected, stored, or transmitted beyond what is required for the service to function
- Datasets used for training or evaluation are either publicly available or anonymized
- Logs do not contain personally identifiable information (PII)
- If you believe PII has been exposed, contact us immediately at **privacy@scandiumlabs.com**

## Responsible Disclosure

We ask that you:
1. Allow us a reasonable timeframe (90 days) to fix the issue before public disclosure
2. Do not exploit the vulnerability beyond what is necessary to demonstrate it
3. Act in good faith to help us protect our users and infrastructure

We will publicly acknowledge your contribution once the fix is released (unless you prefer to remain anonymous).
