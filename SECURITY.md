# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately by email:

- jeremy@subimage.io

Please do not disclose vulnerabilities publicly in GitHub issues, discussions, or pull requests.

## What to include

A strong report includes:

- A clear description of the vulnerability and impact
- Steps to reproduce (proof of concept if possible)
- Affected versions, environments, and configurations
- Any known mitigations or suggested fixes

## Response process

The project will:

1. Acknowledge receipt as quickly as possible
2. Triage severity and impact
3. Validate and develop a fix
4. Coordinate disclosure and release notes

## Disclosure policy

We aim for responsible disclosure. Public disclosure should happen after a fix is available or agreed with the reporter.

## Security practices in this repository

- CI includes dedicated security workflows (`.github/workflows/security.yml`)
- GitHub Actions are SHA-pinned
- Dependency updates are managed via Dependabot (`.github/dependabot.yml`)
