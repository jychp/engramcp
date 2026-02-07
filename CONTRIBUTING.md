# Contributing to EngraMCP

Thanks for your interest in contributing to EngraMCP.

## Ways to contribute

- Report bugs or request features in [GitHub Issues](https://github.com/jychp/engramcp/issues)
- Improve code, tests, or documentation
- Review pull requests

## Development setup

### Prerequisites

- Python `3.13+`
- `uv`
- Docker (required for integration tests using testcontainers)

### Local setup

```bash
git clone git@github.com:jychp/engramcp.git
cd engramcp
uv sync --dev
pre-commit install
```

## Development workflow

1. Create a branch from `main`
2. Write/adjust tests first when possible (outside-in TDD)
3. Implement the change
4. Run local checks:

```bash
uv run pytest
uv run pre-commit run --all-files
```

5. Update docs when needed (`README.md`, `AGENTS.md`, and `docs/design/*`)
6. Open a pull request using `.github/pull_request_template.md`

## Coding standards

- Language for code, comments, commit messages, docs, and issues: English (US)
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- Do not mention internal sprint numbers in commit messages
- Keep architecture and design references up to date in `AGENTS.md`

## Pull request expectations

- Keep PRs focused and reviewable
- Include tests for behavior changes
- Explain design tradeoffs in the PR description
- Ensure CI is green before merge

## Security

Please do not open public issues for vulnerabilities.
See `SECURITY.md` for private reporting instructions.
