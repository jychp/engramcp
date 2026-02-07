# Reviewing Pull Requests

This guide defines review expectations for EngraMCP.

## Review goals

Every review should optimize for:

- Correctness
- Security
- Maintainability
- Test coverage
- Documentation accuracy

## Reviewer checklist

- Confirm the change solves the stated problem
- Verify tests cover the behavior change and regressions
- Check edge cases and failure modes
- Validate compatibility with MCP tool contracts (`send_memory`, `get_memory`, `correct_memory`)
- Ensure docs are updated when needed (`README.md`, `AGENTS.md`, `docs/design/*`)

## Author checklist before requesting review

- Local checks pass:

```bash
uv run pytest
uv run pre-commit run --all-files
```

- PR description is complete and actionable
- Linked issues are included where relevant
- Scope is focused and avoids unrelated edits

## Review process

1. Author opens a PR with context and test evidence
2. Reviewer requests changes or approves
3. Author addresses feedback
4. Maintainer merges once all checks pass

## Security-sensitive changes

For changes touching auth, secrets, dependencies, container/image pipelines, or execution boundaries:

- Require at least one maintainer review
- Prefer explicit threat/abuse-case notes in the PR description
- Verify no sensitive data is committed
