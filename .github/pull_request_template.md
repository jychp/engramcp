### Type of change
<!-- Mark the relevant option with an "x" -->
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Refactoring (no functional changes)
- [ ] Documentation update
- [ ] Other (please describe):


### Summary
<!-- Describe WHAT your changes do and WHY they are needed. -->



### Related issues or links
<!-- Include links to relevant issues or other pages. Use "Fixes #123" or "Closes #123" to auto-close issues. -->

- Fixes #


### Sprint / Layer
<!-- Which sprint and layer(s) does this PR target? -->

- Sprint:
- Layer(s):


### Breaking changes
<!-- If this PR introduces breaking changes, describe the impact and migration path. Otherwise, delete this section. -->



### How was this tested?
<!-- Describe how you tested your changes. Include relevant details such as test configuration, commands run, or manual testing steps. -->



### Checklist

#### General
- [ ] The linter passes locally (`pre-commit run --all-files`).
- [ ] I have added/updated tests that prove my fix is effective or my feature works.
- [ ] All previous sprint tests still pass (no regression).

#### Documentation
- [ ] `CLAUDE.md/AGENTS.md` updated if architecture, conventions, or structure changed.
- [ ] `README.md` updated if user-facing behavior changed.
- [ ] Design doc created/updated in `docs/design/` if needed.

#### If changing the MCP interface
- [ ] API contract is backward-compatible or version bumped.

#### If changing the graph schema
- [ ] Ontology schema doc updated.
- [ ] Indexes and constraints verified.

#### If modifying confidence or consolidation logic
- [ ] Epistemic safeguards maintained (no false confidence amplification).
- [ ] Audit log covers all mutations.


### Notes for reviewers
<!-- Optional: Add any context that would help reviewers, such as areas to focus on, design decisions, or open questions. -->
