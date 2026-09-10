# Guidelines for Contributing

In general, code contributions are in the form of GitHub PRs, and require review and approval from maintainers with write access.

## Conventional Commits

Lance projects use the [Conventional Commits](https://www.conventionalcommits.org) standard for commit messages.
This standard helps differentiate between:

- **Breaking changes** vs **non-breaking changes** (using `!` and `BREAKING CHANGE:` footer)
- **Features** (`feat:`), **fixes** (`fix:`), **documentation updates** (`docs:`), and other change types

Commit messages following this standard are used to automatically generate release notes during each release.

## Feature Design Proposals

Designs in Lance evolve naturally with community input and consensus.
Major technical changes are discussed organically through the following approach:

- **Start a Discussion**: Create a GitHub Discussion to publish your design proposal and gather community feedback. Use discussion threads to explore different aspects and alternatives
- **Iterate on Design**: Engage with the community to refine the approach based on their input and expertise
- **Draft PRs for Details**: Once the general direction is acceptable to the community, publish draft PRs to help hash out implementation details. Draft PRs are encouraged as they facilitate concrete discussions
- **Break Down Changes**: Split large draft PRs into smaller, incremental PRs for easier review and to demonstrate progress
- **Formal Voting**: Maintainers with write access can approve code modifications related to the design. If the design requires Lance format spec changes, those changes go in their own PR and the PMC votes on that PR following the [voting requirements](./voting.md#voting-requirements)

## Format Specification Changes

Changes to the Lance format specification — the protobuf definitions and the docs
under `docs/src/format/` — are proposed as a pull request and voted on there by the
PMC. The PR is the proposal; there is no separate discussion thread to open first.

Scope such a PR to the specification itself, plus the minimum library changes needed
to keep the build green, and put the implementation in follow-up PRs. See
[Lance Format Specification Changes](./voting.md#lance-format-specification-changes)
for what the vote requires and how it is counted.

## AI Tooling Integrations

We encourage contributors to continuously improve integrations with AI tools, including:

- Enhancing coding agent guidelines such as `AGENTS.md` and `CLAUDE.md`
- Providing feedback to AI code reviewers
- Developing and improving AI-driven GitHub actions

## Project Specific Contributing Guidelines

Each project maintains its own detailed contributing guidelines in files named `CONTRIBUTING.md`.
