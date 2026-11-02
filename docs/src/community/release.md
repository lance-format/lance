# Guidelines for Releases

Lance project releases should be automated as much as possible through GitHub Actions.
Such automation includes bumping versions, marking breaking changes, publishing artifacts, and generating release notes.
Overall, our goal is to minimize human interaction beyond initiating the release through GitHub Actions
and voting on the release in GitHub Discussions.

## Release Types

Lance projects follow two types of releases:

- **Preview Releases** (a.k.a. beta releases): Maintainers can publish preview releases at any time to consume the latest changes.
  Preview releases have no stability guarantees and are intended for early testing and feedback.

- **Stable Releases**: Maintainers with write access can initiate stable releases at any time in GitHub Discussions.
  Stable releases must go through a voting process.
  After a stable release is initiated, the community is encouraged to verify the release for any potential bugs and vote for it.
  The PMC is responsible for officially casting binding votes for the stable release. Once the vote has passed,
  a maintainer can continue and finish the stable release.

  Before publishing Lance stable, its LanceDB RC pin PR must be ready to merge:
  relevant CI and reviews must be complete, with no unresolved blocking findings.
  Passing CI alone is insufficient. The actual downstream merge can follow the
  mechanical dependency update and verification of the published stable packages.

## Release Versioning

All Lance projects follow [semantic versioning](https://semver.org/) spec for release versioning:

- **Major version** (`X.0.0`): Incremented for breaking changes that are not backwards compatible
- **Minor version** (`0.X.0`): Incremented for new features that are backwards compatible
- **Patch version** (`0.0.X`): Incremented for critical fixes

Preview releases use the `-beta.X` prerelease suffix appended to the target stable version (e.g., `1.2.3-beta.1`, `1.2.3-beta.2`).

Note that unlike major and minor version releases that are cut from the main branch,
patch version releases should be applied on top of an existing major, minor or patch release commit.
Patch releases should only contain critical fixes for cases such as security vulnerabilities, major correctness issues,
major performance regressions, or reverts of unintended breaking changes.
Any fixes applied in a patch release should have corresponding fixes applied to the main branch.
It is strongly discouraged to continue adding patch releases to old versions.

For major version releases, it is recommended to include a migration guide for users to understand how to
handle any breaking changes introduced in the major version.

## Project Specific Release Process

Each project maintains its own detailed release process in a file named `release_process.md`.
Changes to any project-specific release process are treated as normal code modifications and can be approved by a maintainer with write access.

Project-specific instructions:

- [Lance release process](https://github.com/lance-format/lance/blob/main/release_process.md)
- [LanceDB release process](https://github.com/lancedb/lancedb/blob/main/release_process.md)
