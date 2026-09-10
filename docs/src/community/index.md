# Lance Community Governance

The Lance community is run by volunteers in a collaborative and open way.
Its governance is inspired by open source foundations and projects such as ASF, CNCF, and Substrait.

## Governance Structure

The Lance community recognizes three tiers of participation:

### Contributors

Everyone who has made a contribution to Lance is a contributor.

A "contribution" is not limited to code changes.
Adopting Lance in personal or company projects, providing bug reports and feature requests, performing code reviews,
organizing or planning community gatherings, giving talks, creating and assisting in branding and design,
writing documentation, and many other activities are all counted as contributions.

All contributions, regardless of form, are valued and greatly appreciated.
It is entirely possible to advance through the governance tiers without writing code.

### Maintainers

A maintainer is a contributor who has made sustained and valuable contributions to the Lance community.
Maintainers are recognized for their work and granted various rights to support their ongoing contributions.
For more details of the activities, rights, roster and how to become a maintainer, see [Maintainers](./maintainers.md).

### Project Management Committee (PMC)

A PMC member is a maintainer who has demonstrated leadership in the project.
The PMC guides the long-term direction, makes decisions on governance and project changes, and protects the Lance brand.
For more details of the activities, rights, roster and how to become a PMC member, see [PMC](./pmc.md).

### Roster Information

Maintainer and PMC rosters information follow these guidelines:

- **Ordering**: People in the roster are listed in alphabetical order by last name
- **Self-Report**: Personal information such as _Affiliation_ and _Ecosystem Roles_ is self-reported and updated at the individual's discretion.
- **Ecosystem Roles**: This field documents the individual's involvement in other open source projects if any. It helps identify potential collaboration and integration pathways with the broader open source ecosystem.

## Projects

This section details the projects maintained in the Lance community.

### Core Project

[lance](https://github.com/lance-format/lance) is the core project of the lance-format GitHub Organization,
which hosts most of the development on the table and file format, Rust SDK, Python and Java binding SDKs, documentation and discussions.

The core project is maintained by the Lance community with strict quality and release standards.
[Contributing Guidelines](./contributing.md), [Community Voting Process](./voting.md) and [Release Guidelines](./release.md)
are all applicable to the core project.

### Subprojects

Subprojects are initiatives or repositories that extend Lance's functionality.
They must align with Lance's overall mission and technical direction.
New subprojects are created by graduating from incubating subprojects through a PMC vote.

Subprojects have relaxed requirements compared to core projects:

- Contributors may receive write access even if not maintainers
- Merges may be allowed without review at maintainer discretion
- Release processes may be simplified compared to core projects

Here is the list of current subprojects:

| Project Name      | Repository                                        | Contents                                                               |
|-------------------|---------------------------------------------------|------------------------------------------------------------------------|
| lance-duckdb      | https://github.com/lance-format/lance-duckdb      | DuckDB extension for Lance                                             |
| lance-huggingface | https://github.com/lance-format/lance-huggingface | Hugging Face integration for Lance                                     |
| lance-namespace   | https://github.com/lance-format/lance-namespace   | Lance namespace format specification, Rust/Python/Java Codegen SDKs    |
| lance-namespace-impls | https://github.com/lance-format/lance-namespace-impls | Lance Namespace Implementations - Apache Hive, Apache Polaris, Apache Gravitino, Unity Catalog, AWS Glue and more |
| lance-python-doc  | https://github.com/lance-format/lance-python-doc  | Lance Python SDK generated docs and integration hook with readthedocs  |
| lance-ray         | https://github.com/lance-format/lance-ray         | Ray integration for Lance                                              |
| lance-spark       | https://github.com/lance-format/lance-spark       | Apache Spark connector for Lance                                       |

### Incubating Subprojects

Incubating subprojects are experimental or early-stage repositories in the Lance ecosystem.
Any PMC member can create an incubating subproject without a formal vote.
These projects provide a space for new ideas to develop before committing to full subproject standards.

Incubating subprojects have the most relaxed requirements:

- Anyone can be added as a committer by the project creator or existing PMC members
- Merges without review are allowed
- No formal release process is required

**Important**: All incubating subprojects must include a prominent notice in their README with the following exact notice:

> ⚠️ **Incubating Subproject**: This project is in incubation and is not yet an official Lance subproject.
> APIs and functionality may change without notice. Use it in production at your own risk.

Here is the list of current incubating subprojects:

| Project Name      | Repository                                        | Contents                                            |
|-------------------|---------------------------------------------------|-----------------------------------------------------|
| lance-context     | https://github.com/lance-format/lance-context     | Manage Multimodal Agentic Context Lifecycle with Lance |
| lance-data-viewer | https://github.com/lance-format/lance-data-viewer | Read-only web interface for browsing Lance datasets |
| lance-flink       | https://github.com/lance-format/lance-flink       | Apache Flink connector for Lance                    |
| lance-graph       | https://github.com/lance-format/lance-graph       | Cypher-capable graph query engine on top of Lance   |
| lance-trino       | https://github.com/lance-format/lance-trino       | Trino connector for Lance                           |
| pglance           | https://github.com/lance-format/pglance           | PostgreSQL extension for Lance                      |

### Graduating from Incubating to Subproject

The PMC can vote to promote an incubating subproject to a subproject once the project has demonstrated:

- Proper repository setup including CI, issue tracking, and contributing guide
- Proper code standard enforcement including lint and testing
- Established use cases
- Community adoption outside the primary contributor
- At least one Lance maintainer actively maintaining the project

Contributors with write access will retain their access after graduation to subproject.

### Project License

All Lance projects hosted in the [lance-format](https://github.com/lance-format) GitHub Organization
are licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

### External Integrations

We welcome and encourage Lance integrations in external projects.
These integrations are valuable contributions to the Lance community and help expand the Lance ecosystem.
When integrations are developed in external projects,
the integration code and licensing should follow the guidelines and license of the external project.

## Reporting Security Vulnerability

In case of any security vulnerability,
please contact the PMC through the [Lance Private Mailing List](mailto:private@lance.org)
and refrain from public disclosure until the issue is resolved.

## Reporting Harassment

The Lance community follows the [Rust Community Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
We are committed to providing a welcoming and inspiring community for all.
Harassment of participants will not be tolerated.
For such cases, please report to the [Lance Private Mailing List](mailto:private@lance.org).
