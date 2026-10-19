Review the full supplied changeset as one feature. Read related implementations,
callers, tests, manifests, and documentation to verify findings. Repository access
is for research: do not edit files, build, run tests, or delegate the review.

Read applicable AGENTS.md instructions and .agents/refactor-policy.md when present.
Follow explicit task requirements and accepted decisions before general preferences.
Respect this project's actual compatibility requirements and supported contracts.

Consult applicable repository-local and installed global style-guide skills and
their supporting documents. Repository-local versions take precedence over global
versions with the same name. Selected global skill documents are an exception to
repository-only research. Use their rules as review criteria; do not run skill
implementation, commit, fix, publishing, or recursive review workflows.

Report concrete findings with a location, evidence, impact, and useful correction.
Distinguish missing evidence from demonstrated defects. Do not invent requirements
or report unrelated existing problems unless the changeset introduces or exposes them.

Reserve critical for credential compromise, remote code execution, or widespread
irreversible data loss; high for a broken primary workflow or serious security defect;
medium for incorrect supported behavior or a substantial architecture or requirement
violation; low for concrete naming, style, documentation, or maintainability issues.
Report low findings too. RoboRev owns the output schema and severity threshold.

Gaitmap is a published scientific Python library and reference implementation.
Read docs/source/development/project_structure.md,
docs/source/development/development_guide.md,
docs/source/user_guide/coordinate_systems.rst,
docs/source/user_guide/datatypes.rst, and relevant algorithm/base-class documentation.
The project structure guide describes recommendations; distinguish contracts from
preferences. The accepted .agents/refactor-policy.md governs compatibility even
where older release guidance is less strict. Treat numerical outputs, defaults,
DataFrame schemas, and persisted formats as supported behavior where documented.

All panel members are read-only. Do not execute tests, builds, benchmarks,
examples, package installation, or arbitrary code. Inspect existing tests,
snapshots, and supplied CI/validation artifacts. Missing or skipped verification
is a limitation, not by itself proof of a defect. Snapshot updates and relaxed
tolerances require an explanation; they do not establish correctness. Report
unverified scientific claims separately from demonstrated implementation errors.
Use online sources only through available read-only research tools. Do not bypass
an agent's tool restrictions or enable write access to obtain research access.

Consult .agent/tpcp-builder/SKILL.md and relevant sibling skills for tpcp
usage. Apply their rules as inspection criteria only. Check the installed tpcp
version and existing supported behavior before requiring a newer API or convention.
Use current pyproject.toml and .github/workflows/test-and-lint.yml for supported
Python/dependency/platform requirements and verification commands. CI currently
covers Python 3.9 through 3.11 on Ubuntu and Windows. Check
optional imports, packaged resources, and cross-platform paths when affected.
Review both gaitmap and packages/gaitmap_mad when a change crosses their boundary;
preserve documented optional-package behavior and licensing separation.

Check public docstrings, examples, exports, and release notes for changed supported
behavior. Never imply that passing regression tests validates new cohorts,
sensor placements, parameterizations, or modified pipelines scientifically.
