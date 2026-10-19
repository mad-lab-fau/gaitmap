# Gaitmap feature review panel

`feature_ready` has five required read-only reviewers: `full_stack`,
`plan_conformance`, `conventions`, `simplicity`, and `scientific_correctness`.
Their rubrics and shared guidance live here; `.roborev.toml` wires them together.
The scientific reviewer covers numerical correctness, literature/reference
implementation fidelity, scientific terminology, evaluation, and supported claims.

## Setup scope

This setup follows Mobgap's tracked RoboRev configuration and five-member panel,
with review guidance adapted to Gaitmap's documentation, algorithms, package split,
and existing `.agent/` tpcp skills. It adopts the same mandatory public compatibility
policy. Agent/model choices remain personal; automatic panel selectors are not set.
The shared `.roborev.toml`, rubrics, policy, and agent instructions are tracked, and
`.gitignore` explicitly permits the config even when a local exclude rule hides it.

Validation uses `roborev config validate --local`, `roborev quickstart --json`,
and Git ignore/diff checks. This change does not alter library behavior.

## Run a review

RoboRev v0.69.0 is the verified configuration baseline. Custom review types need
an authenticated supported agent with native JSON Schema output: Codex must support
`exec --output-schema`; Claude Code must support `--json-schema`.
Panels require the daemon. Its normal review adapters and these rubrics keep
members read-only. Reviewers inspect tests and recorded results; they never run
tests, builds, benchmarks, examples, or implementation workflows.

From the feature checkout, use its actual base:

```bash
roborev config validate --local
roborev review --branch --base <feature-base> --panel feature_ready --wait
```

For a finalized stack, `roborev review --since <stack-base> --panel feature_ready
--wait` is another option. `--local` runs a single review and does not fan out.
Keep the accepted plan, acceptance criteria, approved deviations, and relevant
verification evidence available in the reviewed checkout or a configured include.
Reviewers do not inherit the implementation conversation.

Installation does not start reviews or change automatic-review selectors.
`review.default_panel` and `review.hook_review_panel` are not set in the repository.
Personal selectors still apply if configured. `fix_reasoning = "high"` supplies
synthesis reasoning in v0.69.0 and also affects ordinary fix jobs.

## Personal agent choices

The tracked panel deliberately omits agent and model pins. Custom-type members
inherit custom-type/workflow settings and then generic defaults; they do not
necessarily inherit ordinary `review_agent`/`review_model`. Synthesis inherits the
fix workflow. For example, a developer can merge these top-level settings into
`~/.roborev/config.toml`, before any table headers:

```toml
default_agent = "claude-code"
default_model = "opus"
fix_agent = "claude-code"
fix_model = "opus"
```

This example changes personal defaults across repositories. Preserve existing
settings and remove or adjust conflicting personal pins when intentionally
changing a provider. Panel installation does not modify personal provider settings.

In v0.69.0, repo definitions replace complete global entries with the same name.
A personal `[review.subagents.scientific_correctness]` cannot override the tracked
entry. The `[projects.<remote>]` settings support model/reasoning overrides, but
no agent override. There is no separate `.roborev.local.toml` overlay. Keep
provider choices in your personal generic defaults and fix-workflow settings.
Panel member and synthesis selection is resolved independently of ordinary
single-review `--agent`/`--model` flags; do not rely on those flags to switch an
entire panel's provider. Provider profile files are not required or installed.

Source references for the verified release:

- [Panel definitions, inheritance, and merging](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/config/panels.go)
- [Member and synthesis execution selection](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/daemon/panel_enqueue.go)
- [Custom review types and schema support](https://github.com/kenn-io/roborev/blob/v0.69.0/docs/advanced/custom-review-types.md)
- [Personal project defaults](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/config/projects.go)

## Global skill access and parallel reviews

The shared rubric permits reading applicable installed global style-guide skills,
with repository-local skills taking precedence. For Codex, enable skill instructions
in your personal `~/.roborev/config.toml`:

```toml
[agent.codex]
disable_review_skills = false
```

This allows consulting skills as review criteria, not running their implementation
or recursive review workflows. Existing `ignore_review_user_config` settings can
remain unchanged. Other developers need their own installed skills and settings;
the repository cannot enable personal global skills on their machines.

Concurrency is daemon-wide. Set `max_workers = 10` at the top level of your
personal RoboRev config, before any table headers, to allow up to ten concurrent
jobs. A five-member panel consumes up to five worker slots; synthesis runs after
its members finish. In v0.69.0, changing `max_workers` requires a daemon restart.
Wait for running jobs to finish before using `roborev daemon restart`.

## Scientific source access

Use available read-only online tools to consult original papers, supplements,
reference implementations, and relevant literature. Cite the relevant equation,
section, or code version and distinguish an error from a documented adaptation.
A faithful historical reference implementation need not adopt a newer algorithm.
Names of variables and parameters are in scope when they misstate scientific meaning.

Tool availability differs by agent. RoboRev v0.69.0's
[Claude review adapter](https://github.com/kenn-io/roborev/blob/v0.69.0/internal/agent/claude.go)
auto-allows only `Read`, `Glob`, and `Grep`; it does not auto-allow web tools.
Codex uses a read-only sandbox, but actual network/research access depends on the
installed runtime. Do not enable unsafe agents or bypass restrictions to browse.
When sources are inaccessible, supply local papers/reference excerpts as review
artifacts or configured includes and report the research limitation. Findings
must not imply that inaccessible sources were checked.

## Repository-local tpcp skills

Gaitmap's existing tpcp skill bundle lives in `.agent/`:

- `tpcp-builder`: entry point and common guardrails.
- `tpcp-basics`: algorithms, pipelines, parameters, cloning, and action methods.
- `tpcp-datasets`: dataset indexing and access.
- `tpcp-optimization`: training, optimization, and validation.
- `tpcp-multiprocessing`: parallelism and worker state.

Reviewers use these as inspection criteria only and check applicability to the
supported tpcp version. No duplicate skill bundle is installed by this setup.
