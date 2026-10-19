# Conventions and nitpick review

{{ index .Includes "guidance" }}

Research applicable project and global style guides before judging conventions.
Check naming, module boundaries and exports, dependency injection, error handling,
types, formatting, comments, terminology, and use of established library patterns.
Check a guide's migration status before demanding a planned API or naming convention.
Cite the documented rule and affected location. Report concrete nitpicks as low
severity; avoid personal preferences and speculative consistency requirements.

For tpcp classes, inspect unchanged constructor parameter assignment, validation
in action methods, return-self behavior, trailing-underscore results, clone-safe
mutable defaults, and cloning nested algorithms before execution. Preserve input
data and parameters. Check domain base-class interfaces, private implementation
modules with public __init__/__all__ exports, absolute imports without circular
imports, NumPy docstrings, and existing shared docfillers. Inspect meaningful
TestAlgorithmMixin coverage for new algorithm classes. Do not demand stylistic
changes that are only recommendations or unrelated to this changeset.
