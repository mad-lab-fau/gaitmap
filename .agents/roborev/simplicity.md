# Simplicity and necessity review

{{ index .Includes "guidance" }}

Search the library for existing capabilities before recommending new abstractions.
Question redundant wrappers, state, adapters, options, dependencies, and speculative
generality. Identify code that can be removed or replaced with a simpler existing API.
Check compatibility machinery against the project's actual policy.
For each finding, explain the viable simpler approach and the contracts it preserves.
Preserve required security, ownership, lifetime, and supported behavior guarantees.

Look first for existing gaitmap transforms, dtype/frame helpers, iteration,
evaluation, and aggregation utilities. Prefer native NumPy/pandas outputs unless
a custom type adds clear value. Preserve independently usable pipeline stages
and the minimal inputs each algorithm needs. Avoid gratuitous sensor-specific
assumptions, duplicate scientific logic, and validation beyond supported input
contracts. Do not remove compatibility behavior required by the published API.
