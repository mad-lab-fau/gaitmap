# Backwards-compatibility policy

- Backwards compatibility is mandatory for all public interfaces, user-facing behavior, and persisted data formats.
- Internal refactors are allowed when safe; all in-repository consumers must be updated together and existing public behavior preserved.
- Any public breaking change requires the project maintainer's explicit prior approval, a migration plan, and prominent release communication.
