# Contributing to Vulcan

Thank you for contributing to Vulcan. Keep changes focused, testable, and
consistent with the repository's modular design.

## Engineering principles

- **Via negativa:** Prefer the smallest change that fully solves the problem.
  Avoid unnecessary dependencies, abstractions, and configuration.
- **Module boundaries:** Keep HTTP transport and tenant-specific SaaS behavior
  in the API/job-service layer. Keep geometry parsing, orchestration, and
  export logic independent of API credentials, billing, and persistence
  services.
- **Resource hygiene:** Release OpenCascade resources when their lifetime ends,
  and keep preview tessellation separate from precision CAD export.
- **Fail clearly:** Validate inputs at boundaries and report failures rather
  than silently substituting defaults.

## Before submitting

Run the test suite, linter, and bytecode compilation from the project root:

```bash
python -W error -m pytest
python -m ruff check .
python -m compileall -q .
```

Add or update tests for behavior changes. For changes to the selected core
modules, verify they remain independent of API-key handling, usage metering,
and tenant-specific storage.

## Licensing

By submitting a contribution, you agree that it may be distributed under the
GNU Affero General Public License version 3. See [LICENSE](./LICENSE).
