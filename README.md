# ACWM research prototype

This repository is an executable numerical research prototype for implicit
domain design. It is not a production CAD system, validated engineering
solver, or trained world model.

## Install and run

```powershell
python -m pip install -e ".[test]"
pytest
python -m acwm demo --output acwm-output --resolution 18
```

The demo constructs a box SDF, subtracts a spherical exclusion, extracts and
checks the triangle surface, voxelizes its interior into linear hexahedral
elements, solves a small-strain isotropic elasticity problem, and writes
`design.stl` and `report.json`.

## Container API and agent runner

Install the optional cloud dependencies locally with
`python -m pip install -e ".[cloud]"`. The `cloud` extra installs the FastAPI
compiler API and CadQuery.
`acwm_api` accepts authenticated compile requests, runs each CAD job in a
separate spawned process, limits concurrent jobs, and stores artifacts under
`ACWM_JOBS_DIR`. Request bodies are capped at 16 KiB, including chunked
requests. The agent accepts HTTPS endpoints or the local/Compose HTTP hosts
`localhost`, `127.0.0.1`, `::1`, and `api`; it rejects redirects so its bearer
token cannot be forwarded to another origin. The API intentionally binds to
loopback in Compose by default; put a TLS-terminating authenticated proxy in
front of it before remote use.
Do not expose the container port directly to an untrusted network.
Configure API credentials as comma-separated `token:client_id` entries in
`VULCAN_API_KEYS`; keys must contain at least 32 characters. Jobs are owned by
the authenticated client, and a different client's key receives a not-found
response when attempting to download them. For the Compose agent, configure
`VULCAN_API_TOKEN` to one of the tokens present in `VULCAN_API_KEYS`.

The `agent` Compose profile runs `acwm_agent_loop`, which can submit a single
intent or JSONL requests, independently imports each returned STEP file,
requires exactly one valid positive-volume solid, and appends JSONL telemetry.
Telemetry stores a prompt hash rather than the prompt text and records STEP
volume from the imported solid. Compiler-produced STEP artifacts declare
metres, so this volume is in m³. A shared named volume persists job artifacts
and telemetry.
The API serves the matching dark-mode browser UI at `/`; its authenticated
download route is `/v1/download/{job_id}/{file_type}`, where `file_type` is
`step` or `stl`. Compatibility routes remain available at
`/v1/jobs/{job_id}/preview.stl` and `/v1/jobs/{job_id}/model.step`. The UI
uses a pinned Three.js CDN import map,
so browser access to `cdn.jsdelivr.net` is required. Because the UI is served
from the API origin, CORS is not needed; do not enable permissive cross-origin
access for the bearer-token endpoints.

The browser holds the token only in memory, clears it after each operation,
and renders server-provided errors with `textContent`. Compile bodies are
restricted to fewer than 16 KiB client-side, with the server enforcing the
16 KiB hard cap. Manufacturing toggles are passed to the existing
directional-normal screens; they are heuristic filters, not process
qualification.

```powershell
$env:ACWM_API_TOKEN = [guid]::NewGuid().ToString('N') + [guid]::NewGuid().ToString('N')
$env:VULCAN_API_TOKEN = $env:ACWM_API_TOKEN
$env:VULCAN_API_KEYS = "$($env:ACWM_API_TOKEN):local-dev"
docker compose build api
docker compose up -d api
docker compose ps
# Open http://127.0.0.1:8000/ in a browser
docker compose --profile agent run --rm agent `
  --intent "Steel bracket with 60 x 60 x 60 mm envelope, downward 10 N load applied to top face, fixed at bottom face." `
  --material-grade "test steel" `
  --young-modulus-pa 200000000000 `
  --poisson-ratio 0.3 `
  --allowable-stress-pa 1000000000
docker compose --profile agent run --rm `
  -v "${PWD}\requests.jsonl:/requests.jsonl:ro" `
  agent --request-file /requests.jsonl
docker build --target test -t vulcan-acwm:test .
```

The JSONL option expects one JSON object per line with `intent`,
`material_grade`, `young_modulus_pa`, `poisson_ratio`, and
`allowable_stress_pa`; `resolution` and `max_iterations` are optional.

The API worker has a fixed 180-second execution limit and the agent HTTP
timeout defaults to 190 seconds. The API accepts a maximum grid resolution of
32 and eight carving iterations per request. These are resource bounds, not
validated performance guarantees. OCP/CadQuery wheels are platform-specific;
the Dockerfile builds and installs wheels on the same Debian/Python base family
and includes the runtime shared-library packages. The local environment must
provide Docker Engine and the Compose plugin to build or run the image.

Optional native CAD export and FNO training dependencies are separate:

```powershell
python -m pip install -e ".[cad]"
python -m pip install -e ".[surrogate]"
python -m pip install -e ".[edge]"
```

## B-rep feature bridge

`acwm_cad.fit_box_and_one_through_bore` fits the narrow supported feature
family—an axis-aligned box and optionally one straight cylindrical through-bore—
to a sampled SDF. It optimizes feature dimensions and rejects the fit unless
the maximum absolute error over the full input grid is within the caller's
tolerance. `fit_sdf_and_export` then exports that verified feature model as a
native STEP or IGES solid when the optional CadQuery/OpenCascade backend is
installed. All SDF dimensions are in meters; exported CAD geometry is scaled to
millimeters.

This is bounded numerical primitive fitting, not a neural surface-fitting
system. It does not recognize arbitrary topology, rotated boxes, curved
surfaces, multiple/angled/blind bores, blends, fillets, or G1/G2 transition
surfaces. Unsupported inputs fail the SDF-error threshold instead of being
silently approximated as a machinable solid. The STEP/IGES smoke tests import
the files back through OpenCascade and check that each round trip contains one
valid solid with the expected volume. Passing those tests is not manufacturing
qualification.

`acwm_export.export_phi_to_step` provides a separate faceted B-rep path for a
validated level-set surface. It validates the triangle mesh, sews its planar
faces with OCP, requires one closed shell and one valid positive-volume solid,
then writes metre-unit STEP to a temporary file, verifies the unit metadata and
round-trip B-rep volume, and atomically replaces the target. The reported
`final_volume_m3` is computed from the exported level-set surface, so it
matches the actual faceted STEP geometry rather than the analytic primitive
volume estimate.
The orchestrator uses this path for STEP/STP output; IGES continues to use the
verified analytic primitive exporter. Faceted STEP preserves the extracted
surface but is not smooth NURBS reconstruction and can be larger than analytic
primitive output.

## Manufacturing screens

`acwm_manufacturing` estimates outward surface normals by smoothing the SDF,
computing its spacing-aware gradient, sampling at surface vertices, and
rejecting singular or non-finite gradients. Optional CNC directional-normal
and additive overhang screens can be enabled in `GenerationConfig` or with
`--manufacturing-process cnc|additive|both`. The CNC check is a local normal
screen for the supplied tool-pull axis, not a ray-visibility, toolpath, fixture,
or multi-setup accessibility solver. The additive check measures downward
surface-normal angle from vertical; it does not model supports, material,
thermal distortion, or process parameters. These are heuristic proposal filters, not manufacturing
qualification. Manufacturing screens are disabled unless a process is
explicitly selected.

## Physics surrogate scaffold

`acwm_surrogate.FNO3d` is a 3D Fourier Neural Operator with configurable input
and output channels. Its default nine output channels are intended for six
Voigt stress components and three displacement components; input channel
semantics, normalizations, units, and any temperature channels belong to the
dataset/model configuration and are not inferred by the network.

`train_fno` requires caller-supplied finite simulation inputs/targets and
geometry-family labels. It keeps whole geometry groups out of the validation
split to reduce design-family leakage. Checkpoints load with PyTorch's
weights-only loader. `predict` refuses to produce results from a fresh,
untrained model. The repository provides no training corpus, model weights,
accuracy guarantee, milliseconds latency guarantee, or fallback to FEA.
Predictions are not suitable for engineering decisions until independently
validated against held-out, high-fidelity analyses over the intended design
domain.

## Bounded closed-loop orchestration

The CLI joins the intent parser, implicit SDF, filtered level-set proposals,
topology checks, coarse FEA, primitive fitting, and native CAD export:

```powershell
python -m acwm_orchestrator --intent "Steel bracket with 60 x 60 x 60 mm envelope, downward 10 N load applied to top face, fixed at bottom face." --output bracket.step --material-grade "user-specified steel grade" --young-modulus-pa 200000000000 --poisson-ratio 0.3 --allowable-stress-pa 1000000000
```

The caller must supply material properties and an allowable stress. The
orchestrator refuses incomplete geometry, missing support/load faces, an
over-limit initial FEA result, non-manifold extracted surfaces, invalid CAD
solids, and output overwrite without `--overwrite`. It proposes a small number
of conservative Helmholtz-filtered level-set steps using the sampled FEA stress
field. Each proposal must reinitialize, pass mesh topology checks, fit the
supported box/through-bore family within the explicit CAD fit tolerance, remove
resolved grid material without adding any, and pass another FEA check before
acceptance. Rejected proposals are recorded and rolled back to the last
verified state. Fixture and load planes remain anchored to the initial design
datums; a candidate that removes a required boundary is rejected rather than
silently relocating its boundary conditions.

An optional `--surrogate-checkpoint` enables previews from a trained 8-input,
9-output FNO. The preview inputs are normalized SDF, occupancy, normalized
coordinates, and load direction. The FNO's raw output is recorded for
inspection only: it is not interpreted as calibrated stress, does not select
or accept designs, and never replaces an FEA verification. No pretrained,
validated weights are included. Increase `--resolution` only with awareness
that structured voxel FEA cost grows rapidly with grid size.

The optional PyTorch edge module can be supplied to the Python
`generate(..., edge_surrogate=...)` API. A trained `TinyEdgeSurrogate` provides
a bounded, subtractive-only local SDF correction and local stress estimates.
Its `global_compliance_error` context value is a normalized peak-displacement
proxy, not a calibrated compliance sensitivity.
The MacroMicroGovernor screens the initial FEA state and each candidate against
the supplied allowable and topology result; global FEA and KS checks remain
mandatory. Local stress predictions can only veto a candidate, never authorize
it. An untrained or failing edge model falls back to global evolution by
default, with the reason recorded in the result. Set
`strict_edge_inference=True` to turn those cases into explicit errors. The
repository includes no trained edge weights, so the default run does not
perform edge inference.

This loop is a constrained research demonstration, not autonomous
manufacturing approval. Its linear-elastic voxel FEA, user-supplied allowables,
and primitive fit do not certify safety or manufacturing suitability. The
analytic CAD family cannot represent general topology optimization; when an
evolved shape leaves that family or misses the fit tolerance, the candidate is
rejected and the last valid design is retained. STEP output is a faceted
level-set B-rep; IGES output uses the analytic primitive model. Neither is a
released engineering design.

## Sensitivity screening

`acwm_sensitivity.run_sensitivity_sweep` evaluates the Cartesian product of
caller-supplied load vectors, stress allowables, Helmholtz radii, KS
aggregation parameters, and `MacroContext` values. Each proposed carved SDF is
topology-checked and re-solved with the coarse voxel FEM using the requested
support/load faces, anchored to their baseline coordinates. Macro stress
scaling is applied before the KS and
MacroMicroGovernor checks; macro compliance and carving aggressiveness adjust
the proposed time step. Constraint-plane velocities are clamped to zero, and
positive normal velocities are rejected so a sweep cannot expand material or
silently relocate fixtures. Every case starts from the same baseline; results
report baseline/candidate SHA-256 fingerprints rather than retaining large
voxel arrays, and a rejected case's committed fingerprint matches baseline.
The Cartesian product is limited to 256 cases by default; callers can raise
`max_cases` explicitly or split larger sweeps into bounded batches.

This is a deterministic parameter screen, not a statistical uncertainty
analysis, material database, certified sensitivity derivative, or production
optimization. Its acceptance results are limited by the prototype FEM and
discretized topology checks. It does not emit CAD. Inputs that are malformed
or have an invalid baseline raise explicit errors; candidate failures are
recorded as rollbacks. The sensitivity test suite exercises both accepted
candidate checks and fail-closed rollback behavior.

## Natural-language intent parser

`python -m acwm_intent --text "..."` (or `acwm-intent --text "..."` after
installation) runs a local deterministic grammar; it does not call an LLM,
access a network, or autonomously run an optimization loop. It extracts a
limited set of dimensions, supported directional force, shaft dimensions,
shaft axis/center, clearance, material family, and named support/load faces.
The output includes missing requirements and readiness flags. For example:

```powershell
python -m acwm_intent --text "Titanium motor bracket with 120 x 80 x 40 mm envelope, 50 mm diameter shaft along z-axis, shaft center at (0, 0, 0) mm, 2 mm radial clearance, fixed at left face, downward 2,000 N load applied to right face."
```

`intent_to_primitive` maps only a complete supported envelope/shaft description
to the analytic CAD feature model. The parser does not infer a material grade,
design allowables, load paths, fixtures, shaft position, or missing units. Its
shaft-size interpretation is explicitly reported as an assumption that must be
confirmed. Analysis readiness remains false while material grade and verified
allowables are absent. When geometry inputs are complete, the CLI constructs a
sampled SDF for the analytic envelope and exclusions. Use `--export
mount.step` or `--export mount.iges` to export those explicit analytic features;
this does not run FEA, stress-driven carving, or manufacturing qualification.

## Numerical and safety boundaries

- The discrete KS stress aggregate is a conservative smooth upper bound on the
  largest sampled stress, evaluated with log-sum-exp. It is mesh-dependent,
  not a converged continuum stress bound.
- Max-based SDF booleans preserve the inside/outside set but do not generally
  preserve exact signed-distance magnitudes. Reinitialization is a grid
  approximation and can move the zero contour by a fraction of a cell.
- Mesh validation checks edge incidence, directed-edge winding, connected
  vertex links, unused vertices, and the Euler/genus relation for closed
  orientable triangle surfaces. It does not prove the absence of geometric
  self-intersections or certify a CAD kernel's B-rep.
- Topology may change during continuous level-set evolution; only extracted
  meshes are checked. The prototype does not guarantee watertightness at every
  intermediate iteration.
- The FEM is a coarse, structured, 8-node hexahedral linear-elastic solver with
  zero-displacement constraints and optional isotropic thermal strain. It is
  not a validated engineering solver and does not model contact, plasticity,
  buckling, thermal conduction, or manufacturing process effects.
- Level-set evolution accepts an externally supplied normal-velocity field; no
  stress-to-shape sensitivity or closed-loop topology optimizer is included.
  The Helmholtz filter uses homogeneous Neumann boundary conditions.
- Tolerance coefficients and units require calibration for any real process.

Production use still requires independently validated solvers, calibrated
material/process data, geometric self-intersection checks, a wider
surface-to-B-rep pipeline, and manufacturing qualification.

## License

Vulcan is distributed under the GNU Affero General Public License, version 3
only. See [LICENSE](./LICENSE) for the complete license text.
