# Frequently Answered Questions (Design Decisions & Architecture)

### Q: Why is geometry coverage limited to axis-aligned boxes and a single through-bore on the analytic path?

**A:** Freeform fillets, blends, and arbitrary NURBS surface intersections are where traditional CAD topological errors, self-intersections, and non-manifold failures originate. Vulcan v1 restricts geometry to exact analytical primitives to guarantee 100% watertight closure and absolute mathematical determinism. We build the atomic core first; complex feature sets are layered on top of absolute certainty later.

### Q: Where are fillets, threads, ribs, and freeform surfaces?

**A:** Single-edge fillets, helical threads, and multi-body ribs are **explicitly refused** by V-Wrap's airlock with named diagnostics. For smooth edge-rounding, the level-set path supports `rounded_box` as an explicit, bounded approximation. We reject heuristic guessing: if a feature cannot be mathematically bounded or verified, the system fails closed rather than emitting corrupted CAD models.

### Q: Are there any user-tunable tolerances or configuration knobs?

**A:** No. Vulcan and V-Wrap are built on absolute mathematical determinism with zero adjustable heuristics. Inputs that violate strict boundary checks fail closed immediately. We reject configuration parameters to eliminate parameter drift.

### Q: Why does the level-set path use faceted STEP instead of smooth NURBS?

**A:** For additive manufacturing and CNC roughing/finishing toolpaths, a high-density, watertight faceted representation with exact volume-and-unit verification ($1\text{e}-9$ relative volume tolerance, strict meter units via OCP) is fully sufficient and avoids the brittle patching and stitching errors common in automated NURBS translators.

### Q: Why use Python for an orchestration and geometry pipeline instead of a "fast" systems language like Rust or Go?

**A:** The performance-critical geometry kernels (OpenCascade/OCP via CadQuery) and voxel field evaluations (`acwm`) are already compiled C++/NumPy/C extensions. Python serves strictly as a thin, highly auditable orchestration layer (`vwrap`) and intent parser. Python's clean dataclass semantics, immutability support (`frozen=True`), and rapid test iteration make it ideal for building strict JSON-in/JSON-out compiler airlocks without sacrificing execution speed where it actually matters (the underlying numerical C/C++ loops).

### Q: How do you prevent "config creep," schema bloat, and dead parameters from infiltrating the wrapper over time?

**A:** Through our **meta-test discipline (`test_load_bearing.py`)**. Every single field across all data contracts and response schemas must either be explicitly bound to a functional test proving it alters behavior or registered in an explicit waiver list (`DECORATIVE_OK`). If a developer adds a decorative field or a "future-proofing" parameter without a test, **CI fails automatically**. The codebase enforces its own minimalism.

### Q: Why does the intent parser use local deterministic grammars instead of leveraging an LLM?

**A:** Because LLMs are probabilistic text engines, and Vulcan is a deterministic compilation core. Relying on an LLM to parse raw engineering intents directly inside the core compilation loop introduces non-determinism, hallucination risk, and silent parsing errors. Vulcan uses a local, explicit grammar to extract dimensions and parameters safely, ensuring that if an intent is ambiguous, it fails closed rather than hallucinating geometry.

### Q: How does the system handle concurrency and process isolation when exposed to automated AI agents?

**A:** The architecture treats autonomous agents as untrusted clients. Compilation jobs are isolated in separate spawned processes with strict resource boundaries (such as a fixed 180-second worker execution limit and a strict 16 KiB request payload cap). This guarantees that a runaway agent loop, bad memory allocation, or malformed payload cannot destabilize the host system or poison shared state.

### Q: Why enforce strict fail-closed error handling instead of writing fallback code that tries to fix malformed inputs?

**A:** In traditional software, heuristic fallbacks try to guess what the user meant. In mechanical engineering and CAD, **guessing is a liability.** If a user passes an ambiguous parameter, a sub-voxel feature, or an unrepresentable edge fillet, a fallback turns a silent failure into a physically flawed artifact on a CNC machine or 3D printer. V-Wrap’s fail-closed airlock raises structured, named diagnostics (`UnsupportedFeatureError`, `RequestValidationError`) so an autonomous agent or human engineer receives a precise, machine-parsable explanation of *why* the design violated mathematical boundaries.

### Q: Can I use Vulcan and V-Wrap commercially in a closed-source SaaS product?
**A:** Vulcan is distributed under the GNU Affero General Public License, version 3 only (AGPL-3.0). Under AGPL Section 13, if you offer V-Wrap or Vulcan as a network service or integrate it into a proprietary backend network stack, you are required to open-source your corresponding application layer under the same license, unless you have secured a commercial dual-license.

### Q: Why do I hit dependency or compilation errors when installing CadQuery / OCP locally?
**A:** OpenCascade (OCP) and CadQuery rely on heavy C++ CAD kernels packaged as platform-specific Python wheels. They require specific C++ standard libraries and python toolchains. If local installation fails on your host machine, use the provided Docker container (`docker build --target test -t vulcan-acwm:test .`), which encapsulates the exact Debian/Python base family and shared library runtime dependencies.

### Q: How do I add a custom feature kind (like a blind bore or a rib) to V-Wrap?
**A:** You cannot add it to V-Wrap without first adding it to the Vulcan core (`acwm_cad`). As defined in `vwrap/capability.py`, the wrapper is strictly a proxy; it does not invent geometry. Widening either the analytic or level-set capability set requires implementing the underlying geometric primitive in the core, proving it with OCP round-trip volume checks, and passing the meta-test suite.

### Q: Can I increase the grid resolution past 32 to get higher-fidelity level-set meshes?
**A:** Yes, but with extreme caution. Because level-set voxels and field evaluations scale at $O(N^3)$ complexity, cranking the resolution from 32 to 64 multiplies memory consumption and tensor operations by 8x. The default resource bounds are intentionally conservative to prevent autonomous agents or users from triggering out-of-memory (OOM) crashes on API worker nodes.

### Q: Are generated STEP files and volumes bit-for-bit identical across different host architectures (e.g., Apple Silicon vs. Linux x86)?
**A:** While the analytic path is strictly deterministic, the level-set path's marching cubes tessellation and OpenCascade (OCP) B-rep stitching rely on floating-point arithmetic. Minor cross-architecture IEEE 754 rounding differences can occasionally surface between ARM64 and x86_64 hosts. For strict, reproducible audit trails and pipeline determinism, containerized Linux execution via Docker is strongly recommended.

### Q: Can I use Vulcan's built-in voxel FEA solver for final structural engineering sign-off?
**A:** Absolutely not. Vulcan’s physics pass is explicitly engineered to be a **sub-4-second fail-closed screening filter**, not a full non-linear multi-physics simulation suite. It uses a coarse, structured linear-elastic voxel solver that ignores contact, plasticity, buckling, and thermal conduction. It instantly weeds out structurally illiterate designs, but final manufacturing and structural sign-off require exporting verified geometry to heavy solvers like ANSYS or Abaqus.

### Q: What happens to disk state and partial artifacts when the airlock fails closed?
**A:** Nothing. V-Wrap’s exporter and translation airlock enforce strict transaction isolation: geometry is compiled in memory or written to isolated temporary files, and output paths are modified via atomic replacement only after topological and volume-round-trip validations pass. A failed request leaves zero footprint on disk.

### Q: Does the sign-flip effect tracker guarantee that multi-hole patterns do not accidentally merge into a single slot?
**A:** No. The sign-flip test guarantees that a feature altered the voxel field[cite: 2], but it does not perform post-export OCP shell-component analysis to ensure feature cardinality survived rasterization. If you place pattern holes closer than twice the local voxel size, they can merge into an elongated slot while still passing the wrapper's feature-effect gate. Spacing features safely is the caller's responsibility.

### Q: Are compilation results 100% bit-for-bit reproducible across different host CPU architectures (e.g., ARM64 vs. x86_64)?
**A:** The analytic path is strictly deterministic across platforms. However, the level-set path relies on floating-point tensor grids, marching cubes tessellation, and OpenCascade (OCP) B-rep stitching. Minor cross-architecture IEEE 754 rounding discrepancies can occasionally shift boundary ties ($\phi = 0.0$), making containerized Linux execution the only official environment for reproducible audit trails.