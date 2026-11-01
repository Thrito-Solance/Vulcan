from __future__ import annotations

import concurrent.futures
import statistics
import tempfile
import time
from pathlib import Path

from acwm_orchestrator import GenerationConfig, MaterialProperties, generate

BENCHMARK_TASKS = (
    {
        "intent": (
            "Steel bracket with 50 x 40 x 10 mm envelope, downward 10 N load "
            "applied to top face, fixed at bottom face."
        ),
        "material": MaterialProperties(
            grade="AISI 304 Stainless Steel",
            young_modulus_pa=193e9,
            poisson_ratio=0.29,
            allowable_stress_pa=215e6,
        ),
        "asymmetric_load": False,
    },
    {
        "intent": (
            "Aluminum joining plate with 80 x 30 x 8 mm envelope, left "
            "10 N load applied to left face, fixed at right face."
        ),
        "material": MaterialProperties(
            grade="Aluminium 6061-T6 (15% environmental aging)",
            young_modulus_pa=68.9e9,
            poisson_ratio=0.33,
            allowable_stress_pa=276e6 * 0.85,
        ),
        "asymmetric_load": True,
    },
    {
        "intent": (
            "Titanium actuator housing with 60 x 60 x 60 mm envelope, "
            "downward 10 N load applied to top face, fixed at bottom face."
        ),
        "material": MaterialProperties(
            grade="Titanium Grade 5",
            young_modulus_pa=113.8e9,
            poisson_ratio=0.34,
            allowable_stress_pa=880e6,
        ),
        "asymmetric_load": False,
    },
    {
        "intent": (
            "Steel linkage arm with 100 x 15 x 15 mm envelope, forward 10 N "
            "load applied to front face, fixed at back face."
        ),
        "material": MaterialProperties(
            grade="Structural Steel",
            young_modulus_pa=200e9,
            poisson_ratio=0.30,
            allowable_stress_pa=250e6,
        ),
        "asymmetric_load": True,
    },
)


def run_single_compilation(job_index: int, task_spec: dict[str, object]) -> dict[str, object]:
    """Compile one benchmark task to an isolated temporary STEP artifact."""
    material = task_spec["material"]
    if not isinstance(material, MaterialProperties):
        raise TypeError("benchmark task material must be MaterialProperties")

    start_time = time.perf_counter()
    result: dict[str, object] = {
        "job_index": job_index,
        "material": material.grade,
        "asymmetric": task_spec["asymmetric_load"],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="vulcan-benchmark-") as tmpdir:
            output_path = Path(tmpdir) / f"benchmark_part_{job_index}.step"
            generation = generate(
                intent_text=str(task_spec["intent"]),
                material=material,
                output_path=output_path,
                config=GenerationConfig(resolution=10, max_iterations=0),
            )
            result["success"] = True
            result["file_size"] = output_path.stat().st_size
            result["volume_m3"] = generation["final_volume_m3"]
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        result["success"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration"] = time.perf_counter() - start_time
    return result


def _p95(durations: list[float]) -> float | None:
    if not durations:
        return None
    if len(durations) == 1:
        return durations[0]
    return statistics.quantiles(durations, n=20, method="inclusive")[18]


def run_benchmark(total_jobs: int = 8, max_workers: int = 4) -> int:
    if total_jobs < 1:
        raise ValueError("total_jobs must be at least 1")
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    print(
        "\nStarting Vulcan concurrency and material-constraint benchmark",
        flush=True,
    )
    print(f"Total jobs: {total_jobs}", flush=True)
    print(f"Process workers: {max_workers}", flush=True)
    print("Cases: standard/asymmetric loads; aged-material stress scaling\n", flush=True)

    tasks = [
        (index, BENCHMARK_TASKS[index % len(BENCHMARK_TASKS)])
        for index in range(total_jobs)
    ]
    results: list[dict[str, object]] = []
    wall_start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_compilation, index, spec): (index, spec)
            for index, spec in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            index, spec = futures[future]
            try:
                result = future.result()
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                material = spec["material"]
                result = {
                    "job_index": index,
                    "material": (
                        material.grade
                        if isinstance(material, MaterialProperties)
                        else "unknown"
                    ),
                    "asymmetric": spec["asymmetric_load"],
                    "success": False,
                    "duration": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            success = result["success"]
            status = "PASS" if success else "FAIL"
            load_type = "Asymmetric" if result["asymmetric"] else "Standard"
            print(
                f"Job {result['job_index']:02d} | {status} | {load_type} | "
                f"{result['material']} | {result['duration']:.2f}s",
                flush=True,
            )
            if not success:
                print(f"       Error: {result.get('error', 'worker failed')}", flush=True)

    wall_duration = time.perf_counter() - wall_start
    successes = [result for result in results if result["success"]]
    failures = len(results) - len(successes)
    durations = [float(result["duration"]) for result in successes]
    p95 = _p95(durations)

    print("\n" + "=" * 58)
    print("VULCAN CONCURRENCY BENCHMARK SUMMARY")
    print("=" * 58)
    print(f"Total wall time: {wall_duration:.2f}s")
    print(
        f"Successful throughput: "
        f"{len(successes) / wall_duration if wall_duration > 0 else 0.0:.2f} jobs/s"
    )
    print(f"Successful builds: {len(successes)}/{total_jobs}")
    print(f"Failed builds: {failures}")
    if durations:
        print(f"Latency mean: {statistics.mean(durations):.2f}s")
        print(f"Latency p50 (median): {statistics.median(durations):.2f}s")
        if p95 is not None:
            print(f"Latency p95: {p95:.2f}s")
    else:
        print("Latency statistics: unavailable (no successful jobs)")
    print("=" * 58 + "\n")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_benchmark(total_jobs=8, max_workers=4))
