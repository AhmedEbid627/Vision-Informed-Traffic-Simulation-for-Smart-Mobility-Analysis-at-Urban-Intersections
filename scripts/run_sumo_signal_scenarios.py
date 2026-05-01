from __future__ import annotations

import csv
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sumo_dir = project_root / "sumo_simulation"
config_path = sumo_dir / "sumo_config.sumocfg"
routes_path = sumo_dir / "sumo_routes.rou.xml"
vtypes_path = sumo_dir / "vtypes.add.xml"
observed_flow_path = (
    project_root
    / "runs"
    / "track"
    / "infrastructure_batch"
    / "analysis"
    / "infrastructure_1000_flow_rate_summary.csv"
)

scenario_dir = sumo_dir / "signal_scenarios"
tls_id = "cluster_1605546592_28390917_28390982_28783700_#8more"


BASELINE_PHASES = [
    (33, "GGggrrrrGGGggrrrr"),
    (3, "yyggrrrryyyggrrrr"),
    (6, "rrGGrrrrrrrGGrrrr"),
    (3, "rryyrrrrrrryyrrrr"),
    (33, "rrrrGGggrrrrrGGgg"),
    (3, "rrrryyggrrrrryygg"),
    (6, "rrrrrrGGrrrrrrrGG"),
    (3, "rrrrrryyrrrrrrryy"),
]


SCENARIOS: dict[str, dict[str, object]] = {
    "baseline": {
        "description": "Original timing extracted from the SUMO network.",
        "phases": BASELINE_PHASES,
    },
    "favor_west_east": {
        "description": "Longer main green for the west-east corridor, shorter cross direction.",
        "phases": [
            (45, "GGggrrrrGGGggrrrr"),
            (3, "yyggrrrryyyggrrrr"),
            (5, "rrGGrrrrrrrGGrrrr"),
            (3, "rryyrrrrrrryyrrrr"),
            (24, "rrrrGGggrrrrrGGgg"),
            (3, "rrrryyggrrrrryygg"),
            (5, "rrrrrrGGrrrrrrrGG"),
            (3, "rrrrrryyrrrrrrryy"),
        ],
    },
    "favor_north_south": {
        "description": "Longer main green for the north-south corridor, shorter west-east direction.",
        "phases": [
            (24, "GGggrrrrGGGggrrrr"),
            (3, "yyggrrrryyyggrrrr"),
            (5, "rrGGrrrrrrrGGrrrr"),
            (3, "rryyrrrrrrryyrrrr"),
            (45, "rrrrGGggrrrrrGGgg"),
            (3, "rrrryyggrrrrryygg"),
            (5, "rrrrrrGGrrrrrrrGG"),
            (3, "rrrrrryyrrrrrrryy"),
        ],
    },
}


def read_observed_flow(path: Path) -> float:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == "overall_flow_veh_per_min":
                return float(row[1])
    raise ValueError(f"Could not find overall_flow_veh_per_min in {path}")


def write_tls_additional(path: Path, phases: list[tuple[int, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("additional")
    tl = ET.SubElement(
        root,
        "tlLogic",
        {
            "id": tls_id,
            "type": "static",
            "programID": "scenario",
            "offset": "0",
        },
    )
    for duration, state in phases:
        ET.SubElement(tl, "phase", {"duration": str(duration), "state": state})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def run_sumo_for_scenario(
    name: str,
    tls_file: Path | None,
    tripinfo_path: Path,
) -> None:
    additional_files = [str(vtypes_path)]
    if tls_file is not None:
        additional_files.append(str(tls_file))

    cmd = [
        "sumo",
        "-n",
        str(sumo_dir / "sumo_network.net.xml"),
        "-r",
        str(routes_path),
        "--additional-files",
        ",".join(additional_files),
        "--tripinfo-output",
        str(tripinfo_path),
        "-b",
        "0",
        "-e",
        "3600",
        "--no-step-log",
        "true",
        "--verbose",
        "true",
    ]
    print(f"\nRunning scenario: {name}")
    subprocess.run(cmd, check=True)


def summarize_tripinfo(path: Path, sim_seconds: float) -> dict[str, float]:
    tree = ET.parse(path)
    root = tree.getroot()

    trip_count = 0
    total_duration = 0.0
    total_wait = 0.0

    for trip in root.iter("tripinfo"):
        trip_count += 1
        total_duration += float(trip.get("duration", 0))
        total_wait += float(trip.get("waitingTime", 0))

    avg_duration = total_duration / trip_count if trip_count else 0.0
    avg_wait = total_wait / trip_count if trip_count else 0.0
    flow_per_min = (trip_count / sim_seconds) * 60 if sim_seconds > 0 else 0.0

    return {
        "completed_trips": float(trip_count),
        "avg_travel_time_sec": avg_duration,
        "avg_wait_time_sec": avg_wait,
        "flow_rate_veh_min": flow_per_min,
    }


def main() -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)

    if not routes_path.exists():
        raise FileNotFoundError(f"Routes file not found: {routes_path}")
    if not vtypes_path.exists():
        raise FileNotFoundError(f"Vehicle type file not found: {vtypes_path}")
    if not observed_flow_path.exists():
        raise FileNotFoundError(f"Observed flow summary not found: {observed_flow_path}")

    observed_flow = read_observed_flow(observed_flow_path)
    results: list[dict[str, object]] = []

    for name, scenario in SCENARIOS.items():
        tls_file = None
        if name != "baseline":
            tls_file = scenario_dir / f"{name}.add.xml"
            write_tls_additional(tls_file, scenario["phases"])  # type: ignore[arg-type]

        tripinfo_path = scenario_dir / f"{name}_tripinfo.xml"
        run_sumo_for_scenario(name, tls_file, tripinfo_path)
        metrics = summarize_tripinfo(tripinfo_path, 3600.0)

        results.append(
            {
                "scenario": name,
                "description": scenario["description"],
                "completed_trips": int(metrics["completed_trips"]),
                "flow_rate_veh_min": round(metrics["flow_rate_veh_min"], 2),
                "observed_flow_veh_min": round(observed_flow, 2),
                "flow_difference_veh_min": round(abs(metrics["flow_rate_veh_min"] - observed_flow), 2),
                "avg_travel_time_sec": round(metrics["avg_travel_time_sec"], 2),
                "avg_wait_time_sec": round(metrics["avg_wait_time_sec"], 2),
            }
        )

    summary_csv = scenario_dir / "scenario_comparison.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "description",
                "completed_trips",
                "flow_rate_veh_min",
                "observed_flow_veh_min",
                "flow_difference_veh_min",
                "avg_travel_time_sec",
                "avg_wait_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary_txt = scenario_dir / "scenario_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("SUMO Signal Scenario Comparison\n")
        f.write("===============================\n\n")
        f.write(f"Observed flow reference: {observed_flow:.2f} veh/min\n\n")
        for row in results:
            f.write(f"Scenario: {row['scenario']}\n")
            f.write(f"Description: {row['description']}\n")
            f.write(f"Completed trips: {row['completed_trips']}\n")
            f.write(f"Flow: {row['flow_rate_veh_min']} veh/min\n")
            f.write(f"Flow difference vs observed: {row['flow_difference_veh_min']} veh/min\n")
            f.write(f"Average travel time: {row['avg_travel_time_sec']} s\n")
            f.write(f"Average wait time: {row['avg_wait_time_sec']} s\n\n")

    print(f"\nScenario comparison saved to: {summary_csv}")
    print(f"Scenario summary saved to: {summary_txt}")


if __name__ == "__main__":
    main()
