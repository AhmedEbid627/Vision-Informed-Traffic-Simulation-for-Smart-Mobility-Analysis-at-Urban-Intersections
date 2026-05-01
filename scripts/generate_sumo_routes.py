from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# Main drivable approaches at the signalized core of the intersection.
# Using the core incoming/outgoing edges gives us full straight + turning
# connectivity even though some longer outer chains in the imported OSM network
# are only partially routable.
APPROACHES: dict[str, dict[str, str]] = {
    "west": {"in": "29382500#0", "out": "90178163#1"},
    "east": {"in": "681744366#0", "out": "765539624#1"},
    "north": {"in": "8889354#0", "out": "681744358#1"},
    "south": {"in": "69024648#0", "out": "681744359#1"},
}

# Outgoing weights for each incoming approach.
# We bias straight movements highest and keep adjacent turning movements lower.
MOVEMENT_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "west": [("east", 0.55), ("north", 0.20), ("south", 0.25)],
    "east": [("west", 0.55), ("north", 0.25), ("south", 0.20)],
    "north": [("south", 0.55), ("west", 0.25), ("east", 0.20)],
    "south": [("north", 0.55), ("west", 0.20), ("east", 0.25)],
}

# Incoming demand share by approach. Keep balanced until we estimate turn counts
# directly from the video.
INBOUND_WEIGHTS: list[tuple[str, float]] = [
    ("west", 0.32),
    ("east", 0.28),
    ("north", 0.20),
    ("south", 0.20),
]

OPPOSITE_DIRECTION = {
    "west": "east",
    "east": "west",
    "north": "south",
    "south": "north",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate more realistic SUMO routes from observed flow data."
    )
    parser.add_argument(
        "--flow-summary",
        type=Path,
        required=True,
        help="Path to the flow_rate_summary.csv used to extract the observed flow rate.",
    )
    parser.add_argument(
        "--net-file",
        type=Path,
        required=True,
        help="Path to sumo_network.net.xml.",
    )
    parser.add_argument(
        "--output-route",
        type=Path,
        required=True,
        help="Path to the output .rou.xml file.",
    )
    parser.add_argument(
        "--sim-seconds",
        type=int,
        default=3600,
        help="Simulation horizon in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable trip generation.",
    )
    return parser.parse_args()


def read_flow_per_min(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(f"Flow summary CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == "overall_flow_veh_per_min":
                value = float(row[1])
                if value > 0:
                    return value

    print("Warning: Could not read a valid observed flow rate. Defaulting to 10 veh/min.")
    return 10.0


def ensure_required_edges_exist(net_file: Path) -> None:
    if not net_file.exists():
        raise FileNotFoundError(f"Network file not found: {net_file}")

    root = ET.parse(net_file).getroot()
    edge_ids = {edge.get("id") for edge in root.findall("edge")}

    missing: list[str] = []
    for pair in APPROACHES.values():
        for edge_id in pair.values():
            if edge_id not in edge_ids:
                missing.append(edge_id)

    if missing:
        raise ValueError(
            "The SUMO network is missing expected approach edges: " + ", ".join(sorted(set(missing)))
        )


def import_sumolib():
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise OSError("SUMO_HOME environment variable is not set. Cannot import sumolib.")
    tools_path = Path(sumo_home) / "tools"
    if str(tools_path) not in sys.path:
        sys.path.append(str(tools_path))
    import sumolib  # type: ignore

    return sumolib


def compute_valid_movements(net_file: Path) -> tuple[list[tuple[str, float]], dict[str, list[tuple[str, float]]]]:
    sumolib = import_sumolib()
    net = sumolib.net.readNet(str(net_file))

    valid_by_incoming: dict[str, list[tuple[str, float]]] = {}
    filtered_inbound: list[tuple[str, float]] = []

    for incoming, inbound_weight in INBOUND_WEIGHTS:
        from_edge = net.getEdge(APPROACHES[incoming]["in"])
        valid_outgoing: list[tuple[str, float]] = []

        for outgoing, requested_weight in MOVEMENT_WEIGHTS[incoming]:
            to_edge = net.getEdge(APPROACHES[outgoing]["out"])
            path = net.getShortestPath(from_edge, to_edge)
            if path[0] is None:
                continue

            base_weight = requested_weight
            if OPPOSITE_DIRECTION.get(incoming) == outgoing:
                base_weight *= 1.4
            valid_outgoing.append((outgoing, base_weight))

        if valid_outgoing:
            valid_by_incoming[incoming] = valid_outgoing
            filtered_inbound.append((incoming, inbound_weight))

    if not filtered_inbound:
        raise RuntimeError("No valid drivable approach-to-approach movements were found in the network.")

    return filtered_inbound, valid_by_incoming


def weighted_choice(rng: random.Random, options: list[tuple[str, float]]) -> str:
    labels = [label for label, _ in options]
    weights = [weight for _, weight in options]
    return rng.choices(labels, weights=weights, k=1)[0]


def build_trips_xml(
    vehicle_count: int,
    sim_seconds: int,
    output_trip_path: Path,
    seed: int,
    inbound_weights: list[tuple[str, float]],
    movement_weights: dict[str, list[tuple[str, float]]],
) -> None:
    rng = random.Random(seed)
    output_trip_path.parent.mkdir(parents=True, exist_ok=True)

    if vehicle_count <= 0:
        raise ValueError("Vehicle count must be positive.")

    # Spread departures across the horizon with a little jitter.
    base_period = sim_seconds / vehicle_count

    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )

    for idx in range(vehicle_count):
        incoming = weighted_choice(rng, inbound_weights)
        outgoing = weighted_choice(rng, movement_weights[incoming])
        depart = idx * base_period + rng.uniform(-0.35 * base_period, 0.35 * base_period)
        depart = max(0.0, min(float(sim_seconds - 1), depart))

        ET.SubElement(
            routes,
            "trip",
            {
                "id": f"veh_{idx:04d}",
                "depart": f"{depart:.2f}",
                "from": APPROACHES[incoming]["in"],
                "to": APPROACHES[outgoing]["out"],
                "departLane": "best",
                "departSpeed": "max",
            },
        )

    tree = ET.ElementTree(routes)
    ET.indent(tree, space="    ")
    tree.write(output_trip_path, encoding="utf-8", xml_declaration=True)


def run_duarouter(net_file: Path, trip_file: Path, output_route: Path) -> None:
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        raise OSError("SUMO_HOME environment variable is not set. Cannot find duarouter.")

    duarouter = Path(sumo_home) / "bin" / "duarouter.exe"
    if not duarouter.exists():
        duarouter = Path(sumo_home) / "bin" / "duarouter"
    if not duarouter.exists():
        raise FileNotFoundError(f"duarouter not found under {Path(sumo_home) / 'bin'}")

    cmd = [
        str(duarouter),
        "-n",
        str(net_file),
        "--route-files",
        str(trip_file),
        "-o",
        str(output_route),
        "--ignore-errors",
        "--repair",
        "--remove-loops",
    ]

    print("Running duarouter command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    ensure_required_edges_exist(args.net_file)
    flow_per_min = read_flow_per_min(args.flow_summary)
    vehicle_count = max(1, round(flow_per_min * args.sim_seconds / 60.0))
    inbound_weights, movement_weights = compute_valid_movements(args.net_file)

    print(f"Observed flow rate: {flow_per_min:.2f} veh/min")
    print(f"Generating {vehicle_count} vehicles over {args.sim_seconds} seconds")
    print("Using valid movements:")
    for incoming, _ in inbound_weights:
        print(f"  {incoming} -> {', '.join(out for out, _ in movement_weights[incoming])}")

    output_trip_path = args.output_route.with_name(args.output_route.stem + "_trips.xml")
    build_trips_xml(
        vehicle_count=vehicle_count,
        sim_seconds=args.sim_seconds,
        output_trip_path=output_trip_path,
        seed=args.seed,
        inbound_weights=inbound_weights,
        movement_weights=movement_weights,
    )
    print(f"Intermediate trips saved to: {output_trip_path}")

    run_duarouter(args.net_file, output_trip_path, args.output_route)
    print(f"Routes successfully generated at {args.output_route}")


if __name__ == "__main__":
    main()
