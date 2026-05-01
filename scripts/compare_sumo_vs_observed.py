import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Compare SUMO tripinfo against observed metrics.")
    parser.add_argument("--tripinfo", required=True, type=Path, help="Path to sumo_tripinfo.xml")
    parser.add_argument("--observed", required=True, type=Path, help="Path to observed flow_rate_summary.csv")
    parser.add_argument("--output", default="sumo_comparison.csv", type=Path, help="Output comparison CSV")
    parser.add_argument("--sim-seconds", default=3600, type=float, help="Simulation horizon in seconds.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.tripinfo.exists():
        print(f"Error: Tripinfo file {args.tripinfo} not found.")
        return
        
    tree = ET.parse(args.tripinfo)
    root = tree.getroot()
    
    trip_count = 0
    total_duration = 0.0
    total_wait = 0.0
    
    for trip in root.iter('tripinfo'):
        trip_count += 1
        total_duration += float(trip.get('duration', 0))
        total_wait += float(trip.get('waitingTime', 0))
        
    avg_duration = total_duration / trip_count if trip_count > 0 else 0
    avg_wait = total_wait / trip_count if trip_count > 0 else 0
    
    # Estimate overall flow rate in simulation based on a 3600 second simulation
    sim_flow_per_min = (trip_count / args.sim_seconds) * 60 if args.sim_seconds > 0 else 0
    
    # Read observed
    obs_flow_per_min = 0.0
    if args.observed.exists():
        with open(args.observed, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "overall_flow_veh_per_min":
                    obs_flow_per_min = float(row[1])
                    break
    
    print("=== SUMO Simulation Results ===")
    print(f"Completed Trips:  {trip_count}")
    print(f"Avg Travel Time:  {avg_duration:.2f} s")
    print(f"Avg Wait Time:    {avg_wait:.2f} s")
    print(f"Sim Flow Rate:    {sim_flow_per_min:.2f} veh/min")
    print("===============================")
    if obs_flow_per_min > 0:
        print(f"Obs Flow Rate:    {obs_flow_per_min:.2f} veh/min")
        diff = obs_flow_per_min - sim_flow_per_min
        print(f"Flow Difference:  {abs(diff):.2f} veh/min ({(abs(diff)/obs_flow_per_min)*100:.1f}%)")
        
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Simulated", "Observed"])
        writer.writerow(["Flow_Rate_Veh_Min", round(sim_flow_per_min, 2), round(obs_flow_per_min, 2)])
        writer.writerow(["Avg_Travel_Time_Sec", round(avg_duration, 2), "N/A"])
        writer.writerow(["Avg_Wait_Time_Sec", round(avg_wait, 2), "N/A"])
        
    print(f"\nComparison saved to {args.output}")

if __name__ == "__main__":
    main()
