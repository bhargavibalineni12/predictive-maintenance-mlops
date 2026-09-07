from pathlib import Path
import sys

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "engine_data.csv"

API_URL = "http://127.0.0.1:8000/predict"


def simulate_normal_traffic(num_requests=100):
    data = pd.read_csv(DATA_PATH)

    sampled_data = data.sample(
        n=num_requests,
        random_state=42,
    )

    successful_requests = 0

    for _, row in sampled_data.iterrows():

        payload = {
            "engine_rpm": float(row["Engine rpm"]),
            "lub_oil_pressure": float(row["Lub oil pressure"]),
            "fuel_pressure": float(row["Fuel pressure"]),
            "coolant_pressure": float(row["Coolant pressure"]),
            "lub_oil_temp": float(row["lub oil temp"]),
            "coolant_temp": float(row["Coolant temp"]),
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            successful_requests += 1
        else:
            print(
                f"Request failed: "
                f"{response.status_code} - {response.text}"
            )

    print(
        f"Simulation complete: "
        f"{successful_requests}/{num_requests} "
        f"requests successful."
    )

def simulate_drifted_traffic(num_requests=500):
    data = pd.read_csv(DATA_PATH)

    sampled_data = data.sample(
        n=num_requests,
        random_state=123,
    ).copy()

    # Intentionally introduce data drift
    sampled_data["Engine rpm"] = (
        sampled_data["Engine rpm"] * 1.5
    )

    sampled_data["Coolant temp"] = (
        sampled_data["Coolant temp"] + 15
    )

    successful_requests = 0

    for _, row in sampled_data.iterrows():

        payload = {
            "engine_rpm": float(row["Engine rpm"]),
            "lub_oil_pressure": float(row["Lub oil pressure"]),
            "fuel_pressure": float(row["Fuel pressure"]),
            "coolant_pressure": float(row["Coolant pressure"]),
            "lub_oil_temp": float(row["lub oil temp"]),
            "coolant_temp": float(row["Coolant temp"]),
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            successful_requests += 1
        else:
            print(
                f"Request failed: "
                f"{response.status_code} - {response.text}"
            )

    print(
        f"Drift simulation complete: "
        f"{successful_requests}/{num_requests} "
        f"requests successful."
    )


if __name__ == "__main__":
    simulate_normal_traffic()