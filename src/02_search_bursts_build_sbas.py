#!/usr/bin/env python3
"""Search Sentinel-1 bursts and build a reviewable multi-burst SBAS job plan."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import asf_search as asf
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform, unary_union


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_aoi(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    features = data.get("features", [])
    if len(features) != 1:
        raise ValueError("AOI must contain exactly one feature")
    geom = shape(features[0]["geometry"])
    if not geom.is_valid:
        raise ValueError("AOI geometry is invalid")
    return geom


def burst_record(product: Any) -> dict[str, Any]:
    props = product.properties
    burst = props["burst"]
    return {
        "scene_name": props["sceneName"],
        "acquisition_date": props["startTime"][:10],
        "start_time": props["startTime"],
        "platform": props.get("platform"),
        "flight_direction": props.get("flightDirection"),
        "relative_orbit": props.get("pathNumber"),
        "polarization": props.get("polarization"),
        "subswath": burst.get("subswath"),
        "relative_burst_id": burst.get("relativeBurstID"),
        "full_burst_id": burst.get("fullBurstID"),
        "burst_index": burst.get("burstIndex"),
        "absolute_orbit": props.get("orbit"),
        "group_id": props.get("groupID"),
        "url": props.get("url"),
    }


def is_connected(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    if not nodes:
        return False
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    visited = {nodes[0]}
    queue = deque([nodes[0]])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node] - visited:
            visited.add(neighbour)
            queue.append(neighbour)
    return len(visited) == len(nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    root = config_path.parent
    config = load_yaml(config_path)
    aoi = load_aoi((root / config["aoi"]["path"]).resolve())
    search_cfg = config["search"]
    track = config["approved_track"]
    burst_cfg = config["burst_search"]
    coverage_cfg = config["coverage_filter"]
    sbas_cfg = config["sbas"]

    results = asf.geo_search(
        dataset=asf.DATASET.SLC_BURST,
        intersectsWith=aoi.wkt,
        start=search_cfg["start"],
        end=search_cfg["end"],
        relativeOrbit=track["relative_orbit"],
        flightDirection=track["flight_direction"],
        polarization=track["polarization"],
        maxResults=burst_cfg["maximum_results"],
    )
    if not results:
        raise RuntimeError("ASF returned no burst products")

    products = sorted(results, key=lambda p: (p.properties["startTime"], p.properties["sceneName"]))
    records = [burst_record(product) for product in products]
    inventory = pd.DataFrame(records)

    burst_out = (root / config["outputs"]["burst_directory"]).resolve()
    network_out = (root / config["outputs"]["network_directory"]).resolve()
    burst_out.mkdir(parents=True, exist_ok=True)
    network_out.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(burst_out / "burst_inventory.csv", index=False)

    features = [product.geojson() for product in products]
    (burst_out / "burst_inventory.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )

    dates = sorted(inventory["acquisition_date"].unique())
    occurrence = inventory.groupby("full_burst_id")["acquisition_date"].nunique()
    stable_full_ids = sorted(occurrence[occurrence == len(dates)].index)
    stable = inventory[inventory["full_burst_id"].isin(stable_full_ids)]
    burst_definitions = (
        stable[["full_burst_id", "relative_burst_id", "subswath"]]
        .drop_duplicates()
        .sort_values(["subswath", "relative_burst_id"])
    )

    if len(stable_full_ids) > burst_cfg["maximum_bursts_per_job"]:
        raise ValueError("Selected burst set exceeds HyP3 multi-burst limit")
    subswaths = sorted(burst_definitions["subswath"].unique())
    if burst_cfg["require_same_subswath"] and len(subswaths) != 1:
        raise ValueError(f"Burst set spans multiple subswaths: {subswaths}")
    relative_ids = sorted(int(x) for x in burst_definitions["relative_burst_id"])
    contiguous = relative_ids == list(range(min(relative_ids), max(relative_ids) + 1))
    if burst_cfg["require_contiguous_relative_burst_ids"] and not contiguous:
        raise ValueError(f"Relative burst IDs are not contiguous: {relative_ids}")

    transformer = Transformer.from_crs(
        coverage_cfg["source_crs"], coverage_cfg["area_crs"], always_xy=True
    )
    projected_aoi = transform(transformer.transform, aoi)
    products_by_date: dict[str, list[Any]] = defaultdict(list)
    for product in products:
        if product.properties["burst"]["fullBurstID"] in stable_full_ids:
            products_by_date[product.properties["startTime"][:10]].append(product)

    date_rows = []
    for date in dates:
        date_products = products_by_date[date]
        projected_bursts = [
            transform(transformer.transform, shape(product.geojson()["geometry"]))
            for product in date_products
        ]
        union = unary_union(projected_bursts)
        coverage = 100.0 * union.intersection(projected_aoi).area / projected_aoi.area
        date_rows.append({
            "acquisition_date": date,
            "burst_count": len(date_products),
            "aoi_coverage_percent": round(coverage, 6),
            "passes_coverage": coverage >= coverage_cfg["minimum_coverage_percent"],
        })
    date_coverage = pd.DataFrame(date_rows)
    date_coverage.to_csv(burst_out / "burst_union_coverage_by_date.csv", index=False)
    if not date_coverage["passes_coverage"].all():
        failed = date_coverage.loc[~date_coverage["passes_coverage"], "acquisition_date"].tolist()
        raise ValueError(f"Burst union fails AOI coverage on dates: {failed}")

    by_burst_and_date: dict[str, dict[str, Any]] = defaultdict(dict)
    for product in products:
        full_id = product.properties["burst"]["fullBurstID"]
        if full_id in stable_full_ids:
            by_burst_and_date[full_id][product.properties["startTime"][:10]] = product

    pair_rows = []
    job_plan = []
    for ref_date, sec_date in itertools.combinations(dates, 2):
        burst_pairs = []
        valid = True
        for full_id in stable_full_ids:
            pair = asf.Pair(
                by_burst_and_date[full_id][ref_date],
                by_burst_and_date[full_id][sec_date],
            )
            if pair.temporal_baseline is None or pair.perpendicular_baseline is None:
                valid = False
                break
            burst_pairs.append(pair)
        if not valid:
            continue
        temporal_days = max(abs(pair.temporal_baseline.days) for pair in burst_pairs)
        max_abs_bperp = max(abs(pair.perpendicular_baseline) for pair in burst_pairs)
        if temporal_days > sbas_cfg["maximum_temporal_baseline_days"]:
            continue
        if max_abs_bperp > sbas_cfg["maximum_absolute_perpendicular_baseline_m"]:
            continue

        references = [pair.ref.properties["sceneName"] for pair in burst_pairs]
        secondaries = [pair.sec.properties["sceneName"] for pair in burst_pairs]
        pair_id = f"{ref_date}_{sec_date}"
        pair_rows.append({
            "pair_id": pair_id,
            "reference_date": ref_date,
            "secondary_date": sec_date,
            "temporal_baseline_days": temporal_days,
            "maximum_absolute_perpendicular_baseline_m": max_abs_bperp,
            "burst_count": len(burst_pairs),
        })
        job_plan.append({
            "pair_id": pair_id,
            "reference": references,
            "secondary": secondaries,
            "job_type": "INSAR_ISCE_MULTI_BURST",
            "submit": False,
        })

    pair_df = pd.DataFrame(pair_rows).sort_values(["reference_date", "secondary_date"])
    pair_df.to_csv(network_out / "sbas_candidate_pairs.csv", index=False)
    (network_out / "hyp3_multiburst_job_plan.json").write_text(
        json.dumps(job_plan, indent=2), encoding="utf-8"
    )

    edges = list(zip(pair_df["reference_date"], pair_df["secondary_date"]))
    retained_dates = sorted(set(pair_df["reference_date"]) | set(pair_df["secondary_date"]))
    excluded_dates = sorted(set(dates) - set(retained_dates))
    connected = is_connected(retained_dates, edges)
    degree = {date: 0 for date in retained_dates}
    for ref_date, sec_date in edges:
        degree[ref_date] += 1
        degree[sec_date] += 1
    pd.DataFrame([
        {
            "acquisition_date": date,
            "reason": "No valid pair within configured temporal and perpendicular baseline thresholds",
        }
        for date in excluded_dates
    ]).to_csv(network_out / "excluded_acquisitions.csv", index=False)
    summary_lines = [
        "# Multi-burst SBAS network review",
        "",
        f"- Approved track: {track['flight_direction']} relative orbit {track['relative_orbit']}",
        f"- Acquisition dates searched: {len(dates)}",
        f"- Acquisition dates retained in network: {len(retained_dates)}",
        f"- Acquisition dates excluded: {len(excluded_dates)}",
        f"- Excluded dates: {', '.join(excluded_dates) if excluded_dates else 'None'}",
        f"- Stable bursts per acquisition: {len(stable_full_ids)}",
        f"- Full burst IDs: {', '.join(stable_full_ids)}",
        f"- Relative burst IDs: {', '.join(str(x) for x in relative_ids)}",
        f"- Subswath: {', '.join(subswaths)}",
        f"- Minimum burst-union AOI coverage: {date_coverage['aoi_coverage_percent'].min():.6f}%",
        f"- Candidate interferograms: {len(pair_df)}",
        f"- Temporal threshold: {sbas_cfg['maximum_temporal_baseline_days']} days",
        f"- Perpendicular threshold: {sbas_cfg['maximum_absolute_perpendicular_baseline_m']} m",
        f"- Retained network connected: {connected}",
        f"- Minimum date degree: {min(degree.values())}",
        f"- Maximum date degree: {max(degree.values())}",
        "",
        "## Manual approval required before HyP3 submission",
        "",
        "1. Review burst-union coverage and the pair network.",
        "2. Decide whether all candidate pairs are needed for the POC.",
        "3. Confirm HyP3 service/credits, looks and water-mask option.",
        "4. Do not submit the generated job plan without explicit approval.",
    ]
    (network_out / "network_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    if not connected:
        raise RuntimeError("Retained candidate SBAS network is disconnected")

    print("\n".join(summary_lines[:16]))
    print(f"\nSaved burst outputs to: {burst_out}")
    print(f"Saved unsubmitted SBAS plan to: {network_out}")


if __name__ == "__main__":
    main()
