#!/usr/bin/env python3
"""Search ASF for Sentinel-1 IW SLC scenes over the Dammam POC AOI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import asf_search as asf
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_aoi_wkt(path: Path) -> tuple[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    features = data.get("features", [])
    if len(features) != 1:
        raise ValueError("AOI file must contain exactly one feature")
    geom = shape(features[0]["geometry"])
    if not geom.is_valid:
        raise ValueError("AOI geometry is invalid")
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("AOI must be a polygon or multipolygon")
    return geom.wkt, data


def product_record(product: Any) -> dict[str, Any]:
    props = product.properties
    return {
        "scene_name": props.get("sceneName") or props.get("fileID"),
        "start_time": props.get("startTime"),
        "stop_time": props.get("stopTime"),
        "platform": props.get("platform"),
        "flight_direction": props.get("flightDirection"),
        "relative_orbit": props.get("pathNumber"),
        "frame_number": props.get("frameNumber"),
        "beam_mode": props.get("beamModeType"),
        "polarization": props.get("polarization"),
        "processing_level": props.get("processingLevel"),
        "group_id": props.get("groupID"),
        "url": props.get("url"),
    }


def coverage_percent(
    scene_geometry: dict[str, Any],
    projected_aoi: Any,
    transformer: Transformer,
) -> float:
    scene = shape(scene_geometry)
    if not scene.is_valid:
        scene = scene.buffer(0)
    projected_scene = transform(transformer.transform, scene)
    if projected_scene.is_empty:
        return 0.0
    intersection_area = projected_scene.intersection(projected_aoi).area
    return 100.0 * intersection_area / projected_aoi.area


def write_summary(
    df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    orbit_summary: pd.DataFrame,
    path: Path,
    config: dict[str, Any],
) -> None:
    coverage_cfg = config["coverage_filter"]
    lines = [
        "# ASF Sentinel-1 acquisition-search summary",
        "",
        f"- Search period: {config['search']['start']} to {config['search']['end']}",
        f"- Total unique SLC scenes: {len(df)}",
        f"- Scenes passing coverage filter: {len(filtered_df)}",
        f"- Coverage threshold: {coverage_cfg['minimum_coverage_percent']}%",
        f"- Coverage calculation CRS: {coverage_cfg['area_crs']} ({coverage_cfg['area_crs_name']})",
        "- AOI status: **PROVISIONAL - MANUAL REVIEW REQUIRED**",
        "",
        "## Candidate orbit groups",
        "",
    ]
    if orbit_summary.empty:
        lines.append("No matching scenes were returned.")
    else:
        columns = list(orbit_summary.columns)
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in orbit_summary.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.extend([
        "",
        "## Required manual review",
        "",
        "1. Inspect AOI and scene footprints in QGIS.",
        "2. Review scenes/dates flagged by the automated coverage filter.",
        "3. Select one flight direction and one relative orbit from passing scenes.",
        "4. Check temporal regularity and identify long acquisition gaps.",
        "5. Record the approved track before searching burst granules.",
        "",
        "No HyP3 jobs were submitted by this script.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_dir = config_path.parent
    config = load_yaml(config_path)
    aoi_path = (project_dir / config["aoi"]["path"]).resolve()
    aoi_wkt, _ = load_aoi_wkt(aoi_path)
    with aoi_path.open("r", encoding="utf-8") as stream:
        aoi_data = json.load(stream)
    aoi_geometry = shape(aoi_data["features"][0]["geometry"])

    coverage_cfg = config["coverage_filter"]
    transformer = Transformer.from_crs(
        coverage_cfg["source_crs"],
        coverage_cfg["area_crs"],
        always_xy=True,
    )
    projected_aoi = transform(transformer.transform, aoi_geometry)
    if projected_aoi.area <= 0:
        raise ValueError("Projected AOI has zero area")

    search_cfg = config["search"]
    all_products: dict[str, Any] = {}
    for direction in search_cfg["flight_directions"]:
        results = asf.geo_search(
            platform=[asf.PLATFORM.SENTINEL1],
            intersectsWith=aoi_wkt,
            start=search_cfg["start"],
            end=search_cfg["end"],
            beamMode=asf.BEAMMODE.IW,
            processingLevel=[asf.PRODUCT_TYPE.SLC],
            polarization=[asf.POLARIZATION.VV_VH],
            flightDirection=direction,
            maxResults=search_cfg["max_results_per_direction"],
        )
        for product in results:
            key = product.properties.get("sceneName") or product.properties.get("fileID")
            all_products[key] = product

    products = sorted(
        all_products.values(),
        key=lambda item: item.properties.get("startTime", ""),
    )
    if not products:
        raise RuntimeError("ASF returned no matching Sentinel-1 IW SLC scenes")

    out_dir = (project_dir / config["outputs"]["directory"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    product_features = []
    for product in products:
        feature = product.geojson()
        coverage = coverage_percent(feature["geometry"], projected_aoi, transformer)
        passed = coverage >= coverage_cfg["minimum_coverage_percent"]
        record = product_record(product)
        record["coverage_percent"] = round(coverage, 6)
        record["covers_aoi"] = passed
        records.append(record)
        feature["properties"]["coverage_percent"] = round(coverage, 6)
        feature["properties"]["covers_aoi"] = passed
        product_features.append(feature)

    df = pd.DataFrame(records)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["date"] = df["start_time"].dt.date.astype(str)
    df = df.sort_values(["flight_direction", "relative_orbit", "start_time"])
    df.to_csv(out_dir / "sentinel1_slc_scenes.csv", index=False)

    filtered_df = df[df["covers_aoi"]].copy()
    filtered_df.to_csv(out_dir / "sentinel1_slc_scenes_filtered.csv", index=False)
    rejected_df = df[~df["covers_aoi"]].copy()
    rejected_df.to_csv(out_dir / "sentinel1_slc_scenes_rejected.csv", index=False)

    feature_collection = {
        "type": "FeatureCollection",
        "features": product_features,
    }
    (out_dir / "sentinel1_slc_scenes.geojson").write_text(
        json.dumps(feature_collection, indent=2), encoding="utf-8"
    )
    filtered_names = set(filtered_df["scene_name"])
    filtered_feature_collection = {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in product_features
            if (feature["properties"].get("sceneName") or feature["properties"].get("fileID"))
            in filtered_names
        ],
    }
    (out_dir / "sentinel1_slc_scenes_filtered.geojson").write_text(
        json.dumps(filtered_feature_collection, indent=2), encoding="utf-8"
    )
    rejected_names = set(rejected_df["scene_name"])
    rejected_feature_collection = {
        "type": "FeatureCollection",
        "features": [
            feature
            for feature in product_features
            if (feature["properties"].get("sceneName") or feature["properties"].get("fileID"))
            in rejected_names
        ],
    }
    (out_dir / "sentinel1_slc_scenes_rejected.geojson").write_text(
        json.dumps(rejected_feature_collection, indent=2), encoding="utf-8"
    )

    grouped = df.groupby(["flight_direction", "relative_orbit"], dropna=False)
    orbit_summary = grouped.agg(
        scene_count=("scene_name", "count"),
        unique_dates=("date", "nunique"),
        scenes_passing=("covers_aoi", "sum"),
        minimum_coverage_all_scenes=("coverage_percent", "min"),
        first_acquisition=("start_time", "min"),
        last_acquisition=("start_time", "max"),
        frame_count=("frame_number", "nunique"),
    ).reset_index()
    passed_dates = (
        filtered_df.groupby(["flight_direction", "relative_orbit"])["date"]
        .nunique()
        .rename("unique_dates_passing")
        .reset_index()
    )
    orbit_summary = orbit_summary.merge(
        passed_dates,
        on=["flight_direction", "relative_orbit"],
        how="left",
    )
    orbit_summary["unique_dates_passing"] = (
        orbit_summary["unique_dates_passing"].fillna(0).astype(int)
    )
    passing_minimum = (
        filtered_df.groupby(["flight_direction", "relative_orbit"])["coverage_percent"]
        .min()
        .rename("minimum_coverage_passing_scenes")
        .reset_index()
    )
    orbit_summary = orbit_summary.merge(
        passing_minimum,
        on=["flight_direction", "relative_orbit"],
        how="left",
    )
    orbit_summary["first_acquisition"] = orbit_summary["first_acquisition"].dt.strftime("%Y-%m-%d")
    orbit_summary["last_acquisition"] = orbit_summary["last_acquisition"].dt.strftime("%Y-%m-%d")
    orbit_summary = orbit_summary.sort_values(
        ["unique_dates_passing", "scenes_passing", "minimum_coverage_passing_scenes"],
        ascending=False,
    )
    orbit_summary.to_csv(out_dir / "orbit_summary.csv", index=False)
    write_summary(
        df,
        filtered_df,
        orbit_summary,
        out_dir / "search_summary.md",
        config,
    )

    print(orbit_summary.to_string(index=False))
    print(f"\nSaved acquisition inventory to: {out_dir}")
    print("Manual approval is required before burst search or HyP3 submission.")


if __name__ == "__main__":
    main()
