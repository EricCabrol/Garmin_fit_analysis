#!/usr/bin/env python3
"""
Garmin FIT file processor with Kalman filter for speed and heading.
Generates an HTML report with:
  - Speed plot (raw vs filtered)
  - Path plot on OpenStreetMap (raw vs filtered), using Leaflet.js
"""

import sys
import os
import struct
import math
import json
import numpy as np
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# 1.  FIT FILE PARSER  (pure stdlib)
# ─────────────────────────────────────────────

GARMIN_EPOCH = datetime(1989, 12, 31, tzinfo=timezone.utc)

def read_fit(path: str):
    """
    Minimal FIT parser. Returns a list of dicts with keys:
      timestamp, lat, lon, speed, heading (may be None)
    All values in SI units (lat/lon in degrees, speed m/s, heading degrees).
    """
    with open(path, "rb") as f:
        raw = f.read()

    # ── FIT file header ──────────────────────────────────────────
    hdr_size = raw[0]
    data_size = struct.unpack_from("<I", raw, 4)[0]
    # Skip global header
    pos = hdr_size
    end = hdr_size + data_size

    local_mesg_defs = {}   # local_num -> list of (field_def_num, size, base_type)
    records = []

    # FIT base type map: base_type_byte -> (fmt_char, size)
    BASE_TYPES = {
        0x00: ("B",  1),   # enum
        0x01: ("b",  1),   # sint8
        0x02: ("B",  1),   # uint8
        0x83: ("h",  2),   # sint16
        0x84: ("H",  2),   # uint16
        0x85: ("i",  4),   # sint32
        0x86: ("I",  4),   # uint32
        0x07: ("s",  1),   # string (byte)
        0x88: ("f",  4),   # float32
        0x89: ("d",  8),   # float64
        0x0A: ("B",  1),   # uint8z
        0x8B: ("H",  2),   # uint16z
        0x8C: ("I",  4),   # uint32z
        0x8E: ("q",  8),   # sint64
        0x8F: ("Q",  8),   # uint64
        0x90: ("Q",  8),   # uint64z
    }

    while pos < end:
        record_hdr = raw[pos]
        pos += 1

        compressed = (record_hdr & 0x80) != 0

        if compressed:
            # Compressed timestamp – treat as data record
            local_num = (record_hdr >> 5) & 0x03
            if local_num not in local_mesg_defs:
                continue
            fields, total_size = local_mesg_defs[local_num]
            pos += total_size
            continue

        mesg_type = (record_hdr >> 6) & 0x03

        if mesg_type == 0b01 or mesg_type == 0b11:
            # Definition message
            has_dev = (record_hdr & 0x20) != 0
            local_num = record_hdr & 0x0F
            pos += 1  # reserved
            arch = raw[pos]; pos += 1
            endian = ">" if arch == 1 else "<"
            global_num = struct.unpack_from(endian + "H", raw, pos)[0]; pos += 2
            n_fields = raw[pos]; pos += 1
            fields = []
            total_size = 0
            for _ in range(n_fields):
                fdef, fsize, fbase = raw[pos], raw[pos+1], raw[pos+2]
                pos += 3
                fields.append((fdef, fsize, fbase, endian))
                total_size += fsize
            if has_dev:
                n_dev = raw[pos]; pos += 1
                for _ in range(n_dev):
                    pos += 3
            local_mesg_defs[local_num] = (global_num, fields, total_size)

        else:
            # Data message
            local_num = record_hdr & 0x0F
            if local_num not in local_mesg_defs:
                # Unknown – skip (we don't know size); bail out gracefully
                break
            global_num, fields, total_size = local_mesg_defs[local_num]

            field_data = {}
            field_start = pos
            for (fdef, fsize, fbase, endian) in fields:
                chunk = raw[pos:pos+fsize]
                pos += fsize
                bt = BASE_TYPES.get(fbase)
                if bt is None:
                    continue
                fmt_char, unit_size = bt
                if fmt_char == "s":
                    value = chunk.decode("latin-1", errors="replace").rstrip("\x00")
                elif unit_size == fsize:
                    value = struct.unpack(endian + fmt_char, chunk)[0]
                else:
                    # array – skip
                    value = None
                field_data[fdef] = value

            # global_num 20 = record message (GPS data)
            if global_num == 20:
                rec = {}
                # field 253 = timestamp (s since garmin epoch)
                ts = field_data.get(253)
                rec["timestamp"] = ts
                # field 0 = lat (semicircles), field 1 = lon (semicircles)
                lat = field_data.get(0)
                lon = field_data.get(1)
                SEMI_TO_DEG = 180.0 / 2**31
                rec["lat"] = lat * SEMI_TO_DEG if (lat is not None and lat != 0x7FFFFFFF) else None
                rec["lon"] = lon * SEMI_TO_DEG if (lon is not None and lon != 0x7FFFFFFF) else None
                # field 6 = speed (mm/s -> m/s)
                spd = field_data.get(6)
                rec["speed"] = spd / 1000.0 if (spd is not None and spd != 0xFFFF) else None
                # field 7 = heading/course (100 * degrees -> degrees)
                hdg = field_data.get(7)
                rec["heading"] = hdg / 100.0 if (hdg is not None and hdg != 0xFFFF) else None
                # field 3 = heart_rate (bpm, uint8)
                hr = field_data.get(3)
                rec["heart_rate"] = int(hr) if (hr is not None and hr != 0xFF) else None
                records.append(rec)

    return records


# ─────────────────────────────────────────────
# 2.  DEMO DATA GENERATOR (no FIT file)
# ─────────────────────────────────────────────

def generate_demo_data(n=600, seed=42):
    """Simulate a ~10-min run with GPS noise."""
    rng = np.random.default_rng(seed)
    # Centre: Paris, Bois de Boulogne
    lat0, lon0 = 48.8650, 2.2490
    dt = 1.0  # seconds

    # True trajectory: roughly circular loop
    t = np.arange(n) * dt
    true_speed = 3.0 + 0.5 * np.sin(2 * np.pi * t / 300)  # m/s ~3 m/s
    true_heading = np.mod(np.cumsum(np.full(n, 360.0 / (n * dt))) * dt, 360)

    # Convert speed+heading to lat/lon displacement
    lats, lons = [lat0], [lon0]
    for i in range(1, n):
        spd = true_speed[i]
        hdg = math.radians(true_heading[i])
        dy = spd * dt * math.cos(hdg)
        dx = spd * dt * math.sin(hdg)
        dlat = dy / 111320.0
        dlon = dx / (111320.0 * math.cos(math.radians(lats[-1])))
        lats.append(lats[-1] + dlat)
        lons.append(lons[-1] + dlon)

    # Add noise
    noise_spd = rng.normal(0, 0.4, n)
    noise_hdg = rng.normal(0, 8.0, n)
    noise_lat = rng.normal(0, 4.0, n) / 111320.0
    noise_lon = rng.normal(0, 4.0, n) / (111320.0 * math.cos(math.radians(lat0)))

    noise_hr  = rng.normal(0, 2.0, n)
    noise_cad = rng.normal(0, 2.0, n)
    true_hr   = 145 + 5 * np.sin(2 * np.pi * t / 400)
    true_cad  = 170 + 3 * np.sin(2 * np.pi * t / 200)

    records = []
    for i in range(n):
        records.append({
            "timestamp":  i,
            "lat":        lats[i] + noise_lat[i],
            "lon":        lons[i] + noise_lon[i],
            "speed":      max(0.0, true_speed[i] + noise_spd[i]),
            "heading":    np.mod(true_heading[i] + noise_hdg[i], 360.0),
            "heart_rate": int(np.clip(true_hr[i]  + noise_hr[i],  60, 220)),
            "cadence":    int(np.clip(true_cad[i] + noise_cad[i], 60, 220)),
        })
    return records


# ─────────────────────────────────────────────
# 3.  KALMAN FILTER  (constant speed + heading)
# ─────────────────────────────────────────────
#
# State vector x = [lat, lon, speed, heading]  (4 × 1)
# Motion model (constant speed + heading):
#   lat_{k+1}  = lat_k  + speed_k * dt * cos(hdg_k) / R_lat
#   lon_{k+1}  = lon_k  + speed_k * dt * sin(hdg_k) / R_lon(lat)
#   speed_{k+1} = speed_k
#   heading_{k+1} = heading_k
#
# Because the motion model is non-linear we use an Extended Kalman Filter (EKF).
# Observation z = [lat, lon, speed, heading]  (4 × 1)  (direct measurement).

def wrap_heading(h):
    """Wrap heading to [0, 360)."""
    return h % 360.0

def heading_diff(a, b):
    """Signed angular difference a-b in (−180, 180]."""
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d

R_EARTH = 6_371_000.0  # metres

def _derive_heading(records):
    """
    Compute heading from successive GPS positions for records where it is None.
    Uses a centred difference (smoothed) where possible.
    """
    lats = [r["lat"] for r in records]
    lons = [r["lon"] for r in records]
    n = len(records)
    headings = [None] * n

    for i in range(n):
        # Use centred difference when possible (smoother than forward)
        i0 = max(0, i - 1)
        i1 = min(n - 1, i + 1)
        if i0 == i1:
            continue
        dlat = math.radians(lats[i1] - lats[i0])
        dlon = math.radians(lons[i1] - lons[i0])
        cos_lat = math.cos(math.radians(lats[i]))
        hdg = math.degrees(math.atan2(dlon * cos_lat, dlat)) % 360.0
        headings[i] = hdg

    return headings


def run_kalman(records, dt=1.0):
    """
    Extended Kalman Filter on speed & heading.
    Returns list of dicts with filtered lat, lon, speed, heading.

    ── Covariance matrices ──────────────────────────────────────────────────────
    Two matrices govern the filter's behaviour:

    Q  — Process noise covariance  (4×4 diagonal)
         Models how much the TRUE state can change between two steps.
         Larger Q  → filter trusts the model less, follows measurements more.
         Units are in radians² (position) and (m/s)² / rad² (speed/heading).

         Q[0,0] = (σ_lat)²   σ_lat  = allowed position drift per step in rad
                              set to ~0.5 m / R_EARTH  (running is smooth)
         Q[1,1] = (σ_lon)²   same
         Q[2,2] = (σ_spd)²   σ_spd  = allowed speed change per step (m/s)
                              ~0.05 m/s  (running pace changes slowly)
         Q[3,3] = (σ_hdg)²   σ_hdg  = allowed heading change per step (rad)
                              ~3°/s is realistic for running turns

    R  — Observation noise covariance  (4×4 diagonal)
         Models the uncertainty of the RAW MEASUREMENTS.
         Larger R  → filter trusts measurements less, relies more on the model.

         R[0,0] = (σ_pos / R_EARTH)²   σ_pos  = GPS position noise ≈ 5–8 m
         R[1,1] = same
         R[2,2] = (σ_spd)²             Garmin speed sensor noise ≈ 0.1 m/s
         R[3,3] = (σ_hdg_rad)²         Heading derived from GPS ≈ 10–15°

    Key ratio:  Q / R controls smoothing strength.
    If Q << R  → heavy smoothing (filter trusts model, ignores noisy GPS).
    If Q >> R  → light smoothing (filter follows measurements closely).
    ────────────────────────────────────────────────────────────────────────────
    """
    # ── Fill missing headings from GPS positions ───────────────
    headings = _derive_heading(records)
    for i, r in enumerate(records):
        if r.get("heading") is None:
            r = dict(r)
            r["heading"] = headings[i]
            records[i] = r

    valid = [r for r in records if all(r.get(k) is not None
             for k in ("lat", "lon", "speed", "heading"))]
    if len(valid) < 5:
        raise ValueError("Not enough valid GPS records with speed+heading data.")
    has_hr = any(r.get("heart_rate") is not None for r in valid)

    # ── Q — Process noise covariance  (6×6 diagonal) ─────────
    # State: [lat, lon, speed, heading, heart_rate, cadence]
    # Models allowed TRUE state change per 1-second step for a runner.
    #
    # Q[0,0]=Q[1,1]: position    — 0.21 m/step
    # Q[2,2]:        speed       — 0.02 m/s per step
    # Q[3,3]:        heading     — 2° per step
    # Q[4,4]:        heart_rate  — 1 bpm per step  (HR changes slowly)
    # Q[5,5]:        cadence     — 1 rpm per step  (cadence changes slowly)
    σ_pos_proc = 0.21 / R_EARTH
    σ_spd_proc = 0.020
    σ_hdg_proc = math.radians(2.0)
    σ_hr_proc  = 1.0    # 1 bpm / step

    Q = np.diag([σ_pos_proc**2, σ_pos_proc**2, σ_spd_proc**2,
                 σ_hdg_proc**2, σ_hr_proc**2])

    # ── R — Observation noise covariance  (5×5 diagonal) ─────
    # Calibrated from the actual FIT file:
    #   GPS position ≈ 0.5 m,  speed ≈ 0.064 m/s,  heading ≈ 10°
    #   Heart rate optical sensor ≈ 2 bpm (1-σ)
    σ_pos_obs = 0.5 / R_EARTH
    σ_spd_obs = 0.064
    σ_hdg_obs = math.radians(10.0)
    σ_hr_obs  = 2.0    # 2 bpm optical sensor noise

    R_obs = np.diag([σ_pos_obs**2, σ_pos_obs**2, σ_spd_obs**2,
                     σ_hdg_obs**2, σ_hr_obs**2])

    # ── Initialise ──────────────────────────────────────────
    r0 = valid[0]
    x = np.array([
        math.radians(r0["lat"]),
        math.radians(r0["lon"]),
        r0["speed"],
        math.radians(r0["heading"]),
        float(r0.get("heart_rate") or 140),
    ])
    P = np.diag([
        (10.0/R_EARTH)**2,
        (10.0/R_EARTH)**2,
        1.0**2,
        np.radians(20.0)**2,
        5.0**2,
    ])

    results = []

    for r in valid:
        lat_r = math.radians(r["lat"])
        lon_r = math.radians(r["lon"])
        spd_r = r["speed"]
        hdg_r = math.radians(r["heading"])
        hr_r = float(r.get("heart_rate") or x[4])

        # ── PREDICT ─────────────────────────────────────────
        lat, lon, spd, hdg, hr = x
        cos_lat = math.cos(lat)

        # Non-linear state transition (HR: constant model)
        new_lat = lat + spd * dt * math.cos(hdg) / R_EARTH
        new_lon = lon + spd * dt * math.sin(hdg) / (R_EARTH * cos_lat)
        new_spd = spd
        new_hdg = hdg
        new_hr  = hr

        x_pred = np.array([new_lat, new_lon, new_spd, new_hdg, new_hr])

        # Jacobian of f w.r.t. x (5×5; HR row/col is identity)
        F = np.eye(5)
        F[0, 2] = dt * math.cos(hdg) / R_EARTH
        F[0, 3] = -spd * dt * math.sin(hdg) / R_EARTH
        F[1, 0] = spd * dt * math.sin(hdg) * math.sin(lat) / (R_EARTH * cos_lat**2)
        F[1, 2] = dt * math.sin(hdg) / (R_EARTH * cos_lat)
        F[1, 3] = spd * dt * math.cos(hdg) / (R_EARTH * cos_lat)

        P_pred = F @ P @ F.T + Q

        # ── UPDATE ──────────────────────────────────────────
        H = np.eye(5)
        z = np.array([lat_r, lon_r, spd_r, hdg_r, hr_r])

        # Innovation (handle heading wrap-around)
        y = z - x_pred
        y[3] = math.radians(heading_diff(math.degrees(z[3]),
                                         math.degrees(x_pred[3])))

        S = H @ P_pred @ H.T + R_obs
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        x[3] = math.radians(wrap_heading(math.degrees(x[3])))
        x[4] = float(np.clip(x[4], 30, 250))   # guard HR
        P = (np.eye(5) - K @ H) @ P_pred

        results.append({
            "timestamp":  r["timestamp"],
            "lat":        math.degrees(x[0]),
            "lon":        math.degrees(x[1]),
            "speed":      float(x[2]),
            "heading":    math.degrees(x[3]),
            "heart_rate": float(x[4]),
            # raw
            "raw_lat":        r["lat"],
            "raw_lon":        r["lon"],
            "raw_speed":      r["speed"],
            "raw_heading":    r["heading"],
            "raw_heart_rate": r.get("heart_rate"),
        })

    return results


# ─────────────────────────────────────────────
# 4.  HTML REPORT GENERATOR
# ─────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Garmin Run – Kalman Filter Report</title>

<!-- Leaflet (OSM map) -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>

<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
  }
  header {
    background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 100%);
    padding: 28px 40px;
    border-bottom: 1px solid #2d3748;
    display: flex;
    align-items: center;
    gap: 18px;
  }
  header .icon { font-size: 2.2rem; }
  header h1 { font-size: 1.6rem; font-weight: 700; color: #63b3ed; }
  header p  { font-size: 0.88rem; color: #a0aec0; margin-top: 3px; }

  .stats-bar {
    display: flex;
    gap: 16px;
    padding: 20px 40px;
    background: #141820;
    border-bottom: 1px solid #2d3748;
    flex-wrap: wrap;
  }
  .stat {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 12px 20px;
    min-width: 130px;
    flex: 1;
  }
  .stat-label { font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: .06em; }
  .stat-value { font-size: 1.5rem; font-weight: 700; color: #63b3ed; margin-top: 4px; }
  .stat-unit  { font-size: 0.78rem; color: #718096; }

  .plots {
    display: flex;
    flex-direction: column;
    gap: 28px;
    padding: 28px 40px;
    max-width: 1400px;
    margin: 0 auto;
  }
  .card {
    background: #1a2035;
    border: 1px solid #2d3748;
    border-radius: 14px;
    overflow: hidden;
  }
  .card-header {
    padding: 16px 24px;
    background: #141c2e;
    border-bottom: 1px solid #2d3748;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .card-header h2 { font-size: 1.05rem; font-weight: 600; color: #90cdf4; }
  .card-header .badge {
    margin-left: auto;
    font-size: 0.72rem;
    background: #2b4a7a;
    color: #90cdf4;
    padding: 3px 10px;
    border-radius: 20px;
  }
  .chart-wrap {
    padding: 20px 24px;
    height: 320px;
    position: relative;
  }
  #map {
    height: 520px;
    width: 100%;
    border-radius: 0 0 14px 14px;
  }
  .legend {
    padding: 14px 24px;
    background: #141c2e;
    border-top: 1px solid #2d3748;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #a0aec0; }
  .legend-line { width: 28px; height: 3px; border-radius: 2px; }
  footer {
    text-align: center;
    padding: 20px;
    color: #4a5568;
    font-size: 0.78rem;
    border-top: 1px solid #1a2035;
  }
</style>
</head>
<body>

<header>
  <div class="icon">🏃</div>
  <div>
    <h1>Garmin Running Activity — Kalman Filter Analysis</h1>
    <p>Model: constant speed &amp; heading &nbsp;|&nbsp; Extended Kalman Filter (EKF) &nbsp;|&nbsp; {{source_label}}</p>
  </div>
</header>

<div class="stats-bar">
  <div class="stat">
    <div class="stat-label">Data points</div>
    <div class="stat-value">{{n_points}}</div>
    <div class="stat-unit">records</div>
  </div>
  <div class="stat">
    <div class="stat-label">Avg speed</div>
    <div class="stat-value">{{avg_raw_spd}}</div>
    <div class="stat-unit">m/s</div>
  </div>
  <div class="stat">
    <div class="stat-label">Speed std (raw)</div>
    <div class="stat-value">{{std_raw}}</div>
    <div class="stat-unit">m/s</div>
  </div>
  <div class="stat">
    <div class="stat-label">Speed std (filtered)</div>
    <div class="stat-value">{{std_filt}}</div>
    <div class="stat-unit">m/s</div>
  </div>
  <div class="stat">
    <div class="stat-label">Path roughness (raw)</div>
    <div class="stat-value">{{roughness_raw}}</div>
    <div class="stat-unit">m (2nd deriv.)</div>
  </div>
  <div class="stat">
    <div class="stat-label">Path noise reduction</div>
    <div class="stat-value">{{noise_reduction}}</div>
    <div class="stat-unit">%</div>
  </div>
</div>

<div class="plots">

  <!-- SPEED CHART -->
  <div class="card">
    <div class="card-header">
      <span>⚡</span>
      <h2>Pace — Raw vs Kalman Filtered</h2>
      <div class="badge">EKF constant-speed model</div>
    </div>
    <div class="chart-wrap">
      <canvas id="speedChart"></canvas>
    </div>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-line" style="background:#f6ad55;opacity:0.55;"></div>Raw pace
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#63b3ed;"></div>Filtered pace
      </div>
    </div>
  </div>

  <!-- HR CHART -->
  <div class="card">
    <div class="card-header">
      <span>❤️</span>
      <h2>Heart Rate — Raw vs Kalman Filtered</h2>
      <div class="badge">EKF constant model</div>
    </div>
    <div class="chart-wrap" style="height:300px">
      <canvas id="hrChart"></canvas>
    </div>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-line" style="background:#fc8181;opacity:0.55;"></div>Raw HR
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#fc8181;"></div>Filtered HR
      </div>
    </div>
  </div>

  <!-- MAP -->
  <div class="card">
    <div class="card-header">
      <span>🗺️</span>
      <h2>GPS Path — Raw vs Kalman Filtered (OpenStreetMap)</h2>
      <div class="badge">Leaflet + OSM tiles</div>
    </div>
    <div id="map"></div>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-line" style="background:#f6ad55;opacity:0.7;"></div>Raw GPS path
      </div>
      <div class="legend-item">
        <div class="legend-line" style="background:#4299e1;"></div>Kalman filtered path
      </div>
    </div>
  </div>

</div>

<footer>Generated by garmin_kalman.py &nbsp;•&nbsp; Extended Kalman Filter — constant speed &amp; heading model</footer>

<script>
// ── Injected data ────────────────────────────────────────────
const DATA = {{json_data}};

// ── Speed Chart ──────────────────────────────────────────────
(function() {
  const labels = DATA.map((_, i) => i);
  const ctx = document.getElementById('speedChart').getContext('2d');
  // Convert m/s → min/km:  pace = 1000 / (speed * 60)
  const toPace = s => (s > 0.1) ? 1000 / (s * 60) : null;
  const fmtPace = v => {
    if (v === null || v === undefined) return '—';
    const mins = Math.floor(v);
    const secs = Math.round((v - mins) * 60);
    return mins + ':' + String(secs).padStart(2, '0');
  };

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Raw pace (min/km)',
          data: DATA.map(d => toPace(d.raw_speed)),
          borderColor: 'rgba(246,173,85,0.55)',
          backgroundColor: 'transparent',
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: 'Filtered pace (min/km)',
          data: DATA.map(d => toPace(d.speed)),
          borderColor: '#63b3ed',
          backgroundColor: 'rgba(99,179,237,0.07)',
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.3,
          fill: true,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#a0aec0', font: { size: 12 } } },
        tooltip: {
          backgroundColor: '#1a2035', titleColor: '#e2e8f0', bodyColor: '#a0aec0',
          callbacks: {
            label: ctx => ctx.dataset.label + ': ' + fmtPace(ctx.parsed.y) + ' min/km',
            title: items => 'T = ' + items[0].label + ' s',
          }
        },
      },
      scales: {
        x: {
          ticks: { color: '#718096', maxTicksLimit: 12,
                   callback: (v) => v + 's' },
          grid: { color: '#2d3748' },
          title: { display: true, text: 'Time (s)', color: '#718096' }
        },
        y: {
          reverse: true,
          min: 4.0,
          max: 7.0,
          ticks: {
            color: '#718096',
            callback: v => fmtPace(v),
          },
          grid: { color: '#2d3748' },
          title: { display: true, text: 'Pace (min/km)', color: '#718096' }
        }
      }
    }
  });
})();

// ── Map ───────────────────────────────────────────────────────
(function() {
  const rawPath  = DATA.map(d => [d.raw_lat,  d.raw_lon]);
  const filtPath = DATA.map(d => [d.lat,       d.lon]);

  // Centre on filtered path midpoint
  const mid = filtPath[Math.floor(filtPath.length / 2)];
  const map = L.map('map').setView(mid, 16);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);

  L.polyline(rawPath, {
    color: 'rgba(246,173,85,0.7)',
    weight: 2,
    opacity: 0.8,
  }).addTo(map).bindPopup('Raw GPS path');

  L.polyline(filtPath, {
    color: '#4299e1',
    weight: 3.5,
    opacity: 0.95,
  }).addTo(map).bindPopup('Kalman filtered path');

  // Start / end markers
  const iconStart = L.circleMarker(filtPath[0], {
    radius: 7, color: '#68d391', fillColor: '#68d391', fillOpacity: 1, weight: 2
  }).addTo(map).bindTooltip('Start');
  const iconEnd = L.circleMarker(filtPath[filtPath.length-1], {
    radius: 7, color: '#fc8181', fillColor: '#fc8181', fillOpacity: 1, weight: 2
  }).addTo(map).bindTooltip('End');

  // Fit bounds
  const allPoints = rawPath.concat(filtPath);
  map.fitBounds(L.latLngBounds(allPoints).pad(0.05));
})();
// ── HR Chart ─────────────────────────────────────────────────
(function() {
  const labels = DATA.map((_, i) => i);
  const hasHR = DATA.some(d => d.raw_heart_rate !== null && d.raw_heart_rate !== undefined);
  if (!hasHR) return;
  const ctx = document.getElementById('hrChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Raw HR (bpm)',
          data: DATA.map(d => d.raw_heart_rate),
          borderColor: 'rgba(252,129,129,0.45)',
          backgroundColor: 'transparent',
          borderWidth: 1.2, pointRadius: 0, tension: 0.1,
        },
        {
          label: 'Filtered HR (bpm)',
          data: DATA.map(d => d.heart_rate),
          borderColor: '#fc8181',
          backgroundColor: 'rgba(252,129,129,0.07)',
          borderWidth: 2.2, pointRadius: 0, tension: 0.4,
          fill: true,
        },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#a0aec0', font: { size: 12 } } },
        tooltip: {
          backgroundColor: '#1a2035', titleColor: '#e2e8f0', bodyColor: '#a0aec0',
          callbacks: {
            title: items => 'T = ' + items[0].label + ' s',
            label: ctx => ctx.dataset.label + ': ' + Math.round(ctx.parsed.y) + ' bpm',
          }
        },
      },
      scales: {
        x: {
          ticks: { color: '#718096', maxTicksLimit: 12, callback: v => v + 's' },
          grid: { color: '#2d3748' },
          title: { display: true, text: 'Time (s)', color: '#718096' }
        },
        y: {
          ticks: { color: '#fc8181' },
          grid: { color: '#2d3748' },
          title: { display: true, text: 'Heart rate (bpm)', color: '#fc8181' },
        },
      }
    }
  });
})();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────

def main():
    fit_path = sys.argv[1] if len(sys.argv) > 1 else None

    # ── Load data ──────────────────────────────────────────────
    if not fit_path:
        print("Error: no FIT file provided.")
        print("Usage: python garmin_kalman.py <activity.fit>")
        sys.exit(1)

    print(f"[+] Parsing FIT file: {fit_path}")
    records = read_fit(fit_path)
    records = [r for r in records if r.get("lat") and r.get("lon")]
    print(f"    → {len(records)} GPS records found")
    source_label = os.path.basename(fit_path)

    # Fill missing speed from GPS positions (heading derived inside run_kalman)
    for i in range(1, len(records)):
        a, b = records[i-1], records[i]
        if b["speed"] is None:
            dlat = math.radians(b["lat"] - a["lat"])
            dlon = math.radians(b["lon"] - a["lon"])
            dist = R_EARTH * math.sqrt(dlat**2 + (dlon * math.cos(math.radians(a["lat"])))**2)
            b["speed"] = dist  # dt = 1 s

    # ── Kalman filter ──────────────────────────────────────────
    print("[+] Running Extended Kalman Filter …")
    results = run_kalman(records)
    print(f"    → Filtered {len(results)} records")

    # ── Statistics ─────────────────────────────────────────────
    raw_spd  = np.array([r["raw_speed"] for r in results])
    filt_spd = np.array([r["speed"]     for r in results])
    avg_raw  = float(np.mean(raw_spd))
    std_raw  = float(np.std(raw_spd))
    std_filt = float(np.std(filt_spd))

    # Path roughness: RMS of positional 2nd derivative (acceleration proxy)
    raw_lats  = np.array([r["raw_lat"] for r in results])
    raw_lons  = np.array([r["raw_lon"] for r in results])
    filt_lats = np.array([r["lat"]     for r in results])
    filt_lons = np.array([r["lon"]     for r in results])
    cos_lat_s = math.cos(math.radians(float(np.mean(raw_lats))))

    def path_roughness(lats, lons):
        dlat = np.diff(lats) * R_EARTH * math.pi / 180
        dlon = np.diff(lons) * R_EARTH * math.pi / 180 * cos_lat_s
        return float(np.sqrt(np.mean(np.diff(dlat)**2 + np.diff(dlon)**2)))

    rough_raw  = path_roughness(raw_lats,  raw_lons)
    rough_filt = path_roughness(filt_lats, filt_lons)
    noise_red  = 100.0 * (1.0 - rough_filt / rough_raw) if rough_raw > 0 else 0.0

    print(f"    Avg speed  raw={avg_raw:.2f} m/s")
    print(f"    Speed std  raw={std_raw:.4f}  filtered={std_filt:.4f}")
    print(f"    Path roughness  raw={rough_raw:.4f} m  filtered={rough_filt:.4f} m")
    print(f"    Path noise reduction: {noise_red:.1f} %")

    # ── Build HTML ─────────────────────────────────────────────
    json_data = json.dumps(results, indent=None)

    html = HTML_TEMPLATE \
        .replace("{{source_label}}",   source_label) \
        .replace("{{n_points}}",        str(len(results))) \
        .replace("{{avg_raw_spd}}",     f"{avg_raw:.2f}") \
        .replace("{{std_raw}}",         f"{std_raw:.4f}") \
        .replace("{{std_filt}}",        f"{std_filt:.4f}") \
        .replace("{{roughness_raw}}",   f"{rough_raw:.3f}") \
        .replace("{{noise_reduction}}", f"{noise_red:.1f}") \
        .replace("{{json_data}}",       json_data)

    out_path = "./output/garmin_kalman_report.html"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[✓] Report written to {out_path}")
    return out_path


if __name__ == "__main__":
    main()