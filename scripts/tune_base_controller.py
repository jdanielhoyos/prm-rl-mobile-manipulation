# tune_youbot_pathframe_bodyref_faster.py
#
# Faster-oriented tuning for youBot path-frame feedforward + feedback controller
# using /youBot_ref as the body frame and corrected wheel-mixer signs.
#
# Assumptions:
#   /youBot_ref axes:
#       red/x   = right
#       green/y = forward
#       blue/z  = up
#
# Corrected wheel mixer for wheel order [fl, rl, rr, fr]:
#   forward  -> [ -1, -1, -1, -1 ]
#   right    -> [ -1, +1, -1, +1 ]
#   yaw CCW  -> [ +1, +1, -1, -1 ]
#
# Multi-trajectory benchmark:
#   1) rounded L turn
#   2) mirrored rounded L turn
#   3) double lane change
#   4) slalom (+8 deg heading offset)
#   5) 270 deg open arc
#   6) hairpin (-8 deg heading offset)
#
# Goal of this version:
#   Keep success and decent tracking, but bias the search toward faster execution.

import math
import time
import csv
import itertools
from pathlib import Path
from datetime import datetime

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ============================================================
# Scene paths
# ============================================================

BASE_SHAPE_PATH = "/youBot"
BODY_REF_PATH   = "/youBot_ref"

J_FL = "/youBot/rollingJoint_fl"
J_RL = "/youBot/rollingJoint_rl"
J_RR = "/youBot/rollingJoint_rr"
J_FR = "/youBot/rollingJoint_fr"


# ============================================================
# Trial settings
# ============================================================

MAX_SIM_TIME = 55.0
END_TOL = 0.10

SEARCH_BACK = 10
SEARCH_FWD = 120

U_FWD_MAX = 20.0
U_RIGHT_MAX = 20.0
U_YAW_MAX = 4.0
WHEEL_MAX = 25.0


HEADING_GATE_1 = 0.55
HEADING_GATE_2 = 0.95
HEADING_GATE_3 = 1.35

YAW_SIGN = 1.0

DT_SLEEP = 0.0
RESET_SETTLE_STEPS = 3
WAIT_AFTER_STOP_SEC = 0.12

STALL_CHECK_TIME = 20.0
STALL_REQUIRED_PROGRESS_RATIO = 0.18

LARGE_DEVIATION_WARN = 0.30

# Additional early abort if controller is clearly failing
EARLY_ABORT_TIME = 12.0
EARLY_ABORT_PROGRESS_RATIO = 0.10

# Path sampling
PATH_DS = 0.02


# ============================================================
# Faster-oriented search grids
# ============================================================

# Stage 1:
# 4 * 3 * 3 * 3 * 3 = 324 candidates
PREVIEW_DIST_GRID = [0.08, 0.11, 0.14, 0.17]
V_FF_GRID         = [0.80, 1.20, 1.60]
K_ALONG_GRID      = [6.0, 8.0, 10.0]
K_CROSS_GRID      = [14.0, 18.0, 22.0]
K_YAW_GRID        = [1.2, 1.6, 2.0]

# Stage 2:
# TOP_K_STAGE1 * 3 * 4
SMOOTH_ALPHA_GRID    = [0.55, 0.70, 0.85]
MIN_TRANS_SCALE_GRID = [0.55, 0.70, 0.85, 1.00]

TOP_K_STAGE1 = 6


# ============================================================
# Multi-trajectory benchmark
# ============================================================

SCENARIOS = [
    {"name": "rounded_L_right",    "traj_kind": "ROUNDED_L",          "mirror": False, "yaw_offset_deg": 0.0},
    {"name": "rounded_L_left",     "traj_kind": "ROUNDED_L",          "mirror": True,  "yaw_offset_deg": 0.0},
    {"name": "double_lane_change", "traj_kind": "DOUBLE_LANE_CHANGE", "mirror": False, "yaw_offset_deg": 0.0},
    {"name": "slalom_plus8",       "traj_kind": "SLALOM",             "mirror": False, "yaw_offset_deg": 8.0},
    {"name": "open_arc_270",       "traj_kind": "CIRCLE_ARC_270",     "mirror": False, "yaw_offset_deg": 0.0},
    {"name": "hairpin_minus8",     "traj_kind": "HAIRPIN",            "mirror": False, "yaw_offset_deg": -8.0},
]


# ============================================================
# Output
# ============================================================

OUT_DIR = Path("pathframe_bodyref_tuning_results_faster") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Basic helpers
# ============================================================

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def deg2rad(d):
    return d * math.pi / 180.0

def get_pose2d(sim, h):
    p = sim.getObjectPosition(h, -1)
    o = sim.getObjectOrientation(h, -1)
    return p[0], p[1], o[2]

def stop_wheels(sim, wheel_handles):
    for h in wheel_handles:
        sim.setJointTargetVelocity(h, 0.0)

def set_wheels(sim, wheel_handles, w):
    for h, wi in zip(wheel_handles, w):
        sim.setJointTargetVelocity(h, float(wi))

def wait_after_stop():
    time.sleep(WAIT_AFTER_STOP_SEC)

def save_csv(rows, filepath):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def path_length(wps):
    total = 0.0
    for i in range(len(wps) - 1):
        total += math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
    return total


# ============================================================
# Body frame from /youBot_ref
# ============================================================

def get_body_axes_from_ref(sim, body_h):
    """
    Assumes /youBot_ref axes:
      red/x   = right
      green/y = forward
      blue/z  = up

    Returns 2D world-frame unit vectors:
      right_hat, fwd_hat
    """
    m = sim.getObjectMatrix(body_h, -1)

    x_axis = np.array([m[0], m[4], m[8]], dtype=float)  # right
    y_axis = np.array([m[1], m[5], m[9]], dtype=float)  # forward

    right_hat = x_axis[:2]
    fwd_hat   = y_axis[:2]

    right_hat = right_hat / np.linalg.norm(right_hat)
    fwd_hat   = fwd_hat / np.linalg.norm(fwd_hat)
    return right_hat, fwd_hat


# ============================================================
# Local geometry helpers
# local coordinates:
#   r = right
#   f = forward
# ============================================================

def sample_line(p0, p1, ds=0.02):
    x0, y0 = p0
    x1, y1 = p1
    L = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(math.ceil(L / ds)) + 1)
    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return pts

def sample_arc(center, radius, ang0, ang1, ds=0.02):
    arc_len = radius * abs(ang1 - ang0)
    n = max(8, int(math.ceil(arc_len / ds)) + 1)
    pts = []
    for i in range(n):
        a = ang0 + (ang1 - ang0) * i / (n - 1)
        pts.append((center[0] + radius * math.cos(a),
                    center[1] + radius * math.sin(a)))
    return pts

def sample_lateral_shift(r0, r1, f_start, length, ds=0.02):
    """
    Smooth lane-change section:
      r(s) = cosine interpolation from r0 to r1
      f(s) = f_start + s
    """
    n = max(10, int(math.ceil(length / ds)) + 1)
    pts = []
    for i in range(n):
        s = length * i / (n - 1)
        alpha = 0.5 * (1.0 - math.cos(math.pi * s / length))
        r = r0 + (r1 - r0) * alpha
        f = f_start + s
        pts.append((r, f))
    return pts

def sample_slalom(length, amp, cycles, ds=0.02):
    n = max(20, int(math.ceil(length / ds)) + 1)
    pts = []
    for i in range(n):
        s = length * i / (n - 1)
        r = amp * math.sin(2.0 * math.pi * cycles * s / length)
        f = s
        pts.append((r, f))
    return pts

def sample_circle_arc_270(radius=0.40, ds=0.02):
    """
    Start at (0,0), tangent along +f.
    """
    arc_len = 1.5 * math.pi * radius
    n = max(20, int(math.ceil(arc_len / ds)) + 1)
    pts = []
    for i in range(n):
        theta = (1.5 * math.pi) * i / (n - 1)
        r = radius - radius * math.cos(theta)
        f = radius * math.sin(theta)
        pts.append((r, f))
    return pts

def concatenate_segments(segments):
    pts = []
    for seg in segments:
        if not pts:
            pts.extend(seg)
        else:
            pts.extend(seg[1:])
    return pts


# ============================================================
# Trajectory builders in local coordinates
# ============================================================

def build_local_path_rounded_L(mirror=False):
    ds = PATH_DS

    seg1 = sample_line((0.0, 0.0), (0.0, 0.50), ds)

    R = 0.35
    if not mirror:
        center = (R, 0.50)
        seg2 = sample_arc(center, R, math.pi, math.pi / 2.0, ds)
        p_end = seg2[-1]
        seg3 = sample_line(p_end, (0.95, p_end[1]), ds)
    else:
        center = (-R, 0.50)
        seg2 = sample_arc(center, R, 0.0, math.pi / 2.0, ds)
        p_end = seg2[-1]
        seg3 = sample_line(p_end, (-0.95, p_end[1]), ds)

    return concatenate_segments([seg1, seg2, seg3])

def build_local_path_double_lane_change(mirror=False):
    sign = -1.0 if mirror else 1.0
    ds = PATH_DS
    segments = []

    segments.append(sample_line((0.0, 0.0), (0.0, 0.30), ds))
    segments.append(sample_lateral_shift(0.0, sign * 0.28, 0.30, 0.45, ds))
    segments.append(sample_lateral_shift(sign * 0.28, -sign * 0.28, 0.75, 0.70, ds))
    segments.append(sample_lateral_shift(-sign * 0.28, 0.0, 1.45, 0.45, ds))
    segments.append(sample_line((0.0, 1.90), (0.0, 2.25), ds))

    return concatenate_segments(segments)

def build_local_path_slalom(mirror=False):
    sign = -1.0 if mirror else 1.0
    return sample_slalom(length=2.10, amp=sign * 0.22, cycles=1.5, ds=PATH_DS)

def build_local_path_circle_arc_270(mirror=False):
    pts = sample_circle_arc_270(radius=0.40, ds=PATH_DS)
    if mirror:
        pts = [(-r, f) for (r, f) in pts]
    return pts

def build_local_path_hairpin(mirror=False):
    ds = PATH_DS
    segments = []

    segments.append(sample_line((0.0, 0.0), (0.0, 0.55), ds))

    R = 0.35
    if not mirror:
        center = (R, 0.55)
        seg2 = sample_arc(center, R, math.pi, 0.0, ds)
        p_end = seg2[-1]
        seg3 = sample_line(p_end, (p_end[0], -0.15), ds)
    else:
        center = (-R, 0.55)
        seg2 = sample_arc(center, R, 0.0, math.pi, ds)
        p_end = seg2[-1]
        seg3 = sample_line(p_end, (p_end[0], -0.15), ds)

    segments.append(seg2)
    segments.append(seg3)
    return concatenate_segments(segments)

def build_local_reference_path(kind, mirror=False):
    if kind == "ROUNDED_L":
        return build_local_path_rounded_L(mirror=mirror)
    if kind == "DOUBLE_LANE_CHANGE":
        return build_local_path_double_lane_change(mirror=mirror)
    if kind == "SLALOM":
        return build_local_path_slalom(mirror=mirror)
    if kind == "CIRCLE_ARC_270":
        return build_local_path_circle_arc_270(mirror=mirror)
    if kind == "HAIRPIN":
        return build_local_path_hairpin(mirror=mirror)
    raise ValueError(f"Unknown trajectory kind: {kind}")


# ============================================================
# Local path -> world path using the body reference
# ============================================================

def rotate_vec2(v, ang):
    c = math.cos(ang)
    s = math.sin(ang)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

def local_to_world_path(local_pts, start_x, start_y, right_hat0, fwd_hat0, yaw_offset_deg=0.0):
    """
    Maps local (r,f) coordinates into world coordinates using the INITIAL
    body frame from /youBot_ref, optionally rotated by a small yaw offset.
    """
    ang = deg2rad(yaw_offset_deg)
    right_hat = rotate_vec2(right_hat0, ang)
    fwd_hat   = rotate_vec2(fwd_hat0, ang)

    world_pts = []
    p0 = np.array([start_x, start_y], dtype=float)
    for r, f in local_pts:
        p = p0 + r * right_hat + f * fwd_hat
        world_pts.append((float(p[0]), float(p[1])))
    return world_pts


# ============================================================
# Path preprocessing and closest-point helpers
# ============================================================

def path_point_and_tangent_at_s(wps, s_cum, s_query):
    s_query = float(clamp(s_query, 0.0, s_cum[-1]))

    for i in range(len(wps) - 1):
        s0 = s_cum[i]
        s1 = s_cum[i + 1]
        if s_query <= s1 or i == len(wps) - 2:
            x0, y0 = wps[i]
            x1, y1 = wps[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 1e-12:
                return np.array([x0, y0]), np.array([1.0, 0.0]), i
            tau = (s_query - s0) / max(s1 - s0, 1e-12)
            tau = clamp(tau, 0.0, 1.0)
            p = np.array([x0 + tau * (x1 - x0), y0 + tau * (y1 - y0)])
            t_hat = np.array([(x1 - x0) / seg, (y1 - y0) / seg])
            return p, t_hat, i

    x0, y0 = wps[-2]
    x1, y1 = wps[-1]
    seg = math.hypot(x1 - x0, y1 - y0)
    if seg < 1e-12:
        return np.array([wps[-1][0], wps[-1][1]]), np.array([1.0, 0.0]), len(wps) - 2
    return np.array([wps[-1][0], wps[-1][1]]), np.array([(x1 - x0) / seg, (y1 - y0) / seg]), len(wps) - 2

def closest_projection_on_path(wps, s_cum, x, y, last_seg_idx, back=10, fwd=120):
    px = x
    py = y

    i0 = max(0, last_seg_idx - back)
    i1 = min(len(wps) - 2, last_seg_idx + fwd)

    best = None
    best_d2 = float("inf")

    for i in range(i0, i1 + 1):
        ax, ay = wps[i]
        bx, by = wps[i + 1]

        abx = bx - ax
        aby = by - ay
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            continue

        apx = px - ax
        apy = py - ay

        tau = (apx * abx + apy * aby) / ab2
        tau = clamp(tau, 0.0, 1.0)

        qx = ax + tau * abx
        qy = ay + tau * aby

        dx = px - qx
        dy = py - qy
        d2 = dx * dx + dy * dy

        if d2 < best_d2:
            seg_len = math.sqrt(ab2)
            t_hat = np.array([abx / seg_len, aby / seg_len])
            n_left = np.array([-t_hat[1], t_hat[0]])
            signed_cte = float(np.dot(np.array([px - qx, py - qy]), n_left))
            s_proj = s_cum[i] + tau * seg_len

            best = {
                "seg_idx": i,
                "tau": tau,
                "proj": np.array([qx, qy]),
                "t_hat": t_hat,
                "n_left": n_left,
                "signed_cte": signed_cte,
                "s_proj": s_proj,
            }
            best_d2 = d2

    if best is None:
        p, t_hat, seg_idx = path_point_and_tangent_at_s(wps, s_cum, 0.0)
        n_left = np.array([-t_hat[1], t_hat[0]])
        signed_cte = float(np.dot(np.array([x, y]) - p, n_left))
        best = {
            "seg_idx": seg_idx,
            "tau": 0.0,
            "proj": p,
            "t_hat": t_hat,
            "n_left": n_left,
            "signed_cte": signed_cte,
            "s_proj": 0.0,
        }

    return best


# ============================================================
# Faster-oriented scoring
# ============================================================

def compute_trial_score(metrics):
    score = 0.0

    # must finish
    score += 25000.0 * (not metrics["reached_end"])
    score += 12000.0 * metrics["stalled"]
    score += 8000.0 * (1.0 - metrics["progress_ratio"])

    # enforce acceptable tracking, but not ultra-conservative perfection
    if metrics["final_error"] > 0.08:
        score += 12000.0 * (metrics["final_error"] - 0.08)
    if metrics["mean_path_error"] > 0.03:
        score += 12000.0 * (metrics["mean_path_error"] - 0.03)
    if metrics["max_path_error"] > 0.08:
        score += 12000.0 * (metrics["max_path_error"] - 0.08)

    # once acceptable, strongly prefer speed
    if metrics["reached_end"]:
        score += 60.0 * metrics["time_to_finish"]
    else:
        score += 10.0 * metrics["time_to_finish"]

    # softer secondary preferences
    score += 250.0 * metrics["final_error"]
    score += 220.0 * metrics["mean_path_error"]
    score += 120.0 * metrics["max_path_error"]
    score += 8.0   * metrics["mean_abs_yaw_error"]
    score += 15.0  * metrics["final_heading_error"]
    score += 20.0  * metrics["sat_fraction"]

    return score

def aggregate_candidate_metrics(per_scenario_rows):
    n = len(per_scenario_rows)

    num_success = sum(int(r["reached_end"]) for r in per_scenario_rows)
    num_stalled = sum(int(r["stalled"]) for r in per_scenario_rows)

    progress_vals = [r["progress_ratio"] for r in per_scenario_rows]
    final_err_vals = [r["final_error"] for r in per_scenario_rows]
    mean_path_vals = [r["mean_path_error"] for r in per_scenario_rows]
    max_path_vals = [r["max_path_error"] for r in per_scenario_rows]
    mean_yaw_vals = [r["mean_abs_yaw_error"] for r in per_scenario_rows]
    final_heading_vals = [r["final_heading_error"] for r in per_scenario_rows]
    time_vals = [r["time_to_finish"] for r in per_scenario_rows]
    sat_vals = [r["sat_fraction"] for r in per_scenario_rows]
    trial_scores = [r["trial_score"] for r in per_scenario_rows]

    agg = {
        "num_scenarios": n,
        "num_success": num_success,
        "num_fail": n - num_success,
        "num_stalled": num_stalled,
        "min_progress_ratio": float(min(progress_vals)),
        "mean_progress_ratio": float(np.mean(progress_vals)),
        "mean_final_error": float(np.mean(final_err_vals)),
        "worst_final_error": float(max(final_err_vals)),
        "mean_mean_path_error": float(np.mean(mean_path_vals)),
        "worst_mean_path_error": float(max(mean_path_vals)),
        "mean_max_path_error": float(np.mean(max_path_vals)),
        "worst_max_path_error": float(max(max_path_vals)),
        "mean_abs_yaw_error": float(np.mean(mean_yaw_vals)),
        "mean_final_heading_error": float(np.mean(final_heading_vals)),
        "worst_final_heading_error": float(max(final_heading_vals)),
        "mean_time_to_finish": float(np.mean(time_vals)),
        "mean_sat_fraction": float(np.mean(sat_vals)),
        "mean_trial_score": float(np.mean(trial_scores)),
    }

    # aggregate score aligned with "fast but still good"
    agg["aggregate_score"] = (
        25000.0 * agg["num_fail"]
        + 12000.0 * agg["num_stalled"]
        + 8000.0 * (1.0 - agg["min_progress_ratio"])
        + 12000.0 * max(agg["mean_final_error"] - 0.08, 0.0)
        + 12000.0 * max(agg["mean_mean_path_error"] - 0.03, 0.0)
        + 12000.0 * max(agg["worst_max_path_error"] - 0.08, 0.0)
        + 50.0 * agg["mean_time_to_finish"]
        + 300.0 * agg["mean_final_error"]
        + 250.0 * agg["mean_mean_path_error"]
        + 140.0 * agg["mean_max_path_error"]
        + 80.0  * agg["worst_max_path_error"]
        + 8.0   * agg["mean_abs_yaw_error"]
        + 15.0  * agg["mean_final_heading_error"]
        + 20.0  * agg["mean_sat_fraction"]
    )

    return agg


# ============================================================
# Single trial runner: path-frame controller
# ============================================================

def run_single_trial(sim, base_h, body_h, wheel_handles, wps, s_cum, start_pos, start_ori, params, trial_name=""):
    PREVIEW_DIST = params["PREVIEW_DIST"]
    V_FF = params["V_FF"]
    K_ALONG = params["K_ALONG"]
    K_CROSS = params["K_CROSS"]
    K_YAW = params["K_YAW"]
    SMOOTH_ALPHA = params["SMOOTH_ALPHA"]
    MIN_TRANS_SCALE = params["MIN_TRANS_SCALE"]

    # Corrected wheel mixer, order [fl, rl, rr, fr]
    b_fwd = np.array([-1, -1, -1, -1], dtype=float)
    b_right = np.array([-1, +1, -1, +1], dtype=float)
    b_yaw_ccw = np.array([+1, +1, -1, -1], dtype=float)

    gx, gy = wps[-1]

    sim.setObjectPosition(base_h, -1, start_pos)
    sim.setObjectOrientation(base_h, -1, start_ori)

    sim.setStepping(True)
    sim.startSimulation()

    for _ in range(RESET_SETTLE_STEPS):
        stop_wheels(sim, wheel_handles)
        sim.step()

    last_seg_idx = 0
    w_prev = np.zeros(4)

    path_err_hist = []
    yaw_err_hist = []
    sat_hist = []

    reached_end = False
    stalled = False
    finish_time = MAX_SIM_TIME

    last_x = start_pos[0]
    last_y = start_pos[1]
    last_th_fwd = None

    try:
        t_start = sim.getSimulationTime()

        while True:
            t = sim.getSimulationTime()
            t_rel = t - t_start

            if t_rel > MAX_SIM_TIME:
                finish_time = t_rel
                break

            x, y, _ = get_pose2d(sim, base_h)
            last_x, last_y = x, y
            dg = math.hypot(gx - x, gy - y)

            proj_info = closest_projection_on_path(
                wps, s_cum, x, y, last_seg_idx,
                back=SEARCH_BACK, fwd=SEARCH_FWD
            )
            last_seg_idx = proj_info["seg_idx"]
            progress_ratio = proj_info["s_proj"] / max(s_cum[-1], 1e-12)

            if progress_ratio > 0.98 and dg < END_TOL:
                reached_end = True
                finish_time = t_rel
                break

            if t_rel > STALL_CHECK_TIME and progress_ratio < STALL_REQUIRED_PROGRESS_RATIO:
                stalled = True
                finish_time = t_rel
                break

            if t_rel > EARLY_ABORT_TIME and progress_ratio < EARLY_ABORT_PROGRESS_RATIO:
                stalled = True
                finish_time = t_rel
                break

            s_preview = min(proj_info["s_proj"] + PREVIEW_DIST, s_cum[-1])
            p_preview, t_hat, _ = path_point_and_tangent_at_s(wps, s_cum, s_preview)
            n_left = np.array([-t_hat[1], t_hat[0]])

            pos = np.array([x, y], dtype=float)
            vec_to_preview = p_preview - pos

            e_along = float(np.dot(vec_to_preview, t_hat))
            e_cross = proj_info["signed_cte"]

            # body frame from /youBot_ref
            right_hat, fwd_hat = get_body_axes_from_ref(sim, body_h)
            th_fwd = math.atan2(fwd_hat[1], fwd_hat[0])
            last_th_fwd = th_fwd

            th_path = math.atan2(t_hat[1], t_hat[0])
            e_yaw = wrap_pi(th_path - th_fwd)

            path_err_hist.append(abs(e_cross))
            yaw_err_hist.append(abs(e_yaw))

            # tangent command: feedforward + along-track feedback
            v_t = V_FF + K_ALONG * e_along

            # looser heading gating for faster motion
            abs_eyaw = abs(e_yaw)
            if abs_eyaw > HEADING_GATE_3:
                v_t = 0.0
            elif abs_eyaw > HEADING_GATE_2:
                v_t *= 0.25
            elif abs_eyaw > HEADING_GATE_1:
                v_t *= MIN_TRANS_SCALE

            v_t = clamp(v_t, 0.0, U_FWD_MAX)

            # cross-track correction
            v_n = clamp(-K_CROSS * e_cross, -U_RIGHT_MAX, U_RIGHT_MAX)

            # desired world-frame translation
            v_world = v_t * t_hat + v_n * n_left

            # project to robot body axes from /youBot_ref
            u_right_raw = float(np.dot(v_world, right_hat))
            u_fwd_raw   = float(np.dot(v_world, fwd_hat))
            u_yaw_raw   = YAW_SIGN * K_YAW * e_yaw

            u_fwd = clamp(u_fwd_raw, -U_FWD_MAX, U_FWD_MAX)
            u_right = clamp(u_right_raw, -U_RIGHT_MAX, U_RIGHT_MAX)
            u_yaw = clamp(u_yaw_raw, -U_YAW_MAX, U_YAW_MAX)

            sat_flag = int(
                (abs(u_fwd_raw) >= U_FWD_MAX) or
                (abs(u_right_raw) >= U_RIGHT_MAX) or
                (abs(u_yaw_raw) >= U_YAW_MAX)
            )

            w_unsat = u_fwd * b_fwd + u_right * b_right + u_yaw * b_yaw_ccw
            if np.any(np.abs(w_unsat) >= WHEEL_MAX):
                sat_flag = 1
            sat_hist.append(sat_flag)

            w_clip = np.clip(w_unsat, -WHEEL_MAX, WHEEL_MAX)
            w_cmd = (1.0 - SMOOTH_ALPHA) * w_prev + SMOOTH_ALPHA * w_clip
            w_prev = w_cmd.copy()

            set_wheels(sim, wheel_handles, w_cmd)
            sim.step()

            if DT_SLEEP > 0:
                time.sleep(DT_SLEEP)

    finally:
        stop_wheels(sim, wheel_handles)
        sim.stopSimulation()
        wait_after_stop()

    progress_ratio = proj_info["s_proj"] / max(s_cum[-1], 1e-12)
    final_err = math.hypot(gx - last_x, gy - last_y)

    # corrected heading metric: use last in-simulation heading
    if last_th_fwd is None:
        final_heading_err = 1e6
    else:
        th_path_end = math.atan2(wps[-1][1] - wps[-2][1], wps[-1][0] - wps[-2][0])
        final_heading_err = abs(wrap_pi(th_path_end - last_th_fwd))

    metrics = {
        "trial_name": trial_name,
        "PREVIEW_DIST": PREVIEW_DIST,
        "V_FF": V_FF,
        "K_ALONG": K_ALONG,
        "K_CROSS": K_CROSS,
        "K_YAW": K_YAW,
        "SMOOTH_ALPHA": SMOOTH_ALPHA,
        "MIN_TRANS_SCALE": MIN_TRANS_SCALE,
        "reached_end": bool(reached_end),
        "stalled": int(stalled),
        "progress_ratio": float(progress_ratio),
        "final_error": float(final_err),
        "time_to_finish": float(finish_time),
        "mean_path_error": float(np.mean(path_err_hist)) if path_err_hist else 1e6,
        "max_path_error": float(np.max(path_err_hist)) if path_err_hist else 1e6,
        "mean_abs_yaw_error": float(np.mean(yaw_err_hist)) if yaw_err_hist else 1e6,
        "final_heading_error": float(final_heading_err),
        "sat_fraction": float(np.mean(sat_hist)) if sat_hist else 1.0,
    }
    metrics["trial_score"] = compute_trial_score(metrics)
    return metrics


# ============================================================
# Scenario builder and candidate evaluation
# ============================================================

def build_world_path_for_scenario(start_x, start_y, right_hat0, fwd_hat0, scenario):
    local_path = build_local_reference_path(
        kind=scenario["traj_kind"],
        mirror=scenario["mirror"]
    )
    wps = local_to_world_path(
        local_pts=local_path,
        start_x=start_x,
        start_y=start_y,
        right_hat0=right_hat0,
        fwd_hat0=fwd_hat0,
        yaw_offset_deg=scenario["yaw_offset_deg"]
    )
    s_cum = np.array([0.0])
    if len(wps) > 1:
        s_cum = [0.0]
        for i in range(len(wps) - 1):
            ds = math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
            s_cum.append(s_cum[-1] + ds)
        s_cum = np.array(s_cum, dtype=float)
    return local_path, wps, s_cum

def evaluate_candidate(sim, base_h, body_h, wheel_handles, start_pos, start_ori, start_x, start_y, right_hat0, fwd_hat0, params, candidate_name):
    per_scenario_rows = []

    for scenario in SCENARIOS:
        _, wps, s_cum = build_world_path_for_scenario(start_x, start_y, right_hat0, fwd_hat0, scenario)

        row = run_single_trial(
            sim=sim,
            base_h=base_h,
            body_h=body_h,
            wheel_handles=wheel_handles,
            wps=wps,
            s_cum=s_cum,
            start_pos=start_pos,
            start_ori=start_ori,
            params=params,
            trial_name=f"{candidate_name}_{scenario['name']}"
        )

        row["scenario_name"] = scenario["name"]
        row["traj_kind"] = scenario["traj_kind"]
        row["mirror"] = int(scenario["mirror"])
        row["yaw_offset_deg"] = scenario["yaw_offset_deg"]
        row["path_length"] = float(s_cum[-1])

        per_scenario_rows.append(row)

    agg = aggregate_candidate_metrics(per_scenario_rows)

    summary = {
        "candidate_name": candidate_name,
        "PREVIEW_DIST": params["PREVIEW_DIST"],
        "V_FF": params["V_FF"],
        "K_ALONG": params["K_ALONG"],
        "K_CROSS": params["K_CROSS"],
        "K_YAW": params["K_YAW"],
        "SMOOTH_ALPHA": params["SMOOTH_ALPHA"],
        "MIN_TRANS_SCALE": params["MIN_TRANS_SCALE"],
        "num_success": agg["num_success"],
        "num_fail": agg["num_fail"],
        "num_stalled": agg["num_stalled"],
        "min_progress_ratio": agg["min_progress_ratio"],
        "mean_progress_ratio": agg["mean_progress_ratio"],
        "mean_final_error": agg["mean_final_error"],
        "worst_final_error": agg["worst_final_error"],
        "mean_mean_path_error": agg["mean_mean_path_error"],
        "worst_mean_path_error": agg["worst_mean_path_error"],
        "mean_max_path_error": agg["mean_max_path_error"],
        "worst_max_path_error": agg["worst_max_path_error"],
        "mean_abs_yaw_error": agg["mean_abs_yaw_error"],
        "mean_final_heading_error": agg["mean_final_heading_error"],
        "worst_final_heading_error": agg["worst_final_heading_error"],
        "mean_time_to_finish": agg["mean_time_to_finish"],
        "mean_sat_fraction": agg["mean_sat_fraction"],
        "mean_trial_score": agg["mean_trial_score"],
        "aggregate_score": agg["aggregate_score"],
    }

    return summary, per_scenario_rows


# ============================================================
# Main
# ============================================================

client = RemoteAPIClient()
sim = client.require("sim")

base_h = sim.getObject(BASE_SHAPE_PATH)
body_h = sim.getObject(BODY_REF_PATH)

wheel_handles = [
    sim.getObject(J_FL),
    sim.getObject(J_RL),
    sim.getObject(J_RR),
    sim.getObject(J_FR),
]

start_pos = sim.getObjectPosition(base_h, -1)
start_ori = sim.getObjectOrientation(base_h, -1)
start_x, start_y, _ = get_pose2d(sim, base_h)

right_hat0, fwd_hat0 = get_body_axes_from_ref(sim, body_h)

print("Start position:", (round(start_x, 3), round(start_y, 3)))
print("Initial body axes from /youBot_ref:")
print("  right_hat =", [round(float(right_hat0[0]), 4), round(float(right_hat0[1]), 4)])
print("  fwd_hat   =", [round(float(fwd_hat0[0]), 4), round(float(fwd_hat0[1]), 4)])
print("Scenarios:")
for sc in SCENARIOS:
    print(f"  - {sc['name']} ({sc['traj_kind']}, mirror={sc['mirror']}, yaw_offset={sc['yaw_offset_deg']})")

# Save reference paths for inspection
ref_rows = []
for scenario in SCENARIOS:
    _, wps, s_cum = build_world_path_for_scenario(start_x, start_y, right_hat0, fwd_hat0, scenario)
    Lref = path_length(wps)
    for i, p in enumerate(wps):
        ref_rows.append({
            "scenario_name": scenario["name"],
            "traj_kind": scenario["traj_kind"],
            "mirror": int(scenario["mirror"]),
            "yaw_offset_deg": scenario["yaw_offset_deg"],
            "path_length": Lref,
            "idx": i,
            "x": p[0],
            "y": p[1],
        })
save_csv(ref_rows, OUT_DIR / "reference_paths.csv")


# ============================================================
# Stage 1
# ============================================================

stage1_summaries = []
stage1_per_scenario = []

stage1_grid = list(itertools.product(
    PREVIEW_DIST_GRID,
    V_FF_GRID,
    K_ALONG_GRID,
    K_CROSS_GRID,
    K_YAW_GRID
))

print(f"\nStage 1 candidate count: {len(stage1_grid)}")
print(f"Each candidate is tested on {len(SCENARIOS)} scenarios")

for trial_idx, (preview_dist, v_ff, k_along, k_cross, k_yaw) in enumerate(stage1_grid, start=1):
    params = {
        "PREVIEW_DIST": preview_dist,
        "V_FF": v_ff,
        "K_ALONG": k_along,
        "K_CROSS": k_cross,
        "K_YAW": k_yaw,
        "SMOOTH_ALPHA": 0.70,
        "MIN_TRANS_SCALE": 0.70,
    }

    summary, per_scenario_rows = evaluate_candidate(
        sim=sim,
        base_h=base_h,
        body_h=body_h,
        wheel_handles=wheel_handles,
        start_pos=start_pos,
        start_ori=start_ori,
        start_x=start_x,
        start_y=start_y,
        right_hat0=right_hat0,
        fwd_hat0=fwd_hat0,
        params=params,
        candidate_name=f"stage1_{trial_idx}"
    )

    stage1_summaries.append(summary)
    stage1_per_scenario.extend(per_scenario_rows)

    print(
        f"[{trial_idx:03d}/{len(stage1_grid):03d}] "
        f"Pd={preview_dist:.3f} Vff={v_ff:.2f} Ka={k_along:.2f} Kc={k_cross:.2f} Ky={k_yaw:.2f} | "
        f"success={summary['num_success']}/{len(SCENARIOS)} "
        f"meanFinal={summary['mean_final_error']:.3f} "
        f"meanPath={summary['mean_mean_path_error']:.3f} "
        f"worstMeanPath={summary['worst_mean_path_error']:.3f} "
        f"worstMax={summary['worst_max_path_error']:.3f} "
        f"meanTime={summary['mean_time_to_finish']:.2f}"
    )

stage1_summaries = sorted(stage1_summaries, key=lambda r: r["aggregate_score"])

save_csv(stage1_summaries, OUT_DIR / "stage1_candidate_summaries.csv")
save_csv(stage1_per_scenario, OUT_DIR / "stage1_per_scenario_rows.csv")

best_stage1 = stage1_summaries[:TOP_K_STAGE1]

print("\nTop Stage 1 candidates:")
for i, r in enumerate(best_stage1, start=1):
    print(
        f"{i}. "
        f"Pd={r['PREVIEW_DIST']:.3f}, "
        f"Vff={r['V_FF']:.2f}, "
        f"Ka={r['K_ALONG']:.2f}, "
        f"Kc={r['K_CROSS']:.2f}, "
        f"Ky={r['K_YAW']:.2f}, "
        f"success={r['num_success']}/{len(SCENARIOS)}, "
        f"meanFinal={r['mean_final_error']:.3f}, "
        f"meanPath={r['mean_mean_path_error']:.3f}, "
        f"worstMeanPath={r['worst_mean_path_error']:.3f}, "
        f"worstMax={r['worst_max_path_error']:.3f}, "
        f"minProg={r['min_progress_ratio']:.3f}, "
        f"meanTime={r['mean_time_to_finish']:.2f}, "
        f"score={r['aggregate_score']:.2f}"
    )


# ============================================================
# Stage 2
# ============================================================

stage2_summaries = []
stage2_per_scenario = []

stage2_trial_count = TOP_K_STAGE1 * len(SMOOTH_ALPHA_GRID) * len(MIN_TRANS_SCALE_GRID)
print(f"\nStage 2 candidate count: {stage2_trial_count}")

counter = 0
for seed in best_stage1:
    for alpha, mts in itertools.product(SMOOTH_ALPHA_GRID, MIN_TRANS_SCALE_GRID):
        counter += 1

        params = {
            "PREVIEW_DIST": seed["PREVIEW_DIST"],
            "V_FF": seed["V_FF"],
            "K_ALONG": seed["K_ALONG"],
            "K_CROSS": seed["K_CROSS"],
            "K_YAW": seed["K_YAW"],
            "SMOOTH_ALPHA": alpha,
            "MIN_TRANS_SCALE": mts,
        }

        summary, per_scenario_rows = evaluate_candidate(
            sim=sim,
            base_h=base_h,
            body_h=body_h,
            wheel_handles=wheel_handles,
            start_pos=start_pos,
            start_ori=start_ori,
            start_x=start_x,
            start_y=start_y,
            right_hat0=right_hat0,
            fwd_hat0=fwd_hat0,
            params=params,
            candidate_name=f"stage2_{counter}"
        )

        stage2_summaries.append(summary)
        stage2_per_scenario.extend(per_scenario_rows)

        print(
            f"[{counter:03d}/{stage2_trial_count:03d}] "
            f"Pd={params['PREVIEW_DIST']:.3f} "
            f"Vff={params['V_FF']:.2f} "
            f"Ka={params['K_ALONG']:.2f} "
            f"Kc={params['K_CROSS']:.2f} "
            f"Ky={params['K_YAW']:.2f} "
            f"A={alpha:.2f} MTS={mts:.2f} | "
            f"success={summary['num_success']}/{len(SCENARIOS)} "
            f"meanFinal={summary['mean_final_error']:.3f} "
            f"meanPath={summary['mean_mean_path_error']:.3f} "
            f"worstMeanPath={summary['worst_mean_path_error']:.3f} "
            f"worstMax={summary['worst_max_path_error']:.3f} "
            f"meanTime={summary['mean_time_to_finish']:.2f}"
        )

stage2_summaries = sorted(stage2_summaries, key=lambda r: r["aggregate_score"])

save_csv(stage2_summaries, OUT_DIR / "stage2_candidate_summaries.csv")
save_csv(stage2_per_scenario, OUT_DIR / "stage2_per_scenario_rows.csv")

best = stage2_summaries[0]

print("\nBest parameters found:")
print(f"PREVIEW_DIST     = {best['PREVIEW_DIST']}")
print(f"V_FF             = {best['V_FF']}")
print(f"K_ALONG          = {best['K_ALONG']}")
print(f"K_CROSS          = {best['K_CROSS']}")
print(f"K_YAW            = {best['K_YAW']}")
print(f"SMOOTH_ALPHA     = {best['SMOOTH_ALPHA']}")
print(f"MIN_TRANS_SCALE  = {best['MIN_TRANS_SCALE']}")

print("\nBest aggregate metrics:")
print(f"num_success              = {best['num_success']} / {len(SCENARIOS)}")
print(f"num_fail                 = {best['num_fail']}")
print(f"num_stalled              = {best['num_stalled']}")
print(f"min_progress_ratio       = {best['min_progress_ratio']:.4f}")
print(f"mean_progress_ratio      = {best['mean_progress_ratio']:.4f}")
print(f"mean_final_error         = {best['mean_final_error']:.4f}")
print(f"worst_final_error        = {best['worst_final_error']:.4f}")
print(f"mean_path_error          = {best['mean_mean_path_error']:.4f}")
print(f"worst_mean_path          = {best['worst_mean_path_error']:.4f}")
print(f"mean_max_path            = {best['mean_max_path_error']:.4f}")
print(f"worst_max_path           = {best['worst_max_path_error']:.4f}")
print(f"mean_final_heading_error = {best['mean_final_heading_error']:.4f}")
print(f"worst_final_heading_err  = {best['worst_final_heading_error']:.4f}")
print(f"mean_abs_yaw_error       = {best['mean_abs_yaw_error']:.4f}")
print(f"mean_time_to_finish      = {best['mean_time_to_finish']:.4f}")
print(f"mean_sat_fraction        = {best['mean_sat_fraction']:.4f}")
print(f"aggregate_score          = {best['aggregate_score']:.4f}")

with open(OUT_DIR / "best_params.txt", "w") as f:
    f.write("Best parameters found\n")
    f.write(f"PREVIEW_DIST     = {best['PREVIEW_DIST']}\n")
    f.write(f"V_FF             = {best['V_FF']}\n")
    f.write(f"K_ALONG          = {best['K_ALONG']}\n")
    f.write(f"K_CROSS          = {best['K_CROSS']}\n")
    f.write(f"K_YAW            = {best['K_YAW']}\n")
    f.write(f"SMOOTH_ALPHA     = {best['SMOOTH_ALPHA']}\n")
    f.write(f"MIN_TRANS_SCALE  = {best['MIN_TRANS_SCALE']}\n")

    f.write("\nBest aggregate metrics\n")
    f.write(f"num_success              = {best['num_success']} / {len(SCENARIOS)}\n")
    f.write(f"num_fail                 = {best['num_fail']}\n")
    f.write(f"num_stalled              = {best['num_stalled']}\n")
    f.write(f"min_progress_ratio       = {best['min_progress_ratio']}\n")
    f.write(f"mean_progress_ratio      = {best['mean_progress_ratio']}\n")
    f.write(f"mean_final_error         = {best['mean_final_error']}\n")
    f.write(f"worst_final_error        = {best['worst_final_error']}\n")
    f.write(f"mean_path_error          = {best['mean_mean_path_error']}\n")
    f.write(f"worst_mean_path          = {best['worst_mean_path_error']}\n")
    f.write(f"mean_max_path            = {best['mean_max_path_error']}\n")
    f.write(f"worst_max_path           = {best['worst_max_path_error']}\n")
    f.write(f"mean_final_heading_error = {best['mean_final_heading_error']}\n")
    f.write(f"worst_final_heading_err  = {best['worst_final_heading_error']}\n")
    f.write(f"mean_abs_yaw_error       = {best['mean_abs_yaw_error']}\n")
    f.write(f"mean_time_to_finish      = {best['mean_time_to_finish']}\n")
    f.write(f"mean_sat_fraction        = {best['mean_sat_fraction']}\n")
    f.write(f"aggregate_score          = {best['aggregate_score']}\n")

    f.write("\nFixed values\n")
    f.write(f"BODY_REF_PATH                    = {BODY_REF_PATH}\n")
    f.write(f"SEARCH_BACK                      = {SEARCH_BACK}\n")
    f.write(f"SEARCH_FWD                       = {SEARCH_FWD}\n")
    f.write(f"U_FWD_MAX                        = {U_FWD_MAX}\n")
    f.write(f"U_RIGHT_MAX                      = {U_RIGHT_MAX}\n")
    f.write(f"U_YAW_MAX                        = {U_YAW_MAX}\n")
    f.write(f"WHEEL_MAX                        = {WHEEL_MAX}\n")
    f.write(f"HEADING_GATE_1                   = {HEADING_GATE_1}\n")
    f.write(f"HEADING_GATE_2                   = {HEADING_GATE_2}\n")
    f.write(f"HEADING_GATE_3                   = {HEADING_GATE_3}\n")
    f.write(f"YAW_SIGN                         = {YAW_SIGN}\n")
    f.write(f"MAX_SIM_TIME                     = {MAX_SIM_TIME}\n")
    f.write(f"END_TOL                          = {END_TOL}\n")
    f.write(f"STALL_CHECK_TIME                 = {STALL_CHECK_TIME}\n")
    f.write(f"STALL_REQUIRED_PROGRESS_RATIO    = {STALL_REQUIRED_PROGRESS_RATIO}\n")
    f.write(f"LARGE_DEVIATION_WARN             = {LARGE_DEVIATION_WARN}\n")
    f.write(f"EARLY_ABORT_TIME                 = {EARLY_ABORT_TIME}\n")
    f.write(f"EARLY_ABORT_PROGRESS_RATIO       = {EARLY_ABORT_PROGRESS_RATIO}\n")
    f.write(f"PATH_DS                          = {PATH_DS}\n")

print(f"\nSaved tuning results to: {OUT_DIR}")
