# PRM planning + path-frame feedforward/feedback tracking for KUKA youBot base
#
# Planning:
#   - pose2d state space in OMPL
#   - 3D collision-aware checking against environment
#   - optional inflated planning proxy
#
# Tracking:
#   - closest projection onto path
#   - path tangent / left-normal frame
#   - feedforward along tangent + along-track feedback
#   - cross-track feedback
#   - yaw alignment to path tangent
#   - strong heading gating
#  body-frame convention:
#   /youBot_ref axes:
#       red/x   = right
#       green/y = forward
#       blue/z  = up
#
# Wheel order:
#   [fl, rl, rr, fr]
#
# wheel mixer:
#   forward  -> [ -1, -1, -1, -1 ]
#   right    -> [ -1, +1, -1, +1 ]
#   yaw CCW  -> [ +1, +1, -1, -1 ]

import math
import time
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# -------------------- Options --------------------
LIVE_PLOTTING = False
SAVE_RESULTS = True
PLOT_UPDATE_EVERY = 10

# Planning collision mode:
# False -> use full robot tree as planning collider
# True  -> use a simplified/inflated proxy shape attached to /youBot
USE_PLANNING_PROXY = True
PLANNING_PROXY_PATH = "/youBot/PlanningProxy"

# If planner still sneaks through table edges, lower this.
STATE_VALIDITY_RESOLUTION = 0.0001

# Densify PRM polyline for smoother tracking
DENSIFY_PATH_DS = 0.03   # meters; set to None or 0 to disable

# Drawing
DRAW_Z = 0.01
DRAW_DECIMATE = 1

# -------------------- Scene paths --------------------
ROBOT_ROOT_PATH = "/youBot"
BASE_SHAPE_PATH = "/youBot"
BODY_REF_PATH   = "/youBot_ref"
GOAL_PATH = "/Goal"
FLOOR_PATH = "/Floor"

J_FL = "/youBot/rollingJoint_fl"
J_RL = "/youBot/rollingJoint_rl"
J_RR = "/youBot/rollingJoint_rr"
J_FR = "/youBot/rollingJoint_fr"

# -------------------- Drawing cleanup signal --------------------
PATH_DRAWING_SIGNAL = "OAI_PRM_PATH_DRAWING_HANDLE"

# -------------------- PRM --------------------
BOUNDS_LOW = [-2.5, -2.5]
BOUNDS_HIGH = [2.5, 2.5]
MAX_PLAN_TIME_SEC = 20.0

# -------------------- Tracking: best tuned values --------------------
GOAL_TOL = 0.10
MAX_SIM_TIME = 60.0

SEARCH_BACK = 10
SEARCH_FWD = 120

PREVIEW_DIST = 0.14
V_FF = 1.6
K_ALONG = 10.0
K_CROSS = 22.0
K_YAW = 2.0

U_FWD_MAX = 20.0
U_RIGHT_MAX = 20.0
U_YAW_MAX = 4.0

WHEEL_MAX = 25.0
SMOOTH_ALPHA = 0.55

# strong heading gating thresholds
HEADING_GATE_1 = 0.55
HEADING_GATE_2 = 0.95
HEADING_GATE_3 = 1.35
MIN_TRANS_SCALE = 0.55

YAW_SIGN = 1.0

DT_SLEEP = 0.001

# -------------------- Output --------------------
SAVE_DIR = Path("prm_youbot_results_pathframe_best_bodyref") / datetime.now().strftime("%Y%m%d_%H%M%S")
if SAVE_RESULTS:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

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

def remove_previous_path_drawing(sim, signal_name):
    try:
        old_handle = sim.getInt32Signal(signal_name)
        if old_handle is not None:
            try:
                sim.removeDrawingObject(int(old_handle))
            except Exception:
                pass
        try:
            sim.clearInt32Signal(signal_name)
        except Exception:
            pass
    except Exception:
        pass

def draw_path_and_store_handle(sim, wps, signal_name):
    wps_draw = wps[::DRAW_DECIMATE] if DRAW_DECIMATE > 1 else wps
    max_item_count = max(len(wps_draw) + 10, 5000)

    drawing = sim.addDrawingObject(
        sim.drawing_linestrip,
        2,
        0.0,
        -1,
        max_item_count,
        [1.0, 0.0, 0.0]
    )

    pts = []
    for px, py in wps_draw:
        pts.extend([px, py, DRAW_Z])

    sim.addDrawingObjectItem(drawing | sim.handleflag_addmultiple, pts)

    try:
        sim.setInt32Signal(signal_name, int(drawing))
    except Exception:
        pass

    return drawing

def save_csv(log_rows, filepath):
    if not log_rows:
        return
    keys = list(log_rows[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(log_rows)

def densify_polyline(wps, ds=0.03):
    if ds is None or ds <= 0 or len(wps) < 2:
        return wps[:]

    dense = [wps[0]]
    for i in range(len(wps) - 1):
        x0, y0 = wps[i]
        x1, y1 = wps[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-12:
            continue
        n = max(1, int(math.ceil(seg / ds)))
        for k in range(1, n + 1):
            t = k / n
            dense.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return dense

def precompute_path_s(wps):
    s = [0.0]
    for i in range(len(wps) - 1):
        ds = math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
        s.append(s[-1] + ds)
    return np.array(s, dtype=float)

def point_to_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)

def local_path_distance(wps, x, y, idx, back=3, fwd=12):
    i0 = max(0, idx - back)
    i1 = min(len(wps) - 2, idx + fwd)
    best = float("inf")
    for i in range(i0, i1 + 1):
        ax, ay = wps[i]
        bx, by = wps[i + 1]
        d = point_to_segment_distance(x, y, ax, ay, bx, by)
        if d < best:
            best = d
    return best

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

def get_body_axes_from_ref(sim, body_h):
    """
    Assumes /youBot_ref axes:
      +x (red)   = right
      +y (green) = forward
    Returns 2D world-frame unit vectors:
      right_hat, fwd_hat
    """
    m = sim.getObjectMatrix(body_h, -1)

    x_axis = np.array([m[0], m[4], m[8]], dtype=float)  # red = right
    y_axis = np.array([m[1], m[5], m[9]], dtype=float)  # green = forward

    right_hat = x_axis[:2]
    fwd_hat = y_axis[:2]

    right_hat = right_hat / np.linalg.norm(right_hat)
    fwd_hat = fwd_hat / np.linalg.norm(fwd_hat)
    return right_hat, fwd_hat


# ============================================================
# Live plotting setup
# ============================================================

def setup_live_plots(wps, gx, gy):
    plt.ion()

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    ax_xy = axs[0, 0]
    prm_x = [p[0] for p in wps]
    prm_y = [p[1] for p in wps]
    ax_xy.plot(prm_x, prm_y, "--", label="Tracking path")
    line_xy_actual, = ax_xy.plot([], [], label="Actual trajectory")
    line_xy_proj, = ax_xy.plot([], [], alpha=0.8, label="Projection")
    line_xy_preview, = ax_xy.plot([], [], alpha=0.8, label="Preview")
    ax_xy.plot(gx, gy, "x", markersize=10, label="Goal")
    ax_xy.set_title("XY trajectory")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True)
    ax_xy.axis("equal")
    ax_xy.legend()

    ax_dg = axs[0, 1]
    line_dg, = ax_dg.plot([], [], label="dg")
    line_progress, = ax_dg.plot([], [], label="progress_ratio")
    ax_dg.set_title("Goal distance / progress")
    ax_dg.set_xlabel("t [s]")
    ax_dg.grid(True)
    ax_dg.legend()

    ax_err = axs[0, 2]
    line_ealong, = ax_err.plot([], [], label="e_along")
    line_ecross, = ax_err.plot([], [], label="e_cross")
    line_eyaw, = ax_err.plot([], [], label="e_yaw")
    ax_err.axhline(0.0, linestyle="--")
    ax_err.set_title("Path-frame tracking errors")
    ax_err.set_xlabel("t [s]")
    ax_err.grid(True)
    ax_err.legend()

    ax_u = axs[1, 0]
    line_ufwd, = ax_u.plot([], [], label="u_fwd")
    line_uright, = ax_u.plot([], [], label="u_right")
    line_uyaw, = ax_u.plot([], [], label="u_yaw")
    ax_u.axhline(U_FWD_MAX, linestyle="--")
    ax_u.axhline(-U_FWD_MAX, linestyle="--")
    ax_u.axhline(U_RIGHT_MAX, linestyle="--")
    ax_u.axhline(-U_RIGHT_MAX, linestyle="--")
    ax_u.axhline(U_YAW_MAX, linestyle="--")
    ax_u.axhline(-U_YAW_MAX, linestyle="--")
    ax_u.set_title("Control channels")
    ax_u.set_xlabel("t [s]")
    ax_u.grid(True)
    ax_u.legend()

    ax_w = axs[1, 1]
    line_wfl, = ax_w.plot([], [], label="fl")
    line_wrl, = ax_w.plot([], [], label="rl")
    line_wrr, = ax_w.plot([], [], label="rr")
    line_wfr, = ax_w.plot([], [], label="fr")
    ax_w.axhline(WHEEL_MAX, linestyle="--")
    ax_w.axhline(-WHEEL_MAX, linestyle="--")
    ax_w.set_title("Wheel commands")
    ax_w.set_xlabel("t [s]")
    ax_w.grid(True)
    ax_w.legend()

    ax_sat = axs[1, 2]
    line_sat_ufwd, = ax_sat.plot([], [], label="sat_u_fwd")
    line_sat_uright, = ax_sat.plot([], [], label="sat_u_right")
    line_sat_uyaw, = ax_sat.plot([], [], label="sat_u_yaw")
    line_sat_wheel, = ax_sat.plot([], [], label="sat_wheel")
    ax_sat.set_title("Saturation flags")
    ax_sat.set_xlabel("t [s]")
    ax_sat.set_ylim([-0.1, 1.1])
    ax_sat.grid(True)
    ax_sat.legend()

    plt.tight_layout()
    plt.show(block=False)

    return {
        "fig": fig,
        "axs": axs,
        "line_xy_actual": line_xy_actual,
        "line_xy_proj": line_xy_proj,
        "line_xy_preview": line_xy_preview,
        "line_dg": line_dg,
        "line_progress": line_progress,
        "line_ealong": line_ealong,
        "line_ecross": line_ecross,
        "line_eyaw": line_eyaw,
        "line_ufwd": line_ufwd,
        "line_uright": line_uright,
        "line_uyaw": line_uyaw,
        "line_wfl": line_wfl,
        "line_wrl": line_wrl,
        "line_wrr": line_wrr,
        "line_wfr": line_wfr,
        "line_sat_ufwd": line_sat_ufwd,
        "line_sat_uright": line_sat_uright,
        "line_sat_uyaw": line_sat_uyaw,
        "line_sat_wheel": line_sat_wheel,
    }

def update_live_plots(handles, hist):
    handles["line_xy_actual"].set_data(hist["x"], hist["y"])
    handles["line_xy_proj"].set_data(hist["qx"], hist["qy"])
    handles["line_xy_preview"].set_data(hist["px"], hist["py"])

    handles["line_dg"].set_data(hist["t"], hist["dg"])
    handles["line_progress"].set_data(hist["t"], hist["progress_ratio"])

    handles["line_ealong"].set_data(hist["t"], hist["e_along"])
    handles["line_ecross"].set_data(hist["t"], hist["e_cross"])
    handles["line_eyaw"].set_data(hist["t"], hist["e_yaw"])

    handles["line_ufwd"].set_data(hist["t"], hist["u_fwd"])
    handles["line_uright"].set_data(hist["t"], hist["u_right"])
    handles["line_uyaw"].set_data(hist["t"], hist["u_yaw"])

    handles["line_wfl"].set_data(hist["t"], hist["w_fl"])
    handles["line_wrl"].set_data(hist["t"], hist["w_rl"])
    handles["line_wrr"].set_data(hist["t"], hist["w_rr"])
    handles["line_wfr"].set_data(hist["t"], hist["w_fr"])

    handles["line_sat_ufwd"].set_data(hist["t"], hist["sat_u_fwd"])
    handles["line_sat_uright"].set_data(hist["t"], hist["sat_u_right"])
    handles["line_sat_uyaw"].set_data(hist["t"], hist["sat_u_yaw"])
    handles["line_sat_wheel"].set_data(hist["t"], hist["sat_wheel"])

    for ax in handles["axs"].flat:
        ax.relim()
        ax.autoscale_view()

    handles["axs"][1, 2].set_ylim([-0.1, 1.1])
    plt.pause(0.001)


# ============================================================
# Connect
# ============================================================

client = RemoteAPIClient()
sim = client.require("sim")
simOMPL = client.require("simOMPL")

robot_root = sim.getObject(ROBOT_ROOT_PATH)
base_h = sim.getObject(BASE_SHAPE_PATH)
body_h = sim.getObject(BODY_REF_PATH)
goal_h = sim.getObject(GOAL_PATH)
floor_h = sim.getObject(FLOOR_PATH)

wheel_handles = [
    sim.getObject(J_FL),
    sim.getObject(J_RL),
    sim.getObject(J_RR),
    sim.getObject(J_FR),
]

# -------------------- Planning collision collections --------------------
robotCol = sim.createCollection(0)
sim.addItemToCollection(robotCol, sim.handle_tree, robot_root, 0)

envCol = sim.createCollection(0)
sim.addItemToCollection(envCol, sim.handle_all, -1, 0)
sim.addItemToCollection(envCol, sim.handle_tree, robot_root, 1)
sim.addItemToCollection(envCol, sim.handle_single, floor_h, 1)

planningCollider = robotCol
if USE_PLANNING_PROXY:
    planning_proxy_h = sim.getObject(PLANNING_PROXY_PATH)
    planningProxyCol = sim.createCollection(0)
    sim.addItemToCollection(planningProxyCol, sim.handle_single, planning_proxy_h, 0)
    planningCollider = planningProxyCol

# current start / goal
sx, sy, sth = get_pose2d(sim, base_h)
gx, gy, gth = get_pose2d(sim, goal_h)

# -------------------- Plan PRM --------------------
task = simOMPL.createTask("prm_task")
simOMPL.setAlgorithm(task, simOMPL.Algorithm.PRM)

ss = simOMPL.createStateSpace(
    "base",
    simOMPL.StateSpaceType.pose2d,
    base_h,
    BOUNDS_LOW,
    BOUNDS_HIGH,
    1
)
simOMPL.setStateSpace(task, [ss])

simOMPL.setCollisionPairs(task, [planningCollider, envCol])
simOMPL.setStateValidityCheckingResolution(task, STATE_VALIDITY_RESOLUTION)

simOMPL.setStartState(task, [sx, sy, sth])
simOMPL.setGoalState(task, [gx, gy, gth])
simOMPL.setup(task)

print("Planning mode:", "proxy" if USE_PLANNING_PROXY else "full robot")
print("start valid?", simOMPL.isStateValid(task, [sx, sy, sth]))
print("goal  valid?", simOMPL.isStateValid(task, [gx, gy, gth]))

solved, path = simOMPL.compute(task, MAX_PLAN_TIME_SEC)
simOMPL.destroyTask(task)

if not solved or len(path) < 6:
    raise RuntimeError(
        "PRM failed. Check that tables/obstacles are marked Collidable, "
        "that the start/goal are valid, and that bounds/clearance are reasonable."
    )

wps_raw = [(path[i], path[i + 1]) for i in range(0, len(path), 3)]
if DENSIFY_PATH_DS is not None and DENSIFY_PATH_DS > 0:
    wps = densify_polyline(wps_raw, DENSIFY_PATH_DS)
else:
    wps = wps_raw[:]

s_cum = precompute_path_s(wps)

print("PRM solved. Raw waypoints:", len(wps_raw))
print("Tracking waypoints:", len(wps))
print("Tracking path length [m]:", round(float(s_cum[-1]), 3))

# -------------------- Clear old red path and draw new one --------------------
remove_previous_path_drawing(sim, PATH_DRAWING_SIGNAL)
_ = draw_path_and_store_handle(sim, wps, PATH_DRAWING_SIGNAL)

# mixer, wheel order [fl, rl, rr, fr]
b_fwd = np.array([-1, -1, -1, -1], dtype=float)
b_right = np.array([-1, +1, -1, +1], dtype=float)
b_yaw_ccw = np.array([+1, +1, -1, -1], dtype=float)

# -------------------- Logging --------------------
log_rows = []
hist = {
    "t": [], "x": [], "y": [],
    "qx": [], "qy": [],
    "px": [], "py": [],
    "dg": [], "progress_ratio": [],
    "e_along": [], "e_cross": [], "e_yaw": [],
    "u_fwd": [], "u_right": [], "u_yaw": [],
    "w_fl": [], "w_rl": [], "w_rr": [], "w_fr": [],
    "sat_u_fwd": [], "sat_u_right": [], "sat_u_yaw": [], "sat_wheel": [],
}

plot_handles = setup_live_plots(wps, gx, gy) if LIVE_PLOTTING else None

# -------------------- Follow path --------------------
sim.setStepping(True)
sim.startSimulation()

last_seg_idx = 0
collisions = 0
w_prev = np.zeros(4)
step_count = 0

#  final metrics: store last valid in-sim state
last_x = sx
last_y = sy
last_th_fwd = None

try:
    print("Following PRM path with corrected best path-frame controller...")

    t_start = sim.getSimulationTime()
    last_print = t_start

    while True:
        t = sim.getSimulationTime()
        t_rel = t - t_start

        if t_rel > MAX_SIM_TIME:
            print("Timeout.")
            break

        x, y, _ = get_pose2d(sim, base_h)
        last_x, last_y = x, y
        dg = math.hypot(gx - x, gy - y)

        # Full robot execution-time collision check
        col, _ = sim.checkCollision(robotCol, envCol)
        if col:
            collisions += 1

        proj_info = closest_projection_on_path(
            wps, s_cum, x, y, last_seg_idx,
            back=SEARCH_BACK, fwd=SEARCH_FWD
        )
        last_seg_idx = proj_info["seg_idx"]

        progress_ratio = proj_info["s_proj"] / max(s_cum[-1], 1e-12)

        if progress_ratio > 0.98 and dg < GOAL_TOL:
            print("Reached goal tolerance.")
            break

        # preview point / tangent
        s_preview = min(proj_info["s_proj"] + PREVIEW_DIST, s_cum[-1])
        p_preview, t_hat, _ = path_point_and_tangent_at_s(wps, s_cum, s_preview)
        n_left = np.array([-t_hat[1], t_hat[0]])

        pos = np.array([x, y])
        vec_to_preview = p_preview - pos

        # path-frame errors
        e_along = float(np.dot(vec_to_preview, t_hat))
        e_cross = proj_info["signed_cte"]

        # desired heading = path tangent
        th_path = math.atan2(t_hat[1], t_hat[0])

        # corrected robot body axes from /youBot_ref
        right_hat, fwd_hat = get_body_axes_from_ref(sim, body_h)
        th_fwd = math.atan2(fwd_hat[1], fwd_hat[0])
        last_th_fwd = th_fwd

        e_yaw = wrap_pi(th_path - th_fwd)

        # tangent command
        v_t = V_FF + K_ALONG * e_along

        # strong heading gating
        abs_eyaw = abs(e_yaw)
        if abs_eyaw > HEADING_GATE_3:
            v_t = 0.0
        elif abs_eyaw > HEADING_GATE_2:
            v_t *= 0.25
        elif abs_eyaw > HEADING_GATE_1:
            v_t *= MIN_TRANS_SCALE

        v_t = clamp(v_t, 0.0, U_FWD_MAX)

        # cross-track command
        v_n = clamp(-K_CROSS * e_cross, -U_RIGHT_MAX, U_RIGHT_MAX)

        # desired world-frame translation
        v_world = v_t * t_hat + v_n * n_left

        # project to body-frame commands using corrected body frame
        u_right_raw = float(np.dot(v_world, right_hat))
        u_fwd_raw = float(np.dot(v_world, fwd_hat))
        u_yaw_raw = YAW_SIGN * K_YAW * e_yaw

        u_fwd = clamp(u_fwd_raw, -U_FWD_MAX, U_FWD_MAX)
        u_right = clamp(u_right_raw, -U_RIGHT_MAX, U_RIGHT_MAX)
        u_yaw = clamp(u_yaw_raw, -U_YAW_MAX, U_YAW_MAX)

        w_unsat = u_fwd * b_fwd + u_right * b_right + u_yaw * b_yaw_ccw
        w_clip = np.clip(w_unsat, -WHEEL_MAX, WHEEL_MAX)

        w_cmd = (1.0 - SMOOTH_ALPHA) * w_prev + SMOOTH_ALPHA * w_clip
        w_prev = w_cmd.copy()

        set_wheels(sim, wheel_handles, w_cmd)

        log_rows.append({
            "t": t_rel,
            "x": x,
            "y": y,
            "th_fwd": th_fwd,
            "th_path": th_path,

            "qx": float(proj_info["proj"][0]),
            "qy": float(proj_info["proj"][1]),
            "px": float(p_preview[0]),
            "py": float(p_preview[1]),

            "dx_goal": gx - x,
            "dy_goal": gy - y,
            "dg": dg,
            "progress_ratio": progress_ratio,
            "s_proj": float(proj_info["s_proj"]),
            "s_preview": float(s_preview),

            "e_along": e_along,
            "e_cross": e_cross,
            "e_yaw": e_yaw,

            "v_t": float(v_t),
            "v_n": float(v_n),

            "u_fwd_raw": u_fwd_raw,
            "u_right_raw": u_right_raw,
            "u_yaw_raw": u_yaw_raw,

            "u_fwd": u_fwd,
            "u_right": u_right,
            "u_yaw": u_yaw,

            "w_unsat_fl": w_unsat[0],
            "w_unsat_rl": w_unsat[1],
            "w_unsat_rr": w_unsat[2],
            "w_unsat_fr": w_unsat[3],

            "w_clip_fl": w_clip[0],
            "w_clip_rl": w_clip[1],
            "w_clip_rr": w_clip[2],
            "w_clip_fr": w_clip[3],

            "w_cmd_fl": w_cmd[0],
            "w_cmd_rl": w_cmd[1],
            "w_cmd_rr": w_cmd[2],
            "w_cmd_fr": w_cmd[3],

            "sat_u_fwd": int(abs(u_fwd_raw) >= U_FWD_MAX),
            "sat_u_right": int(abs(u_right_raw) >= U_RIGHT_MAX),
            "sat_u_yaw": int(abs(u_yaw_raw) >= U_YAW_MAX),
            "sat_wheel": int(np.any(np.abs(w_unsat) >= WHEEL_MAX)),

            "idx_seg": last_seg_idx,
            "collision_flag": int(bool(col)),
            "local_path_dist": local_path_distance(wps, x, y, last_seg_idx),
        })

        hist["t"].append(t_rel)
        hist["x"].append(x)
        hist["y"].append(y)
        hist["qx"].append(float(proj_info["proj"][0]))
        hist["qy"].append(float(proj_info["proj"][1]))
        hist["px"].append(float(p_preview[0]))
        hist["py"].append(float(p_preview[1]))
        hist["dg"].append(dg)
        hist["progress_ratio"].append(progress_ratio)
        hist["e_along"].append(e_along)
        hist["e_cross"].append(e_cross)
        hist["e_yaw"].append(e_yaw)
        hist["u_fwd"].append(u_fwd)
        hist["u_right"].append(u_right)
        hist["u_yaw"].append(u_yaw)
        hist["w_fl"].append(w_cmd[0])
        hist["w_rl"].append(w_cmd[1])
        hist["w_rr"].append(w_cmd[2])
        hist["w_fr"].append(w_cmd[3])
        hist["sat_u_fwd"].append(int(abs(u_fwd_raw) >= U_FWD_MAX))
        hist["sat_u_right"].append(int(abs(u_right_raw) >= U_RIGHT_MAX))
        hist["sat_u_yaw"].append(int(abs(u_yaw_raw) >= U_YAW_MAX))
        hist["sat_wheel"].append(int(np.any(np.abs(w_unsat) >= WHEEL_MAX)))

        sim.step()
        time.sleep(DT_SLEEP)

        step_count += 1
        if LIVE_PLOTTING and (step_count % PLOT_UPDATE_EVERY == 0):
            update_live_plots(plot_handles, hist)

        if t - last_print > 2.0:
            print(
                f"t={t_rel:5.1f}s  dg={dg:5.2f}m  prog={progress_ratio:5.2f}  "
                f"e_along={e_along:+.2f}  e_cross={e_cross:+.2f}  e_yaw={e_yaw:+.2f}"
            )
            last_print = t

finally:
    stop_wheels(sim, wheel_handles)
    sim.stopSimulation()


final_err = math.hypot(gx - last_x, gy - last_y)
if last_th_fwd is not None:
    th_path_end = math.atan2(wps[-1][1] - wps[-2][1], wps[-1][0] - wps[-2][0])
    final_heading_err = abs(wrap_pi(th_path_end - last_th_fwd))
else:
    final_heading_err = float("nan")

print("\nResults:")
print("  final error [m]:", round(final_err, 3))
print("  final heading err [rad]:", round(final_heading_err, 3) if not math.isnan(final_heading_err) else "nan")
print("  collision hits:", collisions)
print("Done.")

if LIVE_PLOTTING:
    update_live_plots(plot_handles, hist)
    plt.ioff()
    plt.show()

# -------------------- Save CSV --------------------
if SAVE_RESULTS:
    save_csv(log_rows, SAVE_DIR / "run_log.csv")

# -------------------- Make final static plots --------------------
if SAVE_RESULTS and log_rows:
    t = np.array([r["t"] for r in log_rows])

    x = np.array([r["x"] for r in log_rows])
    y = np.array([r["y"] for r in log_rows])

    qx_arr = np.array([r["qx"] for r in log_rows])
    qy_arr = np.array([r["qy"] for r in log_rows])

    px_arr = np.array([r["px"] for r in log_rows])
    py_arr = np.array([r["py"] for r in log_rows])

    dg_arr = np.array([r["dg"] for r in log_rows])
    prog_arr = np.array([r["progress_ratio"] for r in log_rows])

    e_along_arr = np.array([r["e_along"] for r in log_rows])
    e_cross_arr = np.array([r["e_cross"] for r in log_rows])
    e_yaw_arr = np.array([r["e_yaw"] for r in log_rows])

    u_fwd_raw_arr = np.array([r["u_fwd_raw"] for r in log_rows])
    u_right_raw_arr = np.array([r["u_right_raw"] for r in log_rows])
    u_yaw_raw_arr = np.array([r["u_yaw_raw"] for r in log_rows])

    u_fwd_arr = np.array([r["u_fwd"] for r in log_rows])
    u_right_arr = np.array([r["u_right"] for r in log_rows])
    u_yaw_arr = np.array([r["u_yaw"] for r in log_rows])

    w_cmd_fl = np.array([r["w_cmd_fl"] for r in log_rows])
    w_cmd_rl = np.array([r["w_cmd_rl"] for r in log_rows])
    w_cmd_rr = np.array([r["w_cmd_rr"] for r in log_rows])
    w_cmd_fr = np.array([r["w_cmd_fr"] for r in log_rows])

    sat_u_fwd = np.array([r["sat_u_fwd"] for r in log_rows])
    sat_u_right = np.array([r["sat_u_right"] for r in log_rows])
    sat_u_yaw = np.array([r["sat_u_yaw"] for r in log_rows])
    sat_wheel = np.array([r["sat_wheel"] for r in log_rows])

    local_path_dist_arr = np.array([r["local_path_dist"] for r in log_rows])

    plt.figure(figsize=(7, 7))
    plt.plot([p[0] for p in wps], [p[1] for p in wps], "--", label="Tracking path")
    plt.plot(x, y, label="Actual trajectory")
    plt.plot(qx_arr, qy_arr, alpha=0.6, label="Closest projection")
    plt.plot(px_arr, py_arr, alpha=0.6, label="Preview point")
    plt.plot(x[0], y[0], "o", label="Start")
    plt.plot(gx, gy, "x", markersize=10, label="Goal")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Planned path vs actual trajectory")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "trajectory_xy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(t, e_along_arr, label="e_along")
    plt.plot(t, e_cross_arr, label="e_cross")
    plt.plot(t, e_yaw_arr, label="e_yaw")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.title("Path-frame tracking errors vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "tracking_errors.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, dg_arr, label="distance to goal")
    plt.plot(t, prog_arr, label="progress_ratio")
    plt.xlabel("time [s]")
    plt.ylabel("value")
    plt.title("Distance to goal / progress vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "distance_progress.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, local_path_dist_arr, label="local path distance")
    plt.xlabel("time [s]")
    plt.ylabel("distance [m]")
    plt.title("Local path distance vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "local_path_distance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, u_fwd_raw_arr, label="u_fwd_raw")
    plt.plot(t, u_fwd_arr, label="u_fwd_clipped")
    plt.axhline(U_FWD_MAX, linestyle="--")
    plt.axhline(-U_FWD_MAX, linestyle="--")
    plt.ylabel("u_fwd")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, u_right_raw_arr, label="u_right_raw")
    plt.plot(t, u_right_arr, label="u_right_clipped")
    plt.axhline(U_RIGHT_MAX, linestyle="--")
    plt.axhline(-U_RIGHT_MAX, linestyle="--")
    plt.ylabel("u_right")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, u_yaw_raw_arr, label="u_yaw_raw")
    plt.plot(t, u_yaw_arr, label="u_yaw_clipped")
    plt.axhline(U_YAW_MAX, linestyle="--")
    plt.axhline(-U_YAW_MAX, linestyle="--")
    plt.xlabel("time [s]")
    plt.ylabel("u_yaw")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Control channels vs time")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "control_channels.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot(t, w_cmd_fl, label="fl")
    plt.plot(t, w_cmd_rl, label="rl")
    plt.plot(t, w_cmd_rr, label="rr")
    plt.plot(t, w_cmd_fr, label="fr")
    plt.axhline(WHEEL_MAX, linestyle="--")
    plt.axhline(-WHEEL_MAX, linestyle="--")
    plt.xlabel("time [s]")
    plt.ylabel("wheel command")
    plt.title("Wheel commands vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "wheel_commands.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, sat_u_fwd, label="sat_u_fwd")
    plt.plot(t, sat_u_right, label="sat_u_right")
    plt.plot(t, sat_u_yaw, label="sat_u_yaw")
    plt.plot(t, sat_wheel, label="sat_wheel")
    plt.xlabel("time [s]")
    plt.ylabel("flag (0/1)")
    plt.title("Saturation flags vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "saturation_flags.png", dpi=200)
    plt.close()

    dx_goal_arr = np.array([r["dx_goal"] for r in log_rows])
    dy_goal_arr = np.array([r["dy_goal"] for r in log_rows])

    plt.figure(figsize=(10, 5))
    plt.plot(t, dx_goal_arr, label="x_goal - x")
    plt.plot(t, dy_goal_arr, label="y_goal - y")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("time [s]")
    plt.ylabel("error [m]")
    plt.title("World-frame error to final goal vs time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "goal_xy_error.png", dpi=200)
    plt.close()

    sat_u_fwd_count = sum(r["sat_u_fwd"] for r in log_rows)
    sat_u_right_count = sum(r["sat_u_right"] for r in log_rows)
    sat_u_yaw_count = sum(r["sat_u_yaw"] for r in log_rows)
    sat_wheel_count = sum(r["sat_wheel"] for r in log_rows)

    summary_lines = [
        f"Planning mode: {'proxy' if USE_PLANNING_PROXY else 'full robot'}",
        f"State validity resolution: {STATE_VALIDITY_RESOLUTION}",
        f"DENSIFY_PATH_DS: {DENSIFY_PATH_DS}",
        f"Raw PRM waypoints: {len(wps_raw)}",
        f"Tracking waypoints: {len(wps)}",
        f"Tracking path length [m]: {float(s_cum[-1]):.4f}",
        f"BODY_REF_PATH: {BODY_REF_PATH}",
        f"PREVIEW_DIST: {PREVIEW_DIST}",
        f"V_FF: {V_FF}",
        f"K_ALONG: {K_ALONG}",
        f"K_CROSS: {K_CROSS}",
        f"K_YAW: {K_YAW}",
        f"SMOOTH_ALPHA: {SMOOTH_ALPHA}",
        f"MIN_TRANS_SCALE: {MIN_TRANS_SCALE}",
        f"HEADING_GATE_1: {HEADING_GATE_1}",
        f"HEADING_GATE_2: {HEADING_GATE_2}",
        f"HEADING_GATE_3: {HEADING_GATE_3}",
        f"Final error [m]: {final_err:.4f}",
        f"Final heading error [rad]: {final_heading_err:.4f}" if not math.isnan(final_heading_err) else "Final heading error [rad]: nan",
        f"Collision hits: {collisions}",
        f"Log samples: {len(log_rows)}",
        f"Saturation count u_fwd: {sat_u_fwd_count}",
        f"Saturation count u_right: {sat_u_right_count}",
        f"Saturation count u_yaw: {sat_u_yaw_count}",
        f"Saturation count wheel: {sat_wheel_count}",
        "Corrected metric note: final pose/heading taken from last in-simulation sample before stopSimulation().",
    ]

    with open(SAVE_DIR / "summary.txt", "w") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nSaved logs and plots to: {SAVE_DIR}")
