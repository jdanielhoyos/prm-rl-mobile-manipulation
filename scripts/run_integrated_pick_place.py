import math
import time
from pathlib import Path

import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from stable_baselines3 import SAC


# ============================================================
# User settings
# ============================================================

MODEL_PATH = r"models\sac_youbot_grasp_v5_c07_edge030_035.zip"
USE_PLANNING_PROXY = True
PLANNING_PROXY_PATH = "/youBot/PlanningProxy"

STATE_VALIDITY_RESOLUTION = 0.0001
DENSIFY_PATH_DS = 0.03
MAX_PLAN_TIME_SEC = 30.0

BOUNDS_LOW = [-2.6, -2.6]
BOUNDS_HIGH = [2.6, 2.6]

GOAL_TOL = 0.10
MAX_TRACK_TIME = 90.0

SEARCH_BACK = 10
SEARCH_FWD = 120

PREVIEW_DIST = 0.14
V_FF = 1.6
K_ALONG = 10.0
K_CROSS = 22.0
K_YAW = 2.0
K_HOLD_POS = 8.0
K_HOLD_YAW = 2.5
HOLD_SMOOTH_ALPHA = 0.35
PICKUP_FINAL_ALIGN_STEPS = 300

U_FWD_MAX = 20.0
U_RIGHT_MAX = 20.0
U_YAW_MAX = 4.0

WHEEL_MAX = 25.0
SMOOTH_ALPHA = 0.55

HEADING_GATE_1 = 0.55
HEADING_GATE_2 = 0.95
HEADING_GATE_3 = 1.35
MIN_TRANS_SCALE = 0.55

# Learned-lift deployment settings
CORRECTION_STEPS = 12
ACTION_REPEAT = 3

# Motion step counts for arm-only moves
ARM_STEPS_SHORT = 6
ARM_STEPS_MED = 40
HOLD_STEPS_SHORT = 3
HOLD_STEPS_LONG = 6
ALIGN_STEPS = 200

# ---------------- Pickup / drop scene values from you ----------------
START_BASE_POS = np.array([+1.80011, -1.82496, +0.09575], dtype=float)
PICK_CUBE_POS = np.array([-1.83441, +1.94691, +0.02000], dtype=float)
DROP_TARGET_POINT = np.array([-0.27503, -1.20007, +0.02000], dtype=float)

# ---------------- Learning-relative docking geometry ----------------
# Learning nominal setup you used:
LEARN_BASE_XY = np.array([+0.40011, +0.02504], dtype=float)
LEARN_CUBE_XY = np.array([+0.42151, +0.49882], dtype=float)
LEARN_DOCK_YAW = 1.5720  # same yaw used around learning nominal docking

# ---------------- Robot / scene paths ----------------
ROBOT_ROOT_PATH = "/youBot"
BASE_SHAPE_PATH = "/youBot"
BODY_REF_PATH = "/youBot_ref"
FLOOR_PATH = "/Floor"
CUBE_PATH = "/Cube"

J_FL = "/youBot/rollingJoint_fl"
J_RL = "/youBot/rollingJoint_rl"
J_RR = "/youBot/rollingJoint_rr"
J_FR = "/youBot/rollingJoint_fr"

ARM_PATHS = [
    "/youBot/youBotArmJoint0",
    "/youBot/youBotArmJoint0/youBotArmJoint1",
    "/youBot/youBotArmJoint0/youBotArmJoint1/youBotArmJoint2",
    "/youBot/youBotArmJoint0/youBotArmJoint1/youBotArmJoint2/youBotArmJoint3",
    "/youBot/youBotArmJoint0/youBotArmJoint1/youBotArmJoint2/youBotArmJoint3/youBotArmJoint4",
    "/youBot/youBotArmJoint0/youBotArmJoint1/youBotArmJoint2/youBotArmJoint3/youBotArmJoint4/youBotGripperJoint1",
    "/youBot/youBotArmJoint0/youBotArmJoint1/youBotArmJoint2/youBotArmJoint3/youBotArmJoint4/youBotGripperJoint2",
]

# ---------------- Arm poses ----------------
HOME = np.array([
    2.94961,
    0.53948,
    0.91490,
    1.26850,
   -0.00010,
    0.02500,
   -0.05000,
], dtype=float)

PRE_OPEN = np.array([
    3.07178,
    1.13446,
    1.04720,
    0.52360,
   -0.00010,
    0.02500,
   -0.05000,
], dtype=float)

READY_TO_GRASP = np.array([
    3.07178,
    1.30900,
    0.87266,
    0.55851,
    0.00000,
    0.02500,
   -0.05000,
], dtype=float)

GRASP_CLOSED = np.array([
    3.07178,
    1.30900,
    0.87266,
    0.55851,
    0.00000,
    0.01100,
   -0.01500,
], dtype=float)

LIFT_POSE = np.array([
    3.07178,
    1.13446,
    0.698132,
    0.52360,
   -0.00010,
    0.01100,
   -0.01500,
], dtype=float)

# For learned residual deployment
LEARN_START = PRE_OPEN.copy()
CLOSED_GRIPPER = GRASP_CLOSED[5:].copy()
LIFT_DELTA_ARM = (LIFT_POSE[:5] - GRASP_CLOSED[:5]).astype(float)

# Same v5 residual correction settings
# Match the final v5 training environment
SCRIPTED_PRE_CLOSED = np.array([
    3.07178,
    1.13446,
    1.04720,
    0.52360,
   -0.00010,
    0.01100,
   -0.01500,
], dtype=float)

LIFT_DELTA_ARM = (SCRIPTED_PRE_CLOSED[:5] - GRASP_CLOSED[:5]).astype(float)

ACTION_SCALE = np.array([0.012, 0.010, 0.010, 0.008, 0.006], dtype=float)
ARM_MARGIN = np.array([0.16, 0.12, 0.12, 0.10, 0.08], dtype=float)
ARM_LO = LEARN_START[:5] - ARM_MARGIN
ARM_HI = LEARN_START[:5] + ARM_MARGIN

REL_ERR_SCALE = 0.05
BASE_ERR_SCALE = 0.05
DIST_SCALE = 0.05


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

def set_wheels(sim, wheel_handles, w):
    for h, wi in zip(wheel_handles, w):
        sim.setJointTargetVelocity(h, float(wi))

def stop_wheels(sim, wheel_handles):
    for h in wheel_handles:
        sim.setJointTargetVelocity(h, 0.0)

def set_arm_targets(sim, arm_joints, pose7):
    for h, v in zip(arm_joints, pose7):
        sim.setJointTargetPosition(h, float(v))

def set_arm_positions_immediate(sim, arm_joints, pose7):
    for h, v in zip(arm_joints, pose7):
        sim.setJointPosition(h, float(v))
        sim.setJointTargetPosition(h, float(v))

def get_arm_positions(sim, arm_joints):
    return np.array([sim.getJointPosition(h) for h in arm_joints], dtype=float)

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

        apx = x - ax
        apy = y - ay
        tau = clamp((apx * abx + apy * aby) / ab2, 0.0, 1.0)

        qx = ax + tau * abx
        qy = ay + tau * aby
        dx = x - qx
        dy = y - qy
        d2 = dx * dx + dy * dy

        if d2 < best_d2:
            seg_len = math.sqrt(ab2)
            t_hat = np.array([abx / seg_len, aby / seg_len])
            n_left = np.array([-t_hat[1], t_hat[0]])
            signed_cte = float(np.dot(np.array([x - qx, y - qy]), n_left))
            s_proj = s_cum[i] + tau * seg_len
            best = {
                "seg_idx": i,
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
            "proj": p,
            "t_hat": t_hat,
            "n_left": n_left,
            "signed_cte": signed_cte,
            "s_proj": 0.0,
        }
    return best

def get_body_axes_from_ref(sim, body_h):
    m = sim.getObjectMatrix(body_h, -1)
    x_axis = np.array([m[0], m[4], m[8]], dtype=float)  # right
    y_axis = np.array([m[1], m[5], m[9]], dtype=float)  # forward
    right_hat = x_axis[:2] / np.linalg.norm(x_axis[:2])
    fwd_hat = y_axis[:2] / np.linalg.norm(y_axis[:2])
    return right_hat, fwd_hat

def hold_base_pose_step(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref, w_prev):
    x, y, _ = get_pose2d(sim, base_h)
    right_hat, fwd_hat = get_body_axes_from_ref(sim, body_h)
    yaw_now = math.atan2(fwd_hat[1], fwd_hat[0])

    e_world = np.array([x_ref - x, y_ref - y], dtype=float)
    e_right = float(np.dot(e_world, right_hat))
    e_fwd = float(np.dot(e_world, fwd_hat))
    e_yaw = wrap_pi(yaw_ref - yaw_now)

    u_right = clamp(K_HOLD_POS * e_right, -2.0, 2.0)
    u_fwd = clamp(K_HOLD_POS * e_fwd, -2.0, 2.0)
    u_yaw = clamp(K_HOLD_YAW * e_yaw, -1.0, 1.0)

    b_fwd = np.array([-1, -1, -1, -1], dtype=float)
    b_right = np.array([-1, +1, -1, +1], dtype=float)
    b_yaw = np.array([+1, +1, -1, -1], dtype=float)

    w_unsat = u_fwd * b_fwd + u_right * b_right + u_yaw * b_yaw
    w_clip = np.clip(w_unsat, -6.0, 6.0)
    w_cmd = (1.0 - HOLD_SMOOTH_ALPHA) * w_prev + HOLD_SMOOTH_ALPHA * w_clip
    set_wheels(sim, wheel_handles, w_cmd)
    return w_cmd

def interp_pose(p0, p1, alpha):
    return (1.0 - alpha) * p0 + alpha * p1

def move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                            pose_from, pose_to, steps, x_ref, y_ref, yaw_ref,
                            arm_hold_override=None):
    w_prev = np.zeros(4, dtype=float)
    for k in range(1, steps + 1):
        a = k / steps
        pose = interp_pose(pose_from, pose_to, a)
        w_prev = hold_base_pose_step(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref, w_prev)
        set_arm_targets(sim, arm_joints, pose)
        if arm_hold_override is not None:
            set_arm_targets(sim, arm_joints, arm_hold_override)
        sim.step()
    stop_wheels(sim, wheel_handles)

def align_base_pose(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref,
                    arm_joints=None, arm_pose=None, steps=ALIGN_STEPS):
    w_prev = np.zeros(4, dtype=float)
    for _ in range(steps):
        w_prev = hold_base_pose_step(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref, w_prev)
        if arm_joints is not None and arm_pose is not None:
            set_arm_targets(sim, arm_joints, arm_pose)
        sim.step()
    stop_wheels(sim, wheel_handles)

def yaw_to_body_axes(yaw_fwd):
    fwd = np.array([math.cos(yaw_fwd), math.sin(yaw_fwd)], dtype=float)
    right = np.array([math.sin(yaw_fwd), -math.cos(yaw_fwd)], dtype=float)
    return right, fwd

def compute_dock_relative():
    right_hat, fwd_hat = yaw_to_body_axes(LEARN_DOCK_YAW)
    rel = LEARN_CUBE_XY - LEARN_BASE_XY
    rel_right = float(np.dot(rel, right_hat))
    rel_fwd = float(np.dot(rel, fwd_hat))
    return rel_right, rel_fwd

def compute_dock_base_xy(target_xy, dock_yaw, rel_right, rel_fwd):
    right_hat, fwd_hat = yaw_to_body_axes(dock_yaw)
    base_xy = np.array(target_xy, dtype=float) - rel_right * right_hat - rel_fwd * fwd_hat
    return base_xy

def safe_get_handle(sim, path):
    try:
        return sim.getObject(path)
    except Exception:
        return None


# ============================================================
# Planning / tracking
# ============================================================

def build_collections(sim, robot_root, floor_h, cube_h, use_proxy):
    robotCol = sim.createCollection(0)
    sim.addItemToCollection(robotCol, sim.handle_tree, robot_root, 0)

    envCol_with_cube = sim.createCollection(0)
    sim.addItemToCollection(envCol_with_cube, sim.handle_all, -1, 0)
    sim.addItemToCollection(envCol_with_cube, sim.handle_tree, robot_root, 1)
    sim.addItemToCollection(envCol_with_cube, sim.handle_single, floor_h, 1)

    envCol_no_cube = sim.createCollection(0)
    sim.addItemToCollection(envCol_no_cube, sim.handle_all, -1, 0)
    sim.addItemToCollection(envCol_no_cube, sim.handle_tree, robot_root, 1)
    sim.addItemToCollection(envCol_no_cube, sim.handle_single, floor_h, 1)
    sim.addItemToCollection(envCol_no_cube, sim.handle_single, cube_h, 1)

    planningCollider = robotCol
    if use_proxy:
        planning_proxy_h = sim.getObject(PLANNING_PROXY_PATH)
        planningProxyCol = sim.createCollection(0)
        sim.addItemToCollection(planningProxyCol, sim.handle_single, planning_proxy_h, 0)
        planningCollider = planningProxyCol

    return robotCol, envCol_with_cube, envCol_no_cube, planningCollider

def plan_prm(sim, simOMPL, base_h, planningCollider, envCol, start_state, goal_state):
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
    simOMPL.setStartState(task, list(start_state))
    simOMPL.setGoalState(task, list(goal_state))
    simOMPL.setup(task)

    solved, path = simOMPL.compute(task, MAX_PLAN_TIME_SEC)
    simOMPL.destroyTask(task)

    if (not solved) or len(path) < 6:
        raise RuntimeError(f"PRM failed from {start_state} to {goal_state}")

    wps_raw = [(path[i], path[i + 1]) for i in range(0, len(path), 3)]
    wps = densify_polyline(wps_raw, DENSIFY_PATH_DS)
    return wps

def follow_path(sim, base_h, body_h, wheel_handles, wps, goal_xy,
                arm_joints=None, arm_pose=None, envCol_exec=None, robotCol=None):
    s_cum = precompute_path_s(wps)

    b_fwd = np.array([-1, -1, -1, -1], dtype=float)
    b_right = np.array([-1, +1, -1, +1], dtype=float)
    b_yaw = np.array([+1, +1, -1, -1], dtype=float)

    sim_t0 = sim.getSimulationTime()
    w_prev = np.zeros(4, dtype=float)
    last_seg_idx = 0

    while True:
        t_rel = sim.getSimulationTime() - sim_t0
        if t_rel > MAX_TRACK_TIME:
            raise RuntimeError("Path tracking timeout")

        x, y, _ = get_pose2d(sim, base_h)
        dg = math.hypot(goal_xy[0] - x, goal_xy[1] - y)

        proj_info = closest_projection_on_path(
            wps, s_cum, x, y, last_seg_idx,
            back=SEARCH_BACK, fwd=SEARCH_FWD
        )
        last_seg_idx = proj_info["seg_idx"]
        progress_ratio = proj_info["s_proj"] / max(s_cum[-1], 1e-12)

        if progress_ratio > 0.98 and dg < GOAL_TOL:
            break

        s_preview = min(proj_info["s_proj"] + PREVIEW_DIST, s_cum[-1])
        p_preview, t_hat, _ = path_point_and_tangent_at_s(wps, s_cum, s_preview)
        n_left = np.array([-t_hat[1], t_hat[0]])

        pos = np.array([x, y])
        vec_to_preview = p_preview - pos

        e_along = float(np.dot(vec_to_preview, t_hat))
        e_cross = proj_info["signed_cte"]

        th_path = math.atan2(t_hat[1], t_hat[0])

        right_hat, fwd_hat = get_body_axes_from_ref(sim, body_h)
        th_fwd = math.atan2(fwd_hat[1], fwd_hat[0])
        e_yaw = wrap_pi(th_path - th_fwd)

        v_t = V_FF + K_ALONG * e_along
        abs_eyaw = abs(e_yaw)
        if abs_eyaw > HEADING_GATE_3:
            v_t = 0.0
        elif abs_eyaw > HEADING_GATE_2:
            v_t *= 0.25
        elif abs_eyaw > HEADING_GATE_1:
            v_t *= MIN_TRANS_SCALE
        v_t = clamp(v_t, 0.0, U_FWD_MAX)

        v_n = clamp(-K_CROSS * e_cross, -U_RIGHT_MAX, U_RIGHT_MAX)
        v_world = v_t * t_hat + v_n * n_left

        u_right_raw = float(np.dot(v_world, right_hat))
        u_fwd_raw = float(np.dot(v_world, fwd_hat))
        u_yaw_raw = K_YAW * e_yaw

        u_fwd = clamp(u_fwd_raw, -U_FWD_MAX, U_FWD_MAX)
        u_right = clamp(u_right_raw, -U_RIGHT_MAX, U_RIGHT_MAX)
        u_yaw = clamp(u_yaw_raw, -U_YAW_MAX, U_YAW_MAX)

        w_unsat = u_fwd * b_fwd + u_right * b_right + u_yaw * b_yaw
        w_clip = np.clip(w_unsat, -WHEEL_MAX, WHEEL_MAX)
        w_cmd = (1.0 - SMOOTH_ALPHA) * w_prev + SMOOTH_ALPHA * w_clip
        w_prev = w_cmd.copy()
        set_wheels(sim, wheel_handles, w_cmd)

        if arm_joints is not None and arm_pose is not None:
            set_arm_targets(sim, arm_joints, arm_pose)

        sim.step()

    stop_wheels(sim, wheel_handles)


# ============================================================
# Learned lift runner
# ============================================================

class LearnedLiftRunner:
    def __init__(self, model_path, target_rel_ee):
        print(f"Loading learned model: {model_path}")
        self.model = SAC.load(model_path, device="auto")
        self.target_rel_ee = np.array(target_rel_ee, dtype=float)

    def _get_ee_axes(self, sim, ee_ref_h):
        m = sim.getObjectMatrix(ee_ref_h, -1)
        x_axis = np.array([m[0], m[4], m[8]], dtype=float)
        y_axis = np.array([m[1], m[5], m[9]], dtype=float)
        z_axis = np.array([m[2], m[6], m[10]], dtype=float)
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)
        return x_axis, y_axis, z_axis

    def _get_rel_cube_to_gripper_ee(self, sim, cube_h, ee_ref_h, g1_h, g2_h):
        cube = np.array(sim.getObjectPosition(cube_h, -1), dtype=float)
        p1 = np.array(sim.getObjectPosition(g1_h, -1), dtype=float)
        p2 = np.array(sim.getObjectPosition(g2_h, -1), dtype=float)
        ee_mid = 0.5 * (p1 + p2)
        rel_world = cube - ee_mid

        x_axis, y_axis, z_axis = self._get_ee_axes(sim, ee_ref_h)
        rel_ee = np.array([
            float(np.dot(rel_world, x_axis)),
            float(np.dot(rel_world, y_axis)),
            float(np.dot(rel_world, z_axis)),
        ], dtype=float)

        return rel_ee

    def _get_obs(self, sim, base_h, body_h, cube_h, ee_ref_h, g1_h, g2_h, arm_joints,
                 x_ref, y_ref, yaw_ref, episode_step, correction_steps):
        rel_ee = self._get_rel_cube_to_gripper_ee(sim, cube_h, ee_ref_h, g1_h, g2_h)
        err_ee = rel_ee - self.target_rel_ee
        err_ee_scaled = err_ee / REL_ERR_SCALE

        actual = np.array([sim.getJointPosition(h) for h in arm_joints], dtype=float)

        arm_resid_scaled = (actual[:5] - LEARN_START[:5]) / np.maximum(ARM_MARGIN, 1e-6)
        grip_resid_scaled = actual[5:] - LEARN_START[5:]

        x, y, _ = get_pose2d(sim, base_h)
        _, fwd_hat = get_body_axes_from_ref(sim, body_h)
        yaw_now = math.atan2(fwd_hat[1], fwd_hat[0])

        base_dx = (x - x_ref) / BASE_ERR_SCALE
        base_dy = (y - y_ref) / BASE_ERR_SCALE
        base_dyaw = wrap_pi(yaw_now - yaw_ref) / math.pi

        align_xy = float(np.linalg.norm(err_ee[:2]))
        remain = float((correction_steps - episode_step) / max(correction_steps, 1))

        obs = np.array([
            err_ee_scaled[0], err_ee_scaled[1], err_ee_scaled[2],
            arm_resid_scaled[0], arm_resid_scaled[1], arm_resid_scaled[2],
            arm_resid_scaled[3], arm_resid_scaled[4],
            grip_resid_scaled[0], grip_resid_scaled[1],
            base_dx, base_dy, base_dyaw,
            align_xy / DIST_SCALE,
            remain,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    def run(self, sim, base_h, body_h, cube_h, arm_joints, ee_ref_h, g1_h, g2_h,
            wheel_handles, x_ref, y_ref, yaw_ref):
        current_arm = LEARN_START[:5].copy()
        open_gripper = LEARN_START[5:].copy()
        closed_gripper = CLOSED_GRIPPER.copy()

        cube_z_place = float(sim.getObjectPosition(cube_h, -1)[2])

        # Residual correction phase
        for step in range(CORRECTION_STEPS):
            obs = self._get_obs(
                sim, base_h, body_h, cube_h, ee_ref_h, g1_h, g2_h, arm_joints,
                x_ref, y_ref, yaw_ref, step, CORRECTION_STEPS
            )
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=float).reshape(-1)
            action = np.clip(action, -1.0, 1.0)

            current_arm = current_arm + ACTION_SCALE * action
            current_arm = np.minimum(np.maximum(current_arm, ARM_LO), ARM_HI)

            pose_open = np.concatenate([current_arm, open_gripper])

            w_prev = np.zeros(4, dtype=float)
            for _ in range(ACTION_REPEAT):
                w_prev = hold_base_pose_step(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref, w_prev)
                set_arm_targets(sim, arm_joints, pose_open)
                sim.step()
            stop_wheels(sim, wheel_handles)

        # Close
        pose_open = np.concatenate([current_arm, open_gripper])
        pose_closed = np.concatenate([current_arm, closed_gripper])
        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                pose_open, pose_closed, ARM_STEPS_SHORT,
                                x_ref, y_ref, yaw_ref)

        align_base_pose(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref,
                        arm_joints, pose_closed, HOLD_STEPS_SHORT)

        # Lift using the same relative lift used during v5 training
        lift_arm = np.minimum(np.maximum(current_arm + LIFT_DELTA_ARM, ARM_LO), ARM_HI)
        pose_lift = np.concatenate([lift_arm, closed_gripper])
        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                pose_closed, pose_lift, ARM_STEPS_MED,
                                x_ref, y_ref, yaw_ref)

        z_thresh = cube_z_place + 0.015
        success_counter = 0
        lift_amount = 0.0

        align_base_pose(sim, base_h, body_h, wheel_handles, x_ref, y_ref, yaw_ref,
                        arm_joints, pose_lift, HOLD_STEPS_LONG)

        for _ in range(8):
            set_arm_targets(sim, arm_joints, pose_lift)
            sim.step()
            z = float(sim.getObjectPosition(cube_h, -1)[2])
            lift_amount = z - cube_z_place
            if z > z_thresh:
                success_counter += 1
            else:
                success_counter = 0

        success = success_counter >= 3
        return success, pose_lift, lift_amount

# ============================================================
# Main pipeline
# ============================================================
def get_gripper_midpoint(sim, g1_h, g2_h):
    p1 = np.array(sim.getObjectPosition(g1_h, -1), dtype=float)
    p2 = np.array(sim.getObjectPosition(g2_h, -1), dtype=float)
    return 0.5 * (p1 + p2)

def get_rel_cube_to_gripper_ee(sim, cube_h, ee_ref_h, g1_h, g2_h):
    cube = np.array(sim.getObjectPosition(cube_h, -1), dtype=float)
    ee_mid = get_gripper_midpoint(sim, g1_h, g2_h)
    rel_world = cube - ee_mid

    m = sim.getObjectMatrix(ee_ref_h, -1)
    x_axis = np.array([m[0], m[4], m[8]], dtype=float)
    y_axis = np.array([m[1], m[5], m[9]], dtype=float)
    z_axis = np.array([m[2], m[6], m[10]], dtype=float)

    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis /= np.linalg.norm(z_axis)

    return np.array([
        float(np.dot(rel_world, x_axis)),
        float(np.dot(rel_world, y_axis)),
        float(np.dot(rel_world, z_axis)),
    ], dtype=float)

def capture_nominal_target_rel_ee(sim, base_h, cube_h, arm_joints,
                                  ee_ref_h, g1_h, g2_h,
                                  base_ori, cube_ori):
    """
    Recreate the nominal learning setup used by v5 and capture the
    cube-to-gripper relative vector in the EE frame.
    """
    sim.setObjectPosition(base_h, -1, [float(LEARN_BASE_XY[0]), float(LEARN_BASE_XY[1]), 0.09575])
    sim.setObjectOrientation(base_h, -1, base_ori.tolist())

    sim.setObjectPosition(cube_h, -1, [float(LEARN_CUBE_XY[0]), float(LEARN_CUBE_XY[1]), 0.02000])
    sim.setObjectOrientation(cube_h, -1, cube_ori.tolist())

    set_arm_positions_immediate(sim, arm_joints, LEARN_START)

    target_rel_ee = get_rel_cube_to_gripper_ee(sim, cube_h, ee_ref_h, g1_h, g2_h)
    print(f"Captured deployment target_rel_ee = {target_rel_ee.tolist()}")
    return target_rel_ee


def main():
    client = RemoteAPIClient()
    sim = client.require("sim")
    simOMPL = client.require("simOMPL")

    base_h = sim.getObject(BASE_SHAPE_PATH)
    body_h = sim.getObject(BODY_REF_PATH)
    robot_root = sim.getObject(ROBOT_ROOT_PATH)
    floor_h = sim.getObject(FLOOR_PATH)
    cube_h = sim.getObject(CUBE_PATH)

    wheel_handles = [sim.getObject(J_FL), sim.getObject(J_RL), sim.getObject(J_RR), sim.getObject(J_FR)]
    arm_joints = [sim.getObject(p) for p in ARM_PATHS]
    ee_ref_h = arm_joints[4]
    g1_h = arm_joints[5]
    g2_h = arm_joints[6]

    goal_h = safe_get_handle(sim, "/Goal")

    # Capture current object orientations so position reset does not guess them
    start_base_ori = np.array(sim.getObjectOrientation(base_h, -1), dtype=float)
    pick_cube_ori = np.array(sim.getObjectOrientation(cube_h, -1), dtype=float)
    try:
    	sim.stopSimulation()
    except Exception:
        pass
    target_rel_ee = capture_nominal_target_rel_ee(sim, base_h, cube_h, arm_joints,ee_ref_h, g1_h, g2_h,start_base_ori, pick_cube_ori)


    # Compute docking geometry from learning nominal setup
    dock_rel_right, dock_rel_fwd = compute_dock_relative()
    dock_pick_xy = compute_dock_base_xy(PICK_CUBE_POS[:2], LEARN_DOCK_YAW, dock_rel_right, dock_rel_fwd)
    #dock_drop_xy = compute_dock_base_xy(DROP_TARGET_POINT[:2], LEARN_DOCK_YAW, dock_rel_right, dock_rel_fwd)
    dock_drop_xy = compute_dock_base_xy(
    DROP_TARGET_POINT[:2],
    LEARN_DOCK_YAW,
    dock_rel_right,
    dock_rel_fwd + 0.01)

    print("\nComputed docking geometry from learning setup:")
    print(f"  rel_right = {dock_rel_right:.5f} m")
    print(f"  rel_fwd   = {dock_rel_fwd:.5f} m")
    print(f"  pickup dock xy = ({dock_pick_xy[0]:.5f}, {dock_pick_xy[1]:.5f})")
    print(f"  drop   dock xy = ({dock_drop_xy[0]:.5f}, {dock_drop_xy[1]:.5f})")
    print(f"  dock yaw used  = {LEARN_DOCK_YAW:.5f} rad")

    # Reset scene to requested pickup configuration with sim stopped
    try:
        sim.stopSimulation()
    except Exception:
        pass

    sim.setObjectPosition(base_h, -1, START_BASE_POS.tolist())
    sim.setObjectOrientation(base_h, -1, start_base_ori.tolist())

    sim.setObjectPosition(cube_h, -1, PICK_CUBE_POS.tolist())
    sim.setObjectOrientation(cube_h, -1, pick_cube_ori.tolist())

    set_arm_positions_immediate(sim, arm_joints, HOME)
    stop_wheels(sim, wheel_handles)

    if goal_h is not None:
        try:
            sim.setObjectPosition(goal_h, -1, [dock_pick_xy[0], dock_pick_xy[1], 0.03])
        except Exception:
            pass

    # Build planning collections
    robotCol, envCol_with_cube, envCol_no_cube, planningCollider = build_collections(
        sim, robot_root, floor_h, cube_h, USE_PLANNING_PROXY
    )

    # Plan 1: start -> pickup dock
    sx, sy, sth = get_pose2d(sim, base_h)
    start_state = [sx, sy, sth]
    goal_state_pick = [dock_pick_xy[0], dock_pick_xy[1], sth]  # theta here is not critical
    print("\nPlanning path to pickup dock...")
    wps_pick = plan_prm(sim, simOMPL, base_h, planningCollider, envCol_with_cube, start_state, goal_state_pick)
    print(f"  pickup path waypoints: {len(wps_pick)}")

    # Start simulation once and keep it running through the full mission
    sim.setStepping(True)
    sim.startSimulation()

    learned_runner = LearnedLiftRunner(MODEL_PATH, target_rel_ee)

    try:
        # Phase 1: navigate to pickup dock with arm in HOME
        print("\nPhase 1: following PRM path to pickup dock...")
        follow_path(sim, base_h, body_h, wheel_handles, wps_pick, dock_pick_xy,
                    arm_joints=arm_joints, arm_pose=HOME,
                    envCol_exec=envCol_with_cube, robotCol=robotCol)

        print("Aligning base at pickup dock yaw...")
        align_base_pose(sim, base_h, body_h, wheel_handles,
                        dock_pick_xy[0], dock_pick_xy[1], LEARN_DOCK_YAW,
                        arm_joints, HOME, ALIGN_STEPS)

        # Phase 2: move arm to PRE_OPEN
        print("\nPhase 2: moving arm to PRE_OPEN...")
        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                HOME, PRE_OPEN, ARM_STEPS_MED,
                                dock_pick_xy[0], dock_pick_xy[1], LEARN_DOCK_YAW)

        # short settle at PRE_OPEN
        align_base_pose(sim, base_h, body_h, wheel_handles,
                        dock_pick_xy[0], dock_pick_xy[1], LEARN_DOCK_YAW,
                        arm_joints, PRE_OPEN, HOLD_STEPS_SHORT)

        # extra final docking alignment before learned grasp/lift
        print("Extra pickup alignment before learned grasp...")
        align_base_pose(sim, base_h, body_h, wheel_handles,
                        dock_pick_xy[0], dock_pick_xy[1], LEARN_DOCK_YAW,
                        arm_joints, PRE_OPEN, PICKUP_FINAL_ALIGN_STEPS)

        # Phase 3: learned lift
        print("\nPhase 3: learned correction + close + lift...")
        success, carry_pose, lift_amount = learned_runner.run(
            sim, base_h, body_h, cube_h, arm_joints, ee_ref_h, g1_h, g2_h,
            wheel_handles, dock_pick_xy[0], dock_pick_xy[1], LEARN_DOCK_YAW
        )
        print(f"  learned lift success = {success}, lift_amount = {lift_amount:.4f} m")
        if not success:
            raise RuntimeError("Learned lift failed. Aborting mission.")

        # Phase 4: plan carry path to drop dock
        if goal_h is not None:
            try:
                sim.setObjectPosition(goal_h, -1, [dock_drop_xy[0], dock_drop_xy[1], 0.03])
            except Exception:
                pass

        cx, cy, cth = get_pose2d(sim, base_h)
        start_state_carry = [cx, cy, cth]
        goal_state_drop = [dock_drop_xy[0], dock_drop_xy[1], cth]
        print("\nPlanning path to drop dock...")
        wps_drop = plan_prm(sim, simOMPL, base_h, planningCollider, envCol_no_cube, start_state_carry, goal_state_drop)
        print(f"  drop path waypoints: {len(wps_drop)}")

        # Phase 5: carry to drop dock
        print("\nPhase 4: carrying cube to drop dock...")
        follow_path(sim, base_h, body_h, wheel_handles, wps_drop, dock_drop_xy,
                    arm_joints=arm_joints, arm_pose=carry_pose,
                    envCol_exec=envCol_no_cube, robotCol=robotCol)

        print("Aligning base at drop dock yaw...")
        align_base_pose(sim, base_h, body_h, wheel_handles,
                        dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW,
                        arm_joints, carry_pose, 450)

        # Phase 6: lower and drop
        print("\nPhase 5: lowering and dropping cube...")
        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                carry_pose, GRASP_CLOSED, ARM_STEPS_MED,
                                dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW)
        align_base_pose(sim, base_h, body_h, wheel_handles,
                        dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW,
                        arm_joints, GRASP_CLOSED, HOLD_STEPS_SHORT)

        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                GRASP_CLOSED, READY_TO_GRASP, ARM_STEPS_SHORT,
                                dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW)
        align_base_pose(sim, base_h, body_h, wheel_handles,
                        dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW,
                        arm_joints, READY_TO_GRASP, HOLD_STEPS_SHORT)

        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                READY_TO_GRASP, PRE_OPEN, ARM_STEPS_SHORT,
                                dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW)
        move_arm_with_base_hold(sim, arm_joints, wheel_handles, base_h, body_h,
                                PRE_OPEN, HOME, ARM_STEPS_MED,
                                dock_drop_xy[0], dock_drop_xy[1], LEARN_DOCK_YAW)

        print("\nMission complete.")

    finally:
        stop_wheels(sim, wheel_handles)
        try:
            sim.stopSimulation()
        except Exception:
            pass


if __name__ == "__main__":
    main()
