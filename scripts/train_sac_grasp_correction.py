"""
youbot_grasp_correction_curriculum_v5.py

New training file for the youBot arm residual grasp-correction task.

Main changes versus the earlier v4-style script:
  1. The policy observes the ERROR to the manually successful nominal pre-grasp
     geometry, not raw absolute joint positions only.
  2. The dense reward is based on alignment to the captured nominal grasp geometry,
     not blindly forcing cube-to-gripper midpoint distance to zero.
  3. The terminal reward uses partial lift amount, so near-successful grasps are not
     treated the same as completely failed grasps.
  4. Default correction horizon and action scale are larger.
  5. SAC defaults are more conservative and sample-efficient for a slow simulator:
     lower learning rate, lower gamma, more gradient steps per environment step.

Before running:
  - Open the CoppeliaSim scene.
  - Stop the simulation.
  - Place /youBot and /Cube in the desired nominal pose.
  - Place the arm/gripper in the desired OPEN learning-start pre-grasp pose.
  - Make sure the manual scripted close/lift macro still works from that pose.

Example safe 10k chunk:
  python youbot_grasp_correction_curriculum_v5_safe_save.py --mode train \
    --timesteps 10000 --rand-min 0.000 --rand-max 0.008 \
    --correction-steps 12 --action-repeat 3 \
    --model-path models/sac_youbot_grasp_v5_c01.zip \
    --checkpoint-freq 10000 --save-replay-buffer --progress-bar

Resume another 10k chunk:
  python youbot_grasp_correction_curriculum_v5_safe_save.py --mode train --resume \
    --timesteps 10000 --rand-min 0.000 --rand-max 0.008 \
    --correction-steps 12 --action-repeat 3 \
    --model-path models/sac_youbot_grasp_v5_c01.zip \
    --checkpoint-freq 10000 --save-replay-buffer --progress-bar
"""

import argparse
import math
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback


# ============================================================
# Fixed scene / robot paths
# ============================================================

ROBOT_ROOT_PATH = "/youBot"
BODY_REF_PATH = "/youBot_ref"
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

WHEEL_PATHS = [J_FL, J_RL, J_RR, J_FR]


# ============================================================
# Nominal reset positions
# ============================================================

BASE_RESET_POS = np.array([0.40011, 0.02504, 0.09575], dtype=np.float32)
CUBE_RESET_POS = np.array([0.42151, 0.49882, 0.02000], dtype=np.float32)


# ============================================================
# Working close/lift macro from the successful baseline
# ============================================================

SCRIPTED_GRASP_CLOSED = np.array([
    3.07178,
    1.30900,
    0.87266,
    0.55851,
    0.00000,
    0.01100,
   -0.01500,
], dtype=np.float32)

SCRIPTED_PRE_CLOSED = np.array([
    3.07178,
    1.13446,
    1.04720,
    0.52360,
   -0.00010,
    0.01100,
   -0.01500,
], dtype=np.float32)

LIFT_DELTA_ARM = (SCRIPTED_PRE_CLOSED[:5] - SCRIPTED_GRASP_CLOSED[:5]).astype(np.float32)
SCRIPTED_CLOSED_GRIPPER = SCRIPTED_GRASP_CLOSED[5:].copy()


# ============================================================
# Base hold controller
# ============================================================

K_HOLD_POS = 8.0
K_HOLD_YAW = 2.5

U_HOLD_FWD_MAX = 2.0
U_HOLD_RIGHT_MAX = 2.0
U_HOLD_YAW_MAX = 1.0

WHEEL_MAX_HOLD = 6.0
HOLD_SMOOTH_ALPHA = 0.35

B_FWD = np.array([-1, -1, -1, -1], dtype=np.float32)
B_RIGHT = np.array([-1, +1, -1, +1], dtype=np.float32)
B_YAW_CCW = np.array([+1, +1, -1, -1], dtype=np.float32)


# ============================================================
# Observation normalization constants
# ============================================================

# Alignment error is usually centimeters. Scaling it makes the RL signal easier.
REL_ERR_SCALE = 0.05       # 5 cm -> obs magnitude 1
BASE_ERR_SCALE = 0.05      # 5 cm -> obs magnitude 1
DIST_SCALE = 0.05          # 5 cm -> obs magnitude 1


# ============================================================
# Helpers
# ============================================================

def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def clip_vec(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(v, lo), hi)


def replay_buffer_path_for(model_path: Path) -> Path:
    return Path(str(model_path) + ".replay_buffer.pkl")


def latest_model_path_for(model_path: Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_latest.zip")


def default_checkpoint_dir_for(model_path: Path) -> Path:
    model_path = Path(model_path)
    return model_path.parent / f"{model_path.stem}_checkpoints"


def save_training_state(model, model_path: Path, save_replay_buffer: bool = True, label: str = "latest"):
    """
    Save both:
      1. model_path: stable path used by --model-path
      2. *_latest.zip: explicit latest checkpoint path

    If requested, save replay buffers for both names so SAC can resume with data.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    latest_path = latest_model_path_for(model_path)

    model.save(str(model_path))
    model.save(str(latest_path))

    print(f"\n[{label}] Saved model to: {model_path}")
    print(f"[{label}] Saved latest model to: {latest_path}")

    if save_replay_buffer:
        rb_main = replay_buffer_path_for(model_path)
        rb_latest = replay_buffer_path_for(latest_path)
        model.save_replay_buffer(str(rb_main))
        model.save_replay_buffer(str(rb_latest))
        print(f"[{label}] Saved replay buffer to: {rb_main}")
        print(f"[{label}] Saved latest replay buffer to: {rb_latest}")


class SafeCheckpointCallback(BaseCallback):
    """Save model/replay buffer every N timesteps during training."""

    def __init__(
        self,
        model_path: Path,
        save_freq: int = 10_000,
        save_replay_buffer: bool = True,
        checkpoint_dir: Path | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.model_path = Path(model_path)
        self.save_freq = int(save_freq)
        self.save_replay_buffer = bool(save_replay_buffer)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else default_checkpoint_dir_for(self.model_path)
        self.last_save = 0

    def _on_training_start(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.save_freq > 0:
            self.last_save = (self.num_timesteps // self.save_freq) * self.save_freq
        else:
            self.last_save = self.num_timesteps

    def _on_step(self) -> bool:
        if self.save_freq <= 0:
            return True

        if self.num_timesteps - self.last_save >= self.save_freq:
            step = self.num_timesteps
            ckpt_path = self.checkpoint_dir / f"{self.model_path.stem}_step_{step}.zip"

            self.model.save(str(ckpt_path))
            print(f"\n[checkpoint] Saved step checkpoint to: {ckpt_path}")

            # Also update the stable latest/base paths for easy resume.
            save_training_state(
                self.model,
                self.model_path,
                save_replay_buffer=self.save_replay_buffer,
                label=f"checkpoint step {step}",
            )

            self.last_save = step

        return True


def safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.linalg.norm(v) + eps)


# ============================================================
# Environment
# ============================================================

class YouBotGraspCorrectionCurriculumEnv(gym.Env):
    """
    Residual grasp-correction environment.

    The scene's manually successful open pre-grasp pose is treated as the nominal
    reference. At startup, the environment captures:
      - base reset orientation,
      - cube reset orientation,
      - arm/gripper learning-start joint values,
      - nominal cube-to-gripper relative vector in the EE frame.

    During training, the cube is shifted in xy. The policy changes only the first
    five arm joints using residual incremental actions. The objective is to bring
    the relative cube-to-gripper geometry back to the captured nominal geometry.
    Then a scripted close/lift macro is executed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: str = "easy",
        rand_min_xy: float | None = None,
        rand_max_xy: float | None = None,
        correction_steps: int = 12,
        action_repeat: int = 3,
        settle_steps: int = 8,
        action_scale_mult: float = 1.0,
        arm_margin_mult: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__()

        if rand_min_xy is None or rand_max_xy is None:
            if stage == "tiny":
                self.rand_min_xy = 0.000
                self.rand_max_xy = 0.008
            elif stage == "easy":
                self.rand_min_xy = 0.005
                self.rand_max_xy = 0.015
            elif stage == "medium":
                self.rand_min_xy = 0.010
                self.rand_max_xy = 0.025
            elif stage == "hard":
                self.rand_min_xy = 0.020
                self.rand_max_xy = 0.035
            else:
                raise ValueError("stage must be one of: tiny, easy, medium, hard")
        else:
            self.rand_min_xy = float(rand_min_xy)
            self.rand_max_xy = float(rand_max_xy)

        if self.rand_min_xy < 0 or self.rand_max_xy < 0:
            raise ValueError("rand_min_xy and rand_max_xy must be nonnegative")
        if self.rand_min_xy > self.rand_max_xy:
            raise ValueError("rand_min_xy must be <= rand_max_xy")

        self.correction_steps = int(correction_steps)
        self.action_repeat = int(action_repeat)
        self.settle_steps = int(settle_steps)

        self.client = RemoteAPIClient()
        self.sim = self.client.require("sim")

        self.base_h = self.sim.getObject(ROBOT_ROOT_PATH)
        self.body_h = self.sim.getObject(BODY_REF_PATH)
        self.cube_h = self.sim.getObject(CUBE_PATH)

        self.arm_joints = [self.sim.getObject(p) for p in ARM_PATHS]
        self.wheel_handles = [self.sim.getObject(p) for p in WHEEL_PATHS]

        self.ee_ref_h = self.arm_joints[4]
        self.g1_h = self.arm_joints[5]
        self.g2_h = self.arm_joints[6]

        # Capture nominal scene setup from current stopped scene.
        self.base_reset_pos = BASE_RESET_POS.copy()
        self.base_reset_ori = np.array(self.sim.getObjectOrientation(self.base_h, -1), dtype=np.float32)

        self.cube_reset_pos = CUBE_RESET_POS.copy()
        self.cube_reset_ori = np.array(self.sim.getObjectOrientation(self.cube_h, -1), dtype=np.float32)

        self.learn_start = np.array([self.sim.getJointPosition(h) for h in self.arm_joints], dtype=np.float32)
        self.learn_start_arm = self.learn_start[:5].copy()
        self.learn_start_gripper = self.learn_start[5:].copy()

        # Parked base target from /youBot_ref orientation in the captured scene.
        self.x_ref = float(self.base_reset_pos[0])
        self.y_ref = float(self.base_reset_pos[1])
        _, fwd_hat_ref = self._get_body_axes_from_ref(self.body_h)
        self.yaw_ref = math.atan2(float(fwd_hat_ref[1]), float(fwd_hat_ref[0]))

        self.current_arm = self.learn_start_arm.copy()
        self.open_gripper = self.learn_start_gripper.copy()
        self.closed_gripper = SCRIPTED_CLOSED_GRIPPER.copy()

        # Larger than v4 defaults, but still conservative.
        base_action_scale = np.array([0.012, 0.010, 0.010, 0.008, 0.006], dtype=np.float32)
        base_arm_margin = np.array([0.16, 0.12, 0.12, 0.10, 0.08], dtype=np.float32)
        self.action_scale = action_scale_mult * base_action_scale
        self.arm_margin = arm_margin_mult * base_arm_margin
        self.arm_lo = self.learn_start_arm - self.arm_margin
        self.arm_hi = self.learn_start_arm + self.arm_margin

        # Force the nominal reset pose once and capture the successful target geometry.
        # This is the key change: we do not assume target relative vector is zero.
        self._stop_if_needed()
        self._reset_scene_objects(offset_xy=np.zeros(2, dtype=np.float32), start_sim=False)
        self.target_rel_ee = self._get_rel_cube_to_gripper_ee_raw()

        # Observation, all approximately scaled to [-1, 1]:
        #   alignment error in EE frame = rel_ee - target_rel_ee, normalized (3)
        #   arm residual / margin (5)
        #   gripper residual around open pose (2)
        #   base drift normalized (3)
        #   previous xy alignment error normalized (1)
        #   remaining correction fraction (1)
        # Total = 15, same size as the previous file.
        obs_high = np.ones(15, dtype=np.float32) * 5.0
        obs_high[13] = 5.0
        obs_high[14] = 1.0
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        self.w_hold_prev = np.zeros(4, dtype=np.float32)
        self.cube_z_place = float(self.cube_reset_pos[2])
        self.last_cube_offset = np.zeros(2, dtype=np.float32)
        self.prev_align_xy = 0.0
        self.min_align_xy = np.inf
        self.episode_steps = 0
        self._is_started = False

        self.sim.setStepping(True)

        print("\nCaptured nominal manual setup:")
        print(f"  Base reset pos: {self.base_reset_pos.tolist()}")
        print(f"  Base reset ori: {self.base_reset_ori.tolist()}")
        print(f"  Cube reset pos: {self.cube_reset_pos.tolist()}")
        print(f"  Cube reset ori: {self.cube_reset_ori.tolist()}")
        print(f"  LEARN_START joints: {self.learn_start.tolist()}")
        print(f"  Target rel cube-to-gripper in EE frame: {self.target_rel_ee.tolist()}")
        print(f"  Rand xy range: [{self.rand_min_xy:.4f}, {self.rand_max_xy:.4f}] m")
        print(f"  Action scale: {self.action_scale.tolist()}")
        print(f"  Arm margin: {self.arm_margin.tolist()}\n")

        if seed is not None:
            self.reset(seed=seed)

    # ---------------------------
    # Low-level helpers
    # ---------------------------

    def _get_pose2d(self, h) -> Tuple[float, float, float]:
        p = self.sim.getObjectPosition(h, -1)
        o = self.sim.getObjectOrientation(h, -1)
        return float(p[0]), float(p[1]), float(o[2])

    def _get_body_axes_from_ref(self, h):
        m = self.sim.getObjectMatrix(h, -1)
        x_axis = np.array([m[0], m[4], m[8]], dtype=np.float32)[:2]
        y_axis = np.array([m[1], m[5], m[9]], dtype=np.float32)[:2]
        x_axis = x_axis / safe_norm(x_axis)
        y_axis = y_axis / safe_norm(y_axis)
        return x_axis, y_axis

    def _get_ee_axes(self):
        m = self.sim.getObjectMatrix(self.ee_ref_h, -1)
        x_axis = np.array([m[0], m[4], m[8]], dtype=np.float32)
        y_axis = np.array([m[1], m[5], m[9]], dtype=np.float32)
        z_axis = np.array([m[2], m[6], m[10]], dtype=np.float32)
        return (
            x_axis / safe_norm(x_axis),
            y_axis / safe_norm(y_axis),
            z_axis / safe_norm(z_axis),
        )

    def _get_gripper_midpoint(self) -> np.ndarray:
        p1 = np.array(self.sim.getObjectPosition(self.g1_h, -1), dtype=np.float32)
        p2 = np.array(self.sim.getObjectPosition(self.g2_h, -1), dtype=np.float32)
        return 0.5 * (p1 + p2)

    def _get_cube_pos(self) -> np.ndarray:
        return np.array(self.sim.getObjectPosition(self.cube_h, -1), dtype=np.float32)

    def _get_rel_cube_to_gripper_ee_raw(self) -> np.ndarray:
        cube = self._get_cube_pos()
        ee_mid = self._get_gripper_midpoint()
        rel_world = cube - ee_mid

        x_axis, y_axis, z_axis = self._get_ee_axes()
        return np.array([
            float(np.dot(rel_world, x_axis)),
            float(np.dot(rel_world, y_axis)),
            float(np.dot(rel_world, z_axis)),
        ], dtype=np.float32)

    def _get_alignment_error_raw(self) -> np.ndarray:
        return self._get_rel_cube_to_gripper_ee_raw() - self.target_rel_ee

    def _zero_wheels(self):
        for h in self.wheel_handles:
            self.sim.setJointTargetVelocity(h, 0.0)

    def _set_wheels(self, w: np.ndarray):
        for h, wi in zip(self.wheel_handles, w):
            self.sim.setJointTargetVelocity(h, float(wi))

    def _hold_base_pose_step(self):
        x, y, _ = self._get_pose2d(self.base_h)
        right_hat, fwd_hat = self._get_body_axes_from_ref(self.body_h)
        yaw_now = math.atan2(float(fwd_hat[1]), float(fwd_hat[0]))

        e_world = np.array([self.x_ref - x, self.y_ref - y], dtype=np.float32)
        e_right = float(np.dot(e_world, right_hat))
        e_fwd = float(np.dot(e_world, fwd_hat))
        e_yaw = wrap_pi(self.yaw_ref - yaw_now)

        u_right = clamp(K_HOLD_POS * e_right, -U_HOLD_RIGHT_MAX, U_HOLD_RIGHT_MAX)
        u_fwd = clamp(K_HOLD_POS * e_fwd, -U_HOLD_FWD_MAX, U_HOLD_FWD_MAX)
        u_yaw = clamp(K_HOLD_YAW * e_yaw, -U_HOLD_YAW_MAX, U_HOLD_YAW_MAX)

        w_unsat = u_fwd * B_FWD + u_right * B_RIGHT + u_yaw * B_YAW_CCW
        w_clip = np.clip(w_unsat, -WHEEL_MAX_HOLD, WHEEL_MAX_HOLD)
        w_cmd = (1.0 - HOLD_SMOOTH_ALPHA) * self.w_hold_prev + HOLD_SMOOTH_ALPHA * w_clip
        self.w_hold_prev = w_cmd.copy()
        self._set_wheels(w_cmd)

    def _set_targets(self, arm5: np.ndarray, gripper2: np.ndarray):
        target = np.concatenate([arm5, gripper2]).astype(np.float32)
        for h, v in zip(self.arm_joints, target):
            self.sim.setJointTargetPosition(h, float(v))

    def _set_joint_positions_immediate(self, arm5: np.ndarray, gripper2: np.ndarray):
        target = np.concatenate([arm5, gripper2]).astype(np.float32)
        for h, v in zip(self.arm_joints, target):
            self.sim.setJointPosition(h, float(v))
            self.sim.setJointTargetPosition(h, float(v))

    def _sample_cube_offset(self) -> np.ndarray:
        if self.rand_max_xy <= 0.0:
            return np.zeros(2, dtype=np.float32)

        for _ in range(10_000):
            dx = self.np_random.uniform(-self.rand_max_xy, self.rand_max_xy)
            dy = self.np_random.uniform(-self.rand_max_xy, self.rand_max_xy)
            r = math.hypot(float(dx), float(dy))
            if self.rand_min_xy <= r <= self.rand_max_xy:
                return np.array([dx, dy], dtype=np.float32)

        # Fallback should almost never happen.
        theta = self.np_random.uniform(-math.pi, math.pi)
        r = self.rand_max_xy
        return np.array([r * math.cos(theta), r * math.sin(theta)], dtype=np.float32)

    def _stop_if_needed(self):
        try:
            self.sim.stopSimulation()
        except Exception:
            pass
        for _ in range(100):
            try:
                state = self.sim.getSimulationState()
                if state == self.sim.simulation_stopped:
                    break
            except Exception:
                break
            time.sleep(0.01)
        self._is_started = False

    def _start(self):
        self.sim.startSimulation()
        self._is_started = True

    def _reset_scene_objects(self, offset_xy: np.ndarray, start_sim: bool):
        # Exact base reset every episode.
        self.sim.setObjectPosition(
            self.base_h, -1,
            [float(self.base_reset_pos[0]), float(self.base_reset_pos[1]), float(self.base_reset_pos[2])]
        )
        self.sim.setObjectOrientation(
            self.base_h, -1,
            [float(self.base_reset_ori[0]), float(self.base_reset_ori[1]), float(self.base_reset_ori[2])]
        )

        # Exact cube reset plus random xy offset.
        cube_pos = self.cube_reset_pos.copy()
        cube_pos[0] += float(offset_xy[0])
        cube_pos[1] += float(offset_xy[1])

        self.sim.setObjectPosition(
            self.cube_h, -1,
            [float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])]
        )
        self.sim.setObjectOrientation(
            self.cube_h, -1,
            [float(self.cube_reset_ori[0]), float(self.cube_reset_ori[1]), float(self.cube_reset_ori[2])]
        )

        # Exact arm/gripper learning-start reset every episode.
        self._set_joint_positions_immediate(self.learn_start_arm, self.open_gripper)
        self._zero_wheels()

        if start_sim:
            self._start()

    # ---------------------------
    # Scripted macro
    # ---------------------------

    def _interp_motion(
        self,
        arm_from: np.ndarray,
        grip_from: np.ndarray,
        arm_to: np.ndarray,
        grip_to: np.ndarray,
        n_steps: int,
    ):
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            arm = (1.0 - alpha) * arm_from + alpha * arm_to
            grip = (1.0 - alpha) * grip_from + alpha * grip_to
            self._hold_base_pose_step()
            self._set_targets(arm.astype(np.float32), grip.astype(np.float32))
            self.sim.step()

    def _attempt_grasp_macro(self) -> Tuple[bool, float]:
        arm_start = self.current_arm.copy()
        grip_open = self.open_gripper.copy()
        grip_closed = self.closed_gripper.copy()

        # Close.
        self._interp_motion(arm_start, grip_open, arm_start, grip_closed, n_steps=8)

        # Settle after closing.
        for _ in range(4):
            self._hold_base_pose_step()
            self._set_targets(arm_start, grip_closed)
            self.sim.step()

        # Relative lift.
        arm_lift = clip_vec(arm_start + LIFT_DELTA_ARM, self.arm_lo, self.arm_hi)
        self._interp_motion(arm_start, grip_closed, arm_lift, grip_closed, n_steps=10)

        z_thresh = self.cube_z_place + 0.015
        success_counter = 0
        lift_amount = 0.0

        for _ in range(8):
            self._hold_base_pose_step()
            self._set_targets(arm_lift, grip_closed)
            self.sim.step()

            z = float(self._get_cube_pos()[2])
            lift_amount = z - self.cube_z_place
            if z > z_thresh:
                success_counter += 1
            else:
                success_counter = 0

        success = success_counter >= 3
        return bool(success), float(lift_amount)

    # ---------------------------
    # Observation / reward
    # ---------------------------

    def _get_obs(self) -> np.ndarray:
        err_ee = self._get_alignment_error_raw()
        err_ee_scaled = err_ee / REL_ERR_SCALE

        actual_joints = np.array([self.sim.getJointPosition(h) for h in self.arm_joints], dtype=np.float32)
        arm_resid_scaled = (actual_joints[:5] - self.learn_start_arm) / np.maximum(self.arm_margin, 1e-6)
        grip_resid_scaled = actual_joints[5:] - self.open_gripper

        x, y, _ = self._get_pose2d(self.base_h)
        _, fwd_hat = self._get_body_axes_from_ref(self.body_h)
        yaw_now = math.atan2(float(fwd_hat[1]), float(fwd_hat[0]))

        base_dx = (x - self.x_ref) / BASE_ERR_SCALE
        base_dy = (y - self.y_ref) / BASE_ERR_SCALE
        base_dyaw = wrap_pi(yaw_now - self.yaw_ref) / math.pi

        align_xy = float(np.linalg.norm(err_ee[:2]))
        remain = float((self.correction_steps - self.episode_steps) / max(self.correction_steps, 1))

        obs = np.array([
            err_ee_scaled[0], err_ee_scaled[1], err_ee_scaled[2],
            arm_resid_scaled[0], arm_resid_scaled[1], arm_resid_scaled[2], arm_resid_scaled[3], arm_resid_scaled[4],
            grip_resid_scaled[0], grip_resid_scaled[1],
            base_dx, base_dy, base_dyaw,
            align_xy / DIST_SCALE,
            remain,
        ], dtype=np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32)

    def _alignment_from_obs(self, obs: np.ndarray) -> Tuple[np.ndarray, float, float]:
        err_ee = obs[:3].astype(np.float32) * REL_ERR_SCALE
        align_xy = float(np.linalg.norm(err_ee[:2]))
        align_z = abs(float(err_ee[2]))
        return err_ee, align_xy, align_z

    def _compute_dense_reward(self, obs: np.ndarray, action: np.ndarray) -> float:
        err_ee, align_xy, align_z = self._alignment_from_obs(obs)
        progress = self.prev_align_xy - align_xy

        # Smooth shaped reward toward the captured nominal grasp geometry.
        reward = 0.0
        reward += 20.0 * progress
        reward += -3.0 * align_xy
        reward += -2.0 * align_z

        # Extra reward for entering useful pre-grasp neighborhoods.
        if align_xy < 0.020 and align_z < 0.020:
            reward += 0.5
        if align_xy < 0.012 and align_z < 0.015:
            reward += 1.5
        if align_xy < 0.007 and align_z < 0.010:
            reward += 3.0

        # Small bonus for improving best-so-far alignment.
        if align_xy < self.min_align_xy:
            reward += 2.0 * (self.min_align_xy - align_xy if np.isfinite(self.min_align_xy) else 0.0)

        # Keep the residual motion smooth.
        reward += -0.005 * float(np.sum(np.square(action)))
        reward += -0.01
        return float(reward)

    # ---------------------------
    # Gym API
    # ---------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._stop_if_needed()

        self.w_hold_prev = np.zeros(4, dtype=np.float32)
        self.episode_steps = 0
        self.current_arm = self.learn_start_arm.copy()
        self.min_align_xy = np.inf

        offset = self._sample_cube_offset()
        self.last_cube_offset = offset.copy()

        self._reset_scene_objects(offset_xy=offset, start_sim=True)

        for _ in range(self.settle_steps):
            self._hold_base_pose_step()
            self._set_targets(self.current_arm, self.open_gripper)
            self.sim.step()

        self.cube_z_place = float(self._get_cube_pos()[2])
        obs = self._get_obs()
        _, align_xy, _ = self._alignment_from_obs(obs)
        self.prev_align_xy = align_xy
        self.min_align_xy = align_xy

        info = {
            "cube_dx": float(offset[0]),
            "cube_dy": float(offset[1]),
            "cube_z_place": self.cube_z_place,
            "initial_align_xy": float(align_xy),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -1.0, 1.0)

        self.current_arm = self.current_arm + self.action_scale * action
        self.current_arm = clip_vec(self.current_arm, self.arm_lo, self.arm_hi)

        for _ in range(self.action_repeat):
            self._hold_base_pose_step()
            self._set_targets(self.current_arm, self.open_gripper)
            self.sim.step()

        self.episode_steps += 1
        obs = self._get_obs()
        reward = self._compute_dense_reward(obs, action)

        _, align_xy, align_z = self._alignment_from_obs(obs)
        self.prev_align_xy = align_xy
        self.min_align_xy = min(self.min_align_xy, align_xy)

        if self.episode_steps < self.correction_steps:
            terminated = False
            truncated = False
            info = {
                "success": False,
                "triggered": False,
                "episode_steps": self.episode_steps,
                "cube_dx": float(self.last_cube_offset[0]),
                "cube_dy": float(self.last_cube_offset[1]),
                "lift_amount": 0.0,
                "align_xy": float(align_xy),
                "align_z": float(align_z),
                "min_align_xy": float(self.min_align_xy),
            }
            return obs, float(reward), terminated, truncated, info

        # Save pre-macro alignment before the cube is moved by the lift attempt.
        pre_macro_align_xy = float(align_xy)
        pre_macro_align_z = float(align_z)

        success, lift_amount = self._attempt_grasp_macro()
        obs = self._get_obs()

        # Terminal reward: binary success + partial lift + final pregrasp quality.
        reward += 800.0 * max(0.0, lift_amount)
        reward += -5.0 * pre_macro_align_xy
        reward += -3.0 * pre_macro_align_z

        if success:
            reward += 60.0
        else:
            reward += -2.0

        terminated = True
        truncated = False
        info = {
            "success": bool(success),
            "triggered": True,
            "episode_steps": self.episode_steps,
            "cube_dx": float(self.last_cube_offset[0]),
            "cube_dy": float(self.last_cube_offset[1]),
            "lift_amount": float(lift_amount),
            "align_xy": float(pre_macro_align_xy),
            "align_z": float(pre_macro_align_z),
            "min_align_xy": float(self.min_align_xy),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self._stop_if_needed()
        except Exception:
            pass


# ============================================================
# Train / eval helpers
# ============================================================

def make_env(
    stage="tiny",
    rand_min_xy=None,
    rand_max_xy=None,
    correction_steps=12,
    action_repeat=3,
    settle_steps=8,
    action_scale_mult=1.0,
    arm_margin_mult=1.0,
    seed=None,
):
    return YouBotGraspCorrectionCurriculumEnv(
        stage=stage,
        rand_min_xy=rand_min_xy,
        rand_max_xy=rand_max_xy,
        correction_steps=correction_steps,
        action_repeat=action_repeat,
        settle_steps=settle_steps,
        action_scale_mult=action_scale_mult,
        arm_margin_mult=arm_margin_mult,
        seed=seed,
    )


def evaluate(model, env: gym.Env, episodes: int = 20):
    successes = 0
    rewards = []
    lifts = []
    final_aligns = []

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        ep_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

        success = bool(info.get("success", False))
        lift = float(info.get("lift_amount", 0.0))
        align_xy = float(info.get("align_xy", np.nan))

        rewards.append(ep_reward)
        lifts.append(lift)
        final_aligns.append(align_xy)
        successes += int(success)

        print(
            f"Eval ep {ep + 1:02d}/{episodes}: "
            f"success={success}  "
            f"lift={lift:.4f}  "
            f"align_xy={align_xy:.4f}  "
            f"offset=({info.get('cube_dx', 0.0):+.3f}, {info.get('cube_dy', 0.0):+.3f})"
        )

    print("\nEvaluation summary")
    print(f"  successes: {successes}/{episodes}")
    print(f"  success rate: {100.0 * successes / max(episodes, 1):.1f}%")
    print(f"  mean episode reward: {np.mean(rewards):.2f}")
    print(f"  mean lift amount: {np.mean(lifts):.4f} m")
    print(f"  mean final align_xy: {np.nanmean(final_aligns):.4f} m")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=20)

    parser.add_argument("--stage", choices=["tiny", "easy", "medium", "hard"], default="tiny")
    parser.add_argument("--rand-min", type=float, default=None)
    parser.add_argument("--rand-max", type=float, default=None)

    parser.add_argument("--model-path", type=str, default="models/sac_youbot_grasp_correction_v5_safe.zip")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-model", type=str, default=None)
    parser.add_argument("--replay-buffer-path", type=str, default=None)
    parser.add_argument("--save-replay-buffer", dest="save_replay_buffer", action="store_true", default=True)
    parser.add_argument("--no-save-replay-buffer", dest="save_replay_buffer", action="store_false")

    parser.add_argument("--correction-steps", type=int, default=12)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--action-scale-mult", type=float, default=1.0)
    parser.add_argument("--arm-margin-mult", type=float, default=1.0)

    # SAC hyperparameters
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--ent-coef", type=str, default="auto_0.1")
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256, 256])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--tensorboard-log", type=str, default=None)

    # Safe saving / checkpointing
    parser.add_argument("--checkpoint-freq", type=int, default=10_000,
                        help="Save model and replay buffer every N timesteps. Use 0 to disable periodic checkpoints.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Optional folder for numbered step checkpoints.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(
        stage=args.stage,
        rand_min_xy=args.rand_min,
        rand_max_xy=args.rand_max,
        correction_steps=args.correction_steps,
        action_repeat=args.action_repeat,
        settle_steps=args.settle_steps,
        action_scale_mult=args.action_scale_mult,
        arm_margin_mult=args.arm_margin_mult,
        seed=args.seed,
    )

    if args.check_env:
        check_env(env, warn=True, skip_render_check=True)

    env = Monitor(env)

    if args.mode == "train":
        if args.resume:
            # Resume preference:
            #   1. --resume-model if explicitly provided
            #   2. *_latest.zip if it exists
            #   3. --model-path
            if args.resume_model is not None:
                load_path = Path(args.resume_model)
            else:
                latest_path = latest_model_path_for(model_path)
                load_path = latest_path if latest_path.exists() else model_path

            if not load_path.exists():
                raise FileNotFoundError(f"Resume requested, but model not found: {load_path}")

            print(f"Resuming from model: {load_path}")
            model = SAC.load(str(load_path), env=env, device=args.device)

            # Replay-buffer preference:
            #   1. --replay-buffer-path if explicitly provided
            #   2. replay buffer matching loaded model
            #   3. replay buffer matching --model-path
            candidate_rb_paths = []
            if args.replay_buffer_path:
                candidate_rb_paths.append(Path(args.replay_buffer_path))
            candidate_rb_paths.append(replay_buffer_path_for(load_path))
            candidate_rb_paths.append(replay_buffer_path_for(model_path))

            rb_path = next((p for p in candidate_rb_paths if p.exists()), None)
            if rb_path is not None:
                print(f"Loading replay buffer: {rb_path}")
                model.load_replay_buffer(str(rb_path))
            else:
                print("No replay buffer file found. Resuming weights only.")
        else:
            print("Starting fresh SAC model")
            print("SAC settings:")
            print(f"  learning_rate={args.learning_rate}")
            print(f"  buffer_size={args.buffer_size}")
            print(f"  learning_starts={args.learning_starts}")
            print(f"  batch_size={args.batch_size}")
            print(f"  tau={args.tau}")
            print(f"  gamma={args.gamma}")
            print(f"  train_freq={args.train_freq}")
            print(f"  gradient_steps={args.gradient_steps}")
            print(f"  ent_coef={args.ent_coef}")
            print(f"  net_arch={args.net_arch}")

            model = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=args.tensorboard_log,
                learning_rate=args.learning_rate,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                batch_size=args.batch_size,
                tau=args.tau,
                gamma=args.gamma,
                train_freq=args.train_freq,
                gradient_steps=args.gradient_steps,
                ent_coef=args.ent_coef,
                policy_kwargs=dict(net_arch=args.net_arch),
                seed=args.seed,
                device=args.device,
            )

        checkpoint_callback = SafeCheckpointCallback(
            model_path=model_path,
            save_freq=args.checkpoint_freq,
            save_replay_buffer=args.save_replay_buffer,
            checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
            verbose=1,
        )

        interrupted = False
        try:
            model.learn(
                total_timesteps=args.timesteps,
                progress_bar=args.progress_bar,
                reset_num_timesteps=not args.resume,
                callback=checkpoint_callback,
            )
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted by user. Saving current training state before exit...")
        finally:
            label = "interrupted" if interrupted else "final"
            save_training_state(
                model,
                model_path,
                save_replay_buffer=args.save_replay_buffer,
                label=label,
            )

    elif args.mode == "eval":
        load_path = Path(args.resume_model) if args.resume_model is not None else model_path
        if not load_path.exists():
            raise FileNotFoundError(f"Evaluation model not found: {load_path}")

        model = SAC.load(str(load_path), env=env, device=args.device)
        evaluate(model, env, episodes=args.episodes)

    env.close()


if __name__ == "__main__":
    main()
