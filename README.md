# PRM-RL Mobile Manipulation in CoppeliaSim

Hybrid classical planning and reinforcement learning pipeline for a simulated KUKA youBot mobile manipulator. The system combines PRM-based global navigation, path-frame feedback tracking, and SAC-based residual grasp correction for a complete pick-carry-place task.


## Overview

This project demonstrates a modular mobile manipulation pipeline:

1. Plan collision-aware base motion using PRM.
2. Track the path using a tuned path-frame mecanum controller.
3. Dock near the object.
4. Apply a learned SAC residual policy for local grasp correction.
5. Close, lift, carry, and place the object.

The key idea is not to train the full task end-to-end. Instead, classical planning and control solve the structured navigation problem, while reinforcement learning is used only for the locally sensitive grasp-alignment phase.

## Results

- Full offset range: 96% success over 100 trials.
- Edge-case range [0.030, 0.035] m: 94% success over 100 trials.
- Integrated qualitative demonstration: complete pick-carry-place execution.

## Repository Structure

```text
scripts/
  run_integrated_pick_place.py       # Full pick-carry-place demo
  run_prm_tracking_demo.py           # PRM + path tracking only
  tune_base_controller.py            # Synthetic path-tracking tuning
  train_sac_grasp_correction.py      # SAC residual grasp correction training

scenes/
  youbot_prm_rl_scene.ttt            # CoppeliaSim scene

models/
  README.md                          # Instructions to download or place trained model


media/
  demo_short.gif
  thumbnail.png
