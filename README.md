# AI-CoPilot: Data-Driven Optimal Shared Control for UAVs

**A Human-in-the-Loop (HiTL) Safety and Decision Support System for Autonomous UAVs**

This repository implements a novel data-driven optimal shared control mechanism for Unmanned Aerial Vehicles (UAVs). Instead of "hard-switching" between human and autonomous control (which causes instability), this AI Co-Pilot smoothly blends the human pilot's joystick input with an optimal autonomous safety controller based on real-time intention alignment.

## Base Literature
This project is adapted from the state-of-the-art 2025 paper:

> **Tan, J., Xue, S., Guo, Z., Li, H., Cao, H., & Chen, B. (2025).**
> "Data-driven optimal shared control of unmanned aerial vehicles."
> **Neurocomputing**, 622, 129428. DOI: 10.1016/j.neucom.2025.129428

While the base paper utilizes MATLAB/Simulink and RflySim, **this project adapts the mathematical framework to an open-source, multi-simulator Python/ArduPilot stack** for real-world flight controller validation.

---

## Core Mathematical Framework

The system utilizes a **Koopman Operator** to predict nonlinear UAV dynamics from human maneuver data, and an **Actor-Critic Reinforcement Learning** network to approximate the optimal safe controller (U*) online.

### 1. Smooth Shared Control Mechanism (Eq. 21 from Base Paper)
The final control input sent to the UAV is a smooth blend of the AI's optimal input (U*) and the Human's input (U_h):

    U_shared = U* + alpha * U_h

### 2. Adaptive Authority Allocation (Eq. 22)
The authority parameter alpha in [0, 1] is determined by the angle eta between the human's input vector and the AI's safe input vector:

- **alpha = 0 (Full Autonomy):** if eta >= beta_1 (2*pi/3). The human is steering dangerously; the AI takes over to prevent a crash.
- **alpha = 1 (Full Human):** if eta <= beta_2 (pi/2). The human's intention aligns with the safe path; the AI yields control.
- **0 < alpha < 1 (Smooth Transition):** Linear interpolation between beta_1 and beta_2 to prevent jerky control switching.

### 3. Koopman Operator / EDMD Dynamics Learning (Eq. 12)
The linearized system dynamics matrices A and B are learned from human maneuver data:

    [A, B] = Y_kappa * [X_kappa, U_kappa]^dagger

### 4. Actor-Critic Online Optimal Control (Eq. 26)
The optimal control input is approximated online:

    U_hat(X) = -mu * tanh(R^-1 * G_hat^T * W_a^T * phi_a(X) / (2*mu))

---

## Multi-Simulator Architecture (Team of 4)

To ensure aerospace-grade reliability, the project is validated across four distinct simulation environments:

| Simulator | Team Role | Purpose in Project |
| :--- | :--- | :--- |
| **ArduPilot SITL** | **Lead Integration (My Part)** | Industry-standard flight controller. MAVLink telemetry extraction, waypoint navigation, and real-world autopilot logic validation. |
| **Gazebo / ROS** | Sensor Simulation | Renders 3D environments, LiDAR, and depth cameras to feed the Koopman state observer. |
| **MuJoCo** | Dynamics & Physics | High-fidelity contact dynamics, wind disturbance modeling, and collision physics. |
| **Gym-PyBullet** | RL Training | Lightweight environment for initial Actor-Critic neural network training and baseline policy testing. |

---

## Repository Structure

    CD6_AI-drone-copilot/
    |-- docs/                      # Base paper summary, math proofs, and Review 1 slides
    |-- src/
    |   |-- ardupilot_bridge/      # MAVLink telemetry & state extraction (Python/pymavlink)
    |   |-- safety/                # Shared control authority allocation logic (Eq. 21-22)
    |   |-- koopman/               # EDMD dynamics prediction matrices (A, B)
    |   `-- vision/                # YOLOv7 / ZoeDepth obstacle & depth estimation
    |-- simulation/
    |   |-- ardupilot_sitl/        # Mission Planner configs and SITL startup scripts
    |   |-- gazebo/                # ROS/Gazebo world files
    |   |-- mujoco/                # MuJoCo XML drone models
    |   `-- pybullet/              # Gym environment wrappers
    |-- tests/                     # Unit tests for math and telemetry parsing
    `-- requirements.txt           # Python dependencies

---

## Phase 1 Progress (Review 1 Milestone)

- [x] **Literature Review:** Selected and analyzed Tan et al. (Neurocomputing 2025).
- [x] **Mathematical Formulation:** Extracted Koopman EDMD and Shared Control equations.
- [x] **ArduPilot SITL Setup:** Successfully initialized Software-In-The-Loop via Mission Planner.
- [x] **Telemetry Pipeline:** Built Python pymavlink bridge to extract live state vectors X = [Phi, Phi_dot] from ArduPilot.
- [ ] **Phase 2:** Implement Actor-Critic NN weight update laws (Eq. 29-30) in Python.
- [ ] **Phase 3:** Integrate Gazebo camera feed for Koopman state observation.
- [ ] **Phase 4:** Full HiTL (Human-in-the-Loop) joystick testing with authority blending.

---

## Getting Started (ArduPilot Telemetry Bridge)

1. Install dependencies:

       pip install -r requirements.txt

2. Start ArduPilot SITL via Mission Planner (UDP port 14550).
3. Run the telemetry reader to extract live UAV states:

       python src/ardupilot_bridge/telemetry_reader.py

4. Test the shared control authority allocation math:

       python src/safety/shared_control.py

## License
This project is developed for academic research and evaluation purposes.
