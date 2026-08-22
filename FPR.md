# Final Review Plan

## Core Implementation

- [ ] Experience Replay Stack
- [ ] Gazebo perception integration
- [ ] Full Human-in-the-Loop joystick testing
- [ ] Increase the speed of the drone
- [ ] Improve telemetry and experiment logging

## Navigation Environment

### PyBullet — Hollow Cylinder Environment

- [ ] Create a hollow cylinder
- [ ] Place the drone inside the hollow cylinder
- [ ] Add multiple static obstacles
- [ ] Add multiple dynamic obstacles
- [ ] Implement obstacle avoidance
- [ ] Record how the drone navigates around obstacles
- [ ] Visualize the drone trajectory and obstacle movement
- [ ] Increase the speed of the drone
- [ ] Test Human-in-the-Loop navigation

### Gazebo — Virtual Box Environment

- [ ] Create a virtual-box environment
- [ ] Place the drone inside the virtual box
- [ ] Integrate perception
- [ ] Add multiple static obstacles
- [ ] Add multiple dynamic obstacles
- [ ] Implement real-time obstacle detection
- [ ] Implement dynamic obstacle avoidance
- [ ] Increase the speed of the drone
- [ ] Test Human-in-the-Loop joystick control

### MuJoCo

- [ ] Create the navigation environment
- [ ] Add multiple obstacles
- [ ] Add multiple dynamic obstacles
- [ ] Implement obstacle avoidance
- [ ] Test high-speed navigation
- [ ] Implement multi-drone navigation
- [ ] Evaluate controller performance

### ArduPilot SITL

- [ ] Integrate the AI Drone Co-Pilot with ArduPilot SITL
- [ ] Implement Human-in-the-Loop joystick control
- [ ] Test autonomous navigation
- [ ] Test shared human-AI control
- [ ] Add multiple obstacles
- [ ] Test obstacle avoidance
- [ ] Test high-speed navigation
- [ ] Evaluate telemetry and controller performance

## Dynamic Obstacle Navigation

- [ ] Add multiple dynamic obstacles
- [ ] Assign different obstacle trajectories
- [ ] Assign different obstacle speeds
- [ ] Detect moving obstacles
- [ ] Continuously update the drone's navigation path
- [ ] Implement real-time obstacle avoidance
- [ ] Record drone trajectory
- [ ] Record obstacle trajectories
- [ ] Record minimum obstacle clearance
- [ ] Record collision events
- [ ] Analyze how the drone changes its path when avoiding obstacles

## High-Speed Navigation

- [ ] Establish baseline drone speed
- [ ] Test low-speed navigation
- [ ] Test medium-speed navigation
- [ ] Test high-speed navigation
- [ ] Test high-speed obstacle avoidance
- [ ] Measure trajectory tracking error
- [ ] Measure navigation time
- [ ] Measure minimum obstacle clearance
- [ ] Compare performance at different speeds

## Multi-Drone Navigation

- [ ] Implement two-drone navigation
- [ ] Implement three-drone navigation
- [ ] Implement four or more drones where computationally feasible
- [ ] Assign individual start positions
- [ ] Assign individual target positions
- [ ] Prevent drone-to-drone collisions
- [ ] Prevent drone-to-obstacle collisions
- [ ] Test different drone speeds
- [ ] Record trajectories of all drones
- [ ] Measure minimum distance between drones
- [ ] Evaluate scalability

## Human-in-the-Loop Testing

### Human Only

```text
Joystick → Drone
