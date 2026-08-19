import os
import time
import math
import numpy as np
import pybullet as p
import cv2

from ultralytics import YOLO

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl


# ============================================================
# STAGE 25
# HUMAN-IN-THE-LOOP REACTIVE OBSTACLE AVOIDANCE
#
# MAIN IDEA
# ------------------------------------------------------------
# The drone is initially controlled by the human.
#
# W = forward
# S = backward
# A = left
# D = right
# R = up
# F = down
#
# The environment contains multiple obstacles.
#
# The pilot can manually navigate around them.
#
# HOWEVER:
#
# If the pilot continues toward an obstacle and enters the
# safety region, autonomous safety takes over.
#
# The autonomous controller:
#
#   1. Stops forward motion.
#   2. Determines whether left or right is safer.
#   3. Moves sideways.
#   4. Passes the obstacle.
#   5. Returns toward the original path.
#   6. Stabilizes.
#   7. Gives control back to the pilot.
#
# If another obstacle appears later, the same process repeats.
#
# IMPORTANT:
# ------------------------------------------------------------
# Physical PyBullet obstacle geometry is the PRIMARY safety
# detector.
#
# YOLO is kept as a secondary perception demonstration.
#
# YOLO alone should NOT be trusted for the synthetic PyBullet
# objects because the default YOLO model was not trained on
# these exact objects.
#
# ============================================================


print()
print("======================================================")
print("STAGE 25 - HUMAN-IN-THE-LOOP REACTIVE AVOIDANCE")
print("======================================================")
print()


# ============================================================
# SETTINGS
# ============================================================

DRONE_MODEL = DroneModel("cf2x")

INITIAL_POSITION = np.array(
    [0.0, 0.0, 1.20],
    dtype=float
)

# Mission endpoint.
FINAL_TARGET = np.array(
    [12.0, 0.0, 1.20],
    dtype=float
)


# ============================================================
# SIMULATION
# ============================================================

SIMULATION_FREQ = 240
CONTROL_FREQ = 48

MAX_DURATION = 240


# ============================================================
# CAMERA
# ============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_FOV = 70
CAMERA_NEAR = 0.02
CAMERA_FAR = 15.0

CAMERA_INTERVAL = 8


# ============================================================
# YOLO
# ============================================================

YOLO_ENABLED = True

YOLO_MODEL_NAME = "yolo11n.pt"

YOLO_CONFIDENCE = 0.35


# ============================================================
# FLIGHT LIMITS
# ============================================================

MIN_ALTITUDE = 0.75
MAX_ALTITUDE = 2.80

NORMAL_ALTITUDE = 1.20

AVOIDANCE_ALTITUDE = 1.80


# ============================================================
# MANUAL CONTROL
# ============================================================

# Position target movement speed.
MANUAL_SPEED = 0.55

VERTICAL_SPEED = 0.35

# Prevent the PID controller from chasing a target that is
# excessively far away.
MAX_MANUAL_TARGET_ERROR = 0.45

MANUAL_MIN_X = -1.0
MANUAL_MAX_X = 13.0

MANUAL_MIN_Y = -4.5
MANUAL_MAX_Y = 4.5


# ============================================================
# SAFETY
# ============================================================

# Obstacle must be in front of the drone.
SAFETY_LOOKAHEAD = 1.35

# Strong emergency region.
EMERGENCY_DISTANCE = 0.55

# Horizontal clearance required around obstacle.
DRONE_SAFETY_RADIUS = 0.38

# Lateral overlap threshold.
LATERAL_OVERLAP_MARGIN = 0.15

# Minimum obstacle clearance.
MIN_SIDE_CLEARANCE = 0.65

# Additional distance beyond obstacle before returning.
PASS_MARGIN = 0.70

# Waypoint tolerance.
WAYPOINT_TOLERANCE = 0.15

# ============================================================
# AUTONOMOUS TARGET SAFETY LIMITS
# ============================================================
# These limits are deliberately small.  The avoidance planner may
# calculate a waypoint several metres away, but the PID controller is
# NEVER allowed to jump directly to it in one control cycle.
AUTO_MAX_TARGET_STEP = 0.07
AUTO_MAX_X_STEP = 0.045
AUTO_MAX_Y_STEP = 0.065
AUTO_MAX_Z_STEP = 0.045

# A short brake/settling period is used before lateral motion.
BRAKE_TIME = 0.45
BRAKE_SPEED_THRESHOLD = 0.18

# Require the drone to be clearly past the blocking obstacle before
# lateral path restoration is allowed.
CLEAR_OBSTACLE_MARGIN = 0.55

# Require a path segment to be safe before accepting a waypoint.
PATH_SAFETY_SAMPLES = 40


# ============================================================
# MANUAL HANDOFF
# ============================================================

# Time spent stabilizing before manual control resumes.
HANDOFF_TIME = 1.25


# ============================================================
# ENVIRONMENT
# ============================================================

ENV_X_MIN = -2.0
ENV_X_MAX = 14.0

ENV_Y_MIN = -5.0
ENV_Y_MAX = 5.0


# ============================================================
# RESULTS
# ============================================================

RESULT_DIR = os.path.join(
    "results",
    "stage25_human_in_loop_reactive_avoidance"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# RANDOMNESS
# ============================================================

# Fixed environment.
#
# This is intentional.
#
# A fixed environment makes debugging much easier than the
# Stage 24 random regeneration system.
#
# Once this works reliably, randomization can be added later.
# ============================================================

np.random.seed(25)


# ============================================================
# ENVIRONMENT OBJECT STORAGE
# ============================================================

environment_objects = []

obstacle_boxes = []


# ============================================================
# PYBULLET ENVIRONMENT
# ============================================================

print("Creating PyBullet environment...")
print()


env = CtrlAviary(
    drone_model=DRONE_MODEL,
    num_drones=1,
    initial_xyzs=np.array([
        INITIAL_POSITION
    ]),
    initial_rpys=np.array([
        [0.0, 0.0, 0.0]
    ]),
    physics=Physics.PYB,
    pyb_freq=SIMULATION_FREQ,
    ctrl_freq=CONTROL_FREQ,
    gui=True,
    record=False,
    obstacles=False
)


CLIENT = env.getPyBulletClient()


# ============================================================
# GROUND
# ============================================================

ground_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[
        8.5,
        5.5,
        0.05
    ],
    physicsClientId=CLIENT
)

ground_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[
        8.5,
        5.5,
        0.05
    ],
    rgbaColor=[
        0.30,
        0.30,
        0.30,
        1.0
    ],
    physicsClientId=CLIENT
)

ground_id = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=ground_collision,
    baseVisualShapeIndex=ground_visual,
    basePosition=[
        6.0,
        0.0,
        -0.05
    ],
    physicsClientId=CLIENT
)

environment_objects.append(
    ground_id
)


# ============================================================
# CONTROLLER
# ============================================================

controller = DSLPIDControl(
    drone_model=DRONE_MODEL
)


# ============================================================
# ADD BOX
# ============================================================

def add_box(
    center,
    half_extents,
    rgba,
    obstacle_type
):

    center = np.asarray(
        center,
        dtype=float
    )

    half_extents = np.asarray(
        half_extents,
        dtype=float
    )

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents.tolist(),
        physicsClientId=CLIENT
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents.tolist(),
        rgbaColor=rgba,
        physicsClientId=CLIENT
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=center.tolist(),
        physicsClientId=CLIENT
    )

    environment_objects.append(
        body_id
    )

    obstacle_boxes.append({
        "min": center - half_extents,
        "max": center + half_extents,
        "type": obstacle_type,
        "body_id": body_id
    })

    return body_id


# ============================================================
# ADD CYLINDER
# ============================================================

def add_cylinder(
    center,
    radius,
    height,
    rgba,
    obstacle_type
):

    center = np.asarray(
        center,
        dtype=float
    )

    collision_shape = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height,
        physicsClientId=CLIENT
    )

    visual_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=rgba,
        physicsClientId=CLIENT
    )

    body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=center.tolist(),
        physicsClientId=CLIENT
    )

    environment_objects.append(
        body_id
    )

    obstacle_boxes.append({
        "min": center - np.array([
            radius,
            radius,
            height / 2.0
        ]),
        "max": center + np.array([
            radius,
            radius,
            height / 2.0
        ]),
        "type": obstacle_type,
        "body_id": body_id
    })

    return body_id


# ============================================================
# ADD TREE
# ============================================================

def add_tree(
    x,
    y,
    trunk_height=1.3,
    canopy_radius=0.55
):

    # Trunk
    add_cylinder(
        center=[
            x,
            y,
            trunk_height / 2.0
        ],
        radius=0.16,
        height=trunk_height,
        rgba=[
            0.35,
            0.16,
            0.05,
            1.0
        ],
        obstacle_type="tree_trunk"
    )

    # Canopy represented by a box for conservative collision
    # planning and reliable geometric safety checks.
    add_box(
        center=[
            x,
            y,
            trunk_height + canopy_radius * 0.55
        ],
        half_extents=[
            canopy_radius,
            canopy_radius,
            canopy_radius
        ],
        rgba=[
            0.05,
            0.45,
            0.08,
            1.0
        ],
        obstacle_type="tree_canopy"
    )


# ============================================================
# ADD POLE
# ============================================================

def add_pole(
    x,
    y
):

    add_cylinder(
        center=[
            x,
            y,
            1.5
        ],
        radius=0.10,
        height=3.0,
        rgba=[
            0.15,
            0.15,
            0.15,
            1.0
        ],
        obstacle_type="pole"
    )


# ============================================================
# CREATE SIMPLE 4-BLOCK ENVIRONMENT
# ============================================================

print("Creating simple 4-block obstacle environment...")
print()

# Four clean block obstacles.
# (x, y, width, depth, height)
block_specs = [
    (3.0,  1.8, 1.0, 1.0, 1.8),
    (5.5, -1.8, 1.0, 1.0, 2.0),
    (8.0,  1.8, 1.0, 1.0, 1.8),
    (10.5, -1.8, 1.0, 1.0, 2.0),
]

for (x, y, width, depth, height) in block_specs:
    add_box(
        center=[x, y, height / 2.0],
        half_extents=[width / 2.0, depth / 2.0, height / 2.0],
        rgba=[0.55, 0.55, 0.60, 1.0],
        obstacle_type="building"
    )

print(
    f"Environment obstacles created: {len(obstacle_boxes)}"
)
print()


# ============================================================
# CAMERA
# ============================================================

def capture_camera(position):

    camera_position = np.array([
        position[0] - 0.08,
        position[1],
        position[2]
    ])

    camera_target = np.array([
        position[0] + 3.0,
        position[1],
        position[2]
    ])

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_position.tolist(),
        cameraTargetPosition=camera_target.tolist(),
        cameraUpVector=[
            0,
            0,
            1
        ],
        physicsClientId=CLIENT
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=CAMERA_FOV,
        aspect=float(CAMERA_WIDTH)
        / float(CAMERA_HEIGHT),
        nearVal=CAMERA_NEAR,
        farVal=CAMERA_FAR
    )

    image = p.getCameraImage(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=CLIENT
    )

    raw = np.asarray(
        image[2],
        dtype=np.uint8
    )

    expected_rgb = (
        CAMERA_WIDTH
        * CAMERA_HEIGHT
        * 3
    )

    expected_rgba = (
        CAMERA_WIDTH
        * CAMERA_HEIGHT
        * 4
    )

    if raw.ndim == 1:

        if raw.size == expected_rgba:

            raw = raw.reshape(
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                4
            )

        elif raw.size == expected_rgb:

            raw = raw.reshape(
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                3
            )

        else:

            raise RuntimeError(
                "Unexpected camera buffer size: "
                f"{raw.size}"
            )

    if raw.ndim == 2:

        if raw.shape[1] == 4:

            raw = raw.reshape(
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                4
            )

        elif raw.shape[1] == 3:

            raw = raw.reshape(
                CAMERA_HEIGHT,
                CAMERA_WIDTH,
                3
            )

    if raw.ndim != 3:

        raise RuntimeError(
            f"Unexpected camera dimensions: "
            f"{raw.shape}"
        )

    rgb = raw[:, :, :3]

    return np.ascontiguousarray(
        rgb,
        dtype=np.uint8
    )


# ============================================================
# YOLO INITIALIZATION
# ============================================================

model = None

if YOLO_ENABLED:

    try:

        print(
            "Loading YOLO model..."
        )

        model = YOLO(
            YOLO_MODEL_NAME
        )

        print(
            "YOLO model loaded."
        )

    except Exception as exc:

        print(
            "[WARNING] YOLO could not be loaded:"
        )

        print(exc)

        print(
            "Continuing with physical "
            "obstacle safety."
        )

        model = None


print()


# ============================================================
# KEYBOARD
# ============================================================

def key_is_down(
    keys,
    key
):

    if key not in keys:

        return False

    return bool(
        keys[key] & p.KEY_IS_DOWN
    )


def read_manual_input():

    keys = p.getKeyboardEvents()

    return {

        "forward":
            key_is_down(
                keys,
                ord("w")
            ),

        "backward":
            key_is_down(
                keys,
                ord("s")
            ),

        "left":
            key_is_down(
                keys,
                ord("a")
            ),

        "right":
            key_is_down(
                keys,
                ord("d")
            ),

        "up":
            key_is_down(
                keys,
                ord("r")
            ),

        "down":
            key_is_down(
                keys,
                ord("f")
            )
    }


# ============================================================
# MANUAL TARGET
# ============================================================

def update_manual_target(
    target,
    manual_input,
    dt
):

    target = target.copy()

    forward = 0.0
    lateral = 0.0
    vertical = 0.0

    if manual_input["forward"]:
        forward += 1.0

    if manual_input["backward"]:
        forward -= 1.0

    if manual_input["left"]:
        lateral += 1.0

    if manual_input["right"]:
        lateral -= 1.0

    if manual_input["up"]:
        vertical += 1.0

    if manual_input["down"]:
        vertical -= 1.0

    # Normalize diagonal input.
    magnitude = math.sqrt(
        forward ** 2
        + lateral ** 2
        + vertical ** 2
    )

    if magnitude > 1.0:

        forward /= magnitude
        lateral /= magnitude
        vertical /= magnitude

    target[0] += (
        forward
        * MANUAL_SPEED
        * dt
    )

    target[1] += (
        lateral
        * MANUAL_SPEED
        * dt
    )

    target[2] += (
        vertical
        * VERTICAL_SPEED
        * dt
    )

    target[0] = np.clip(
        target[0],
        MANUAL_MIN_X,
        MANUAL_MAX_X
    )

    target[1] = np.clip(
        target[1],
        MANUAL_MIN_Y,
        MANUAL_MAX_Y
    )

    target[2] = np.clip(
        target[2],
        MIN_ALTITUDE,
        MAX_ALTITUDE
    )

    return target


# ============================================================
# BOX EXPANSION
# ============================================================

def expanded_box(
    box,
    radius=DRONE_SAFETY_RADIUS
):

    minimum = box["min"].copy()
    maximum = box["max"].copy()

    minimum -= np.array([
        radius,
        radius,
        radius
    ])

    maximum += np.array([
        radius,
        radius,
        radius
    ])

    return minimum, maximum


# ============================================================
# CHECK POINT AGAINST OBSTACLES
# ============================================================

def point_is_safe(
    point,
    extra_margin=0.0
):

    point = np.asarray(
        point,
        dtype=float
    )

    radius = (
        DRONE_SAFETY_RADIUS
        + extra_margin
    )

    for box in obstacle_boxes:

        minimum, maximum = expanded_box(
            box,
            radius
        )

        inside = (
            point[0] >= minimum[0]
            and
            point[0] <= maximum[0]
            and
            point[1] >= minimum[1]
            and
            point[1] <= maximum[1]
            and
            point[2] >= minimum[2]
            and
            point[2] <= maximum[2]
        )

        if inside:

            return False

    return True


# ============================================================
# SEGMENT SAFETY
# ============================================================

def segment_is_safe(
    start,
    end,
    samples=20
):

    start = np.asarray(
        start,
        dtype=float
    )

    end = np.asarray(
        end,
        dtype=float
    )

    for i in range(
        samples + 1
    ):

        alpha = (
            float(i)
            /
            float(samples)
        )

        point = (
            start
            +
            alpha
            * (end - start)
        )

        if not point_is_safe(
            point
        ):

            return False

    return True


# ============================================================
# FIND OBSTACLE AHEAD
#
# We do NOT just check Euclidean distance.
#
# The obstacle must:
#
# 1. Be ahead in +X.
# 2. Be close enough in X.
# 3. Overlap the drone's current lateral corridor.
# 4. Overlap the current altitude corridor.
# ============================================================

def find_obstacle_ahead(
    position
):

    nearest = None

    nearest_distance = float("inf")

    for box in obstacle_boxes:

        minimum, maximum = expanded_box(
            box
        )

        forward_distance = (
            minimum[0]
            -
            position[0]
        )

        if (
            forward_distance
            <
            -0.20
        ):

            continue

        if (
            forward_distance
            >
            SAFETY_LOOKAHEAD
        ):

            continue

        lateral_overlap = (
            position[1]
            >= minimum[1]
            - LATERAL_OVERLAP_MARGIN
            and
            position[1]
            <= maximum[1]
            + LATERAL_OVERLAP_MARGIN
        )

        if not lateral_overlap:

            continue

        vertical_overlap = (
            position[2]
            >= minimum[2]
            and
            position[2]
            <= maximum[2]
        )

        if not vertical_overlap:

            continue

        distance = max(
            0.0,
            forward_distance
        )

        if distance < nearest_distance:

            nearest_distance = distance

            nearest = box

    return nearest, nearest_distance


# ============================================================
# FIND SIDE CLEARANCE
#
# Returns the amount of usable space on a side.
# ============================================================

def calculate_side_clearance(
    position,
    side
):

    current_y = position[1]

    if side > 0:

        nearest_boundary = ENV_Y_MAX

        for box in obstacle_boxes:

            minimum, maximum = expanded_box(
                box,
                DRONE_SAFETY_RADIUS
                + 0.10
            )

            if (
                maximum[0]
                >=
                position[0]
                -
                0.20
                and
                minimum[0]
                <=
                position[0]
                +
                SAFETY_LOOKAHEAD
            ):

                if (
                    minimum[1]
                    >
                    current_y
                ):

                    nearest_boundary = min(
                        nearest_boundary,
                        minimum[1]
                    )

        return (
            nearest_boundary
            -
            current_y
        )

    else:

        nearest_boundary = ENV_Y_MIN

        for box in obstacle_boxes:

            minimum, maximum = expanded_box(
                box,
                DRONE_SAFETY_RADIUS
                + 0.10
            )

            if (
                maximum[0]
                >=
                position[0]
                -
                0.20
                and
                minimum[0]
                <=
                position[0]
                +
                SAFETY_LOOKAHEAD
            ):

                if (
                    maximum[1]
                    <
                    current_y
                ):

                    nearest_boundary = max(
                        nearest_boundary,
                        maximum[1]
                    )

        return (
            current_y
            -
            nearest_boundary
        )


# ============================================================
# CHOOSE SAFE SIDE
# ============================================================

def choose_avoidance_side(
    position,
    obstacle
):

    left_clearance = calculate_side_clearance(
        position,
        +1
    )

    right_clearance = calculate_side_clearance(
        position,
        -1
    )

    print()
    print(
        "[SAFETY] Left clearance:",
        f"{left_clearance:.2f} m"
    )

    print(
        "[SAFETY] Right clearance:",
        f"{right_clearance:.2f} m"
    )

    # Prefer the side with greater clearance.
    if (
        left_clearance
        >
        right_clearance
    ):

        return +1.0

    return -1.0


# ============================================================
# BUILD AVOIDANCE PLAN
#
# The plan is generated relative to the obstacle currently
# blocking the drone.
# ============================================================

def build_avoidance_plan(
    current_position,
    obstacle,
    side
):
    """
    Build a LOCAL avoidance route.

    Important design rule:
        The planner can calculate a route, but the planner does not
        directly command a distant point to the PID controller.

    The route is:
        1. brake/hold at the current position,
        2. move laterally while staying at roughly the same X,
        3. pass the obstacle,
        4. only then return to the saved lateral path,
        5. restore altitude.

    Every segment is checked geometrically.  The main loop also applies
    a per-cycle target slew limit, so even a bad/recomputed waypoint
    cannot produce a sudden multi-metre target jump.
    """

    current_position = np.asarray(
        current_position,
        dtype=float
    ).copy()

    # Use a conservative obstacle envelope.
    minimum, maximum = expanded_box(
        obstacle,
        DRONE_SAFETY_RADIUS + 0.18
    )

    # Find a side target with enough clearance.
    if side > 0:
        desired_y = (
            maximum[1]
            + MIN_SIDE_CLEARANCE
        )
    else:
        desired_y = (
            minimum[1]
            - MIN_SIDE_CLEARANCE
        )

    desired_y = float(
        np.clip(
            desired_y,
            ENV_Y_MIN + 0.55,
            ENV_Y_MAX - 0.55
        )
    )

    # Keep the same altitude initially.  A sudden vertical jump was one
    # of the failure modes this controller is specifically designed to
    # prevent.
    safe_z = float(
        np.clip(
            current_position[2],
            MIN_ALTITUDE + 0.05,
            MAX_ALTITUDE - 0.05
        )
    )

    # The drone should pass the complete expanded obstacle before
    # returning toward its original lateral path.
    pass_x = float(
        maximum[0] + PASS_MARGIN
    )

    # Keep the saved lateral path local to the current flight path.
    resume_y = float(
        np.clip(
            current_position[1],
            ENV_Y_MIN + 0.55,
            ENV_Y_MAX - 0.55
        )
    )

    # Candidate route.
    #
    # Note that the first waypoint is exactly the current position.
    # This removes the old "jump to a far-away waypoint" behaviour.
    candidates = [
        np.array([
            current_position[0],
            current_position[1],
            safe_z
        ], dtype=float),

        np.array([
            current_position[0],
            desired_y,
            safe_z
        ], dtype=float),

        np.array([
            pass_x,
            desired_y,
            safe_z
        ], dtype=float),

        np.array([
            pass_x,
            resume_y,
            safe_z
        ], dtype=float),

        np.array([
            pass_x,
            resume_y,
            float(
                np.clip(
                    current_position[2],
                    MIN_ALTITUDE,
                    MAX_ALTITUDE
                )
            )
        ], dtype=float)
    ]

    # Remove duplicate/near-duplicate points.
    plan = []
    previous = current_position.copy()

    for point in candidates:
        point = np.asarray(point, dtype=float)

        if np.linalg.norm(point - previous) < 0.05:
            continue

        plan.append(point)
        previous = point.copy()

    # Validate every complete segment, not just each endpoint.
    # This is important because a straight line between two safe points
    # can still cut through an obstacle.
    safe_plan = []
    previous = current_position.copy()

    for point in plan:
        if segment_is_safe(
            previous,
            point,
            samples=PATH_SAFETY_SAMPLES
        ):
            safe_plan.append(point.copy())
            previous = point.copy()
        else:
            # Do not return a geometrically unsafe route.  The main
            # loop will replan from the current position.
            print(
                "[SAFETY] Rejected unsafe avoidance segment:",
                previous,
                "->",
                point
            )
            break

    return safe_plan


# ============================================================
# SAFE AUTONOMOUS TARGET SLEW
# ============================================================

def limit_autonomous_target(
    current_position,
    requested_target
):
    """
    Convert an arbitrary requested waypoint into a small, bounded
    target increment.

    This is the final safety gate immediately before PID control.
    Therefore the PID can never suddenly receive a target several
    metres away from the drone.
    """

    current_position = np.asarray(
        current_position,
        dtype=float
    )

    requested_target = np.asarray(
        requested_target,
        dtype=float
    )

    delta = requested_target - current_position

    delta[0] = np.clip(
        delta[0],
        -AUTO_MAX_X_STEP,
        AUTO_MAX_X_STEP
    )

    delta[1] = np.clip(
        delta[1],
        -AUTO_MAX_Y_STEP,
        AUTO_MAX_Y_STEP
    )

    delta[2] = np.clip(
        delta[2],
        -AUTO_MAX_Z_STEP,
        AUTO_MAX_Z_STEP
    )

    limited_delta = delta.copy()

    distance = np.linalg.norm(
        limited_delta
    )

    if distance > AUTO_MAX_TARGET_STEP:
        limited_delta *= (
            AUTO_MAX_TARGET_STEP
            /
            distance
        )

    safe_target = (
        current_position
        +
        limited_delta
    )

    safe_target[2] = np.clip(
        safe_target[2],
        MIN_ALTITUDE,
        MAX_ALTITUDE
    )

    return safe_target


# ============================================================
# WAYPOINT REACHED
# ============================================================
# ============================================================

def waypoint_reached(
    position,
    target
):

    return (
        np.linalg.norm(
            position - target
        )
        <=
        WAYPOINT_TOLERANCE
    )


# ============================================================
# STATE
# ============================================================

mode = "MANUAL"

manual_target = INITIAL_POSITION.copy()

avoidance_waypoints = []

avoidance_index = 0

avoidance_side = 0.0

avoidance_obstacle = None

resume_position = INITIAL_POSITION.copy()

handoff_start = 0.0
avoidance_start = 0.0
emergency_hold_start = 0.0


# ============================================================
# STATISTICS
# ============================================================

start_time = time.time()

last_status_time = time.time()

last_camera_time = time.time()

camera_frames = 0

total_yolo_detections = 0

yolo_center_detections = 0

safety_takeovers = 0

completed_avoidances = 0

collision_events = 0

last_yolo_detected = False

last_frame = None


# ============================================================
# START MESSAGE
# ============================================================

print("======================================================")
print("HUMAN-IN-THE-LOOP SYSTEM READY")
print("======================================================")
print()

print("MANUAL CONTROL:")
print()
print("W = FORWARD")
print("S = BACKWARD")
print("A = LEFT")
print("D = RIGHT")
print("R = UP")
print("F = DOWN")
print()
print("HOLD KEY = MOVE")
print("RELEASE KEY = STOP")
print()

print("SAFETY SYSTEM:")
print("AUTOMATIC OBSTACLE AVOIDANCE ENABLED")
print()

print("YOLO:")
print(
    "ENABLED"
    if model is not None
    else
    "UNAVAILABLE - PHYSICAL SAFETY ACTIVE"
)

print()

print("Press Q in the camera window to stop.")
print()

print("Drone is now under MANUAL control.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

try:

    total_steps = int(
        MAX_DURATION
        *
        CONTROL_FREQ
    )

    for step in range(
        total_steps
    ):

        # ====================================================
        # CURRENT STATE
        # ====================================================

        state = env._getDroneStateVector(
            0
        )

        current_position = np.array(
            state[0:3],
            dtype=float
        )

        current_quaternion = np.array(
            state[3:7],
            dtype=float
        )

        current_velocity = np.array(
            state[10:13],
            dtype=float
        )

        current_angular_velocity = np.array(
            state[13:16],
            dtype=float
        )


        # ====================================================
        # KEYBOARD
        # ====================================================

        manual_input = read_manual_input()

        pilot_forward = (
            manual_input["forward"]
            and
            not manual_input["backward"]
        )

        pilot_moving = any(
            manual_input.values()
        )


        # ====================================================
        # CAMERA / YOLO
        # ====================================================

        detected_center = False

        if (
            step
            %
            CAMERA_INTERVAL
            ==
            0
        ):

            try:

                frame = capture_camera(
                    current_position
                )

                camera_frames += 1

                if model is not None:

                    results = model.predict(
                        source=frame,
                        conf=YOLO_CONFIDENCE,
                        verbose=False
                    )

                    result = results[0]

                    image_width = int(
                        result.orig_shape[1]
                    )

                    for box in result.boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0].tolist()
                        )

                        center_x = (
                            x1 + x2
                        ) / 2.0

                        confidence = float(
                            box.conf[0]
                        )

                        total_yolo_detections += 1

                        if (
                            image_width * 0.25
                            <=
                            center_x
                            <=
                            image_width * 0.75
                        ):

                            detected_center = True

                            yolo_center_detections += 1

                            break

                    annotated = result.plot()

                else:

                    annotated = frame.copy()

                last_frame = annotated.copy()

            except Exception as exc:

                print(
                    "[CAMERA WARNING]",
                    exc
                )


        last_yolo_detected = (
            detected_center
        )


        # ====================================================
        # FIND PHYSICAL OBSTACLE
        # ====================================================

        obstacle_ahead, obstacle_distance = (
            find_obstacle_ahead(
                current_position
            )
        )


        # ====================================================
        # DEFAULT TARGET
        # ====================================================

        control_target = (
            current_position.copy()
        )


        # ====================================================
        # MANUAL MODE
        # ====================================================

        if mode == "MANUAL":

            dt = (
                1.0
                /
                CONTROL_FREQ
            )

            manual_target = (
                update_manual_target(
                    manual_target,
                    manual_input,
                    dt
                )
            )

            target_error = (
                manual_target
                -
                current_position
            )

            target_error = np.clip(
                target_error,
                -MAX_MANUAL_TARGET_ERROR,
                MAX_MANUAL_TARGET_ERROR
            )

            control_target = (
                current_position
                +
                target_error
            )


            # =================================================
            # SAFETY DECISION
            # =================================================

            takeover = False

            # Physical obstacle safety.
            if (
                obstacle_ahead is not None
                and
                pilot_forward
            ):

                takeover = True


            # Emergency condition.
            if (
                obstacle_ahead is not None
                and
                obstacle_distance
                <=
                EMERGENCY_DISTANCE
            ):

                takeover = True


            # If the pilot is not moving forward but the drone
            # has drifted dangerously close to an obstacle,
            # safety still activates.
            if (
                obstacle_ahead is not None
                and
                obstacle_distance
                <=
                0.35
            ):

                takeover = True


            if takeover:

                mode = "AUTONOMOUS_AVOIDANCE"

                safety_takeovers += 1

                avoidance_obstacle = obstacle_ahead

                # Save the position at the instant of takeover.  This
                # is the path position to which we will eventually
                # return laterally.
                resume_position = (
                    current_position.copy()
                )

                avoidance_side = (
                    choose_avoidance_side(
                        current_position,
                        obstacle_ahead
                    )
                )

                avoidance_waypoints = (
                    build_avoidance_plan(
                        current_position,
                        obstacle_ahead,
                        avoidance_side
                    )
                )

                avoidance_index = 0
                avoidance_start = time.time()

                # Kill the old manual target immediately.  Otherwise
                # manual input from the previous frame could pull the
                # PID back toward the obstacle.
                manual_target = (
                    current_position.copy()
                )

                if not avoidance_waypoints:
                    # Never continue with an invalid route.  Hold the
                    # current position and let the controller replan.
                    mode = "EMERGENCY_HOLD"
                    emergency_hold_start = time.time()

                    print()
                    print(
                        ">>> NO SAFE AVOIDANCE PATH FOUND - HOLDING"
                    )
                    print(
                        ">>> Manual control remains suspended."
                    )
                    print()

                else:
                    print()
                    print(
                        "======================================================"
                    )
                    print(
                        ">>> AUTONOMOUS SAFETY TAKEOVER"
                    )
                    print(
                        "======================================================"
                    )

                    print(
                        "Obstacle type:",
                        obstacle_ahead["type"]
                    )

                    print(
                        "Obstacle distance:",
                        f"{obstacle_distance:.3f} m"
                    )

                    print(
                        "Selected side:",
                        "LEFT"
                        if avoidance_side > 0
                        else "RIGHT"
                    )

                    print(
                        "Pilot control temporarily suspended."
                    )

                    print(
                        "BRAKING -> LATERAL CLEARANCE -> PASS -> RESUME"
                    )

                    print()

        # ====================================================
        # AUTONOMOUS AVOIDANCE
        # ====================================================

        if mode == "AUTONOMOUS_AVOIDANCE":

            # Re-check the environment every control cycle.
            # The important behavior here is: once the drone has moved
            # far enough sideways that the original obstacle is no longer
            # blocking the current flight corridor, STOP autonomous
            # navigation and begin the manual handoff immediately.
            current_obstacle, current_distance = (
                find_obstacle_ahead(
                    current_position
                )
            )

            # ----------------------------------------------------
            # BRAKE / HOLD PHASE
            # ----------------------------------------------------
            elapsed_avoidance = (
                time.time()
                - avoidance_start
            )

            if elapsed_avoidance < BRAKE_TIME:

                control_target = current_position.copy()

            else:

                # ------------------------------------------------
                # EARLY MANUAL HANDOFF
                # ------------------------------------------------
                #
                # The old logic forced the drone to follow the complete
                # avoidance route: move sideways -> pass obstacle ->
                # return toward the original path.  That is why the
                # drone could keep travelling autonomously even after
                # the obstacle was already clear.
                #
                # New behavior:
                # If there is NO obstacle in the current forward corridor,
                # stop generating/using the remaining avoidance waypoints
                # and hand control back to the pilot.
                if current_obstacle is None:

                    avoidance_waypoints = []
                    avoidance_index = 0
                    manual_target = current_position.copy()
                    mode = "MANUAL_HANDOFF"
                    handoff_start = time.time()

                    print()
                    print(
                        ">>> PATH CLEAR AFTER LATERAL AVOIDANCE"
                    )
                    print(
                        ">>> SWITCHING TO MANUAL HANDOFF"
                    )
                    print()

                else:

                    # Continue the current local avoidance only while an
                    # obstacle is still blocking the corridor.
                    if (
                        avoidance_index
                        >=
                        len(avoidance_waypoints)
                    ):
                        # Route ended but something is still ahead.
                        # Replan rather than blindly continuing.
                        avoidance_obstacle = current_obstacle
                        avoidance_side = choose_avoidance_side(
                            current_position,
                            current_obstacle
                        )
                        avoidance_waypoints = build_avoidance_plan(
                            current_position,
                            current_obstacle,
                            avoidance_side
                        )
                        avoidance_index = 0
                        avoidance_start = time.time()

                        if not avoidance_waypoints:
                            mode = "EMERGENCY_HOLD"
                            emergency_hold_start = time.time()

                    if mode == "AUTONOMOUS_AVOIDANCE":

                        requested_target = (
                            avoidance_waypoints[
                                avoidance_index
                            ].copy()
                        )

                        # Re-check the requested segment before moving.
                        if not segment_is_safe(
                            current_position,
                            requested_target,
                            samples=PATH_SAFETY_SAMPLES
                        ):
                            print(
                                "[SAFETY] Current avoidance segment "
                                "became unsafe. Replanning."
                            )

                            avoidance_obstacle = current_obstacle
                            avoidance_side = choose_avoidance_side(
                                current_position,
                                current_obstacle
                            )
                            avoidance_waypoints = build_avoidance_plan(
                                current_position,
                                current_obstacle,
                                avoidance_side
                            )
                            avoidance_index = 0

                            if not avoidance_waypoints:
                                mode = "EMERGENCY_HOLD"
                                emergency_hold_start = time.time()

                        if mode == "AUTONOMOUS_AVOIDANCE":

                            control_target = limit_autonomous_target(
                                current_position,
                                requested_target
                            )

                            if waypoint_reached(
                                current_position,
                                requested_target
                            ):
                                avoidance_index += 1

                                # Do NOT automatically execute the next
                                # "pass" or "resume" waypoint here.
                                # The next control cycle will first check
                                # whether the forward corridor is still
                                # blocked. If it is clear, control returns
                                # to MANUAL.

                                if (
                                    avoidance_index
                                    >=
                                    len(avoidance_waypoints)
                                ):
                                    # The route finished. Verify once more.
                                    remaining_obstacle, remaining_distance = (
                                        find_obstacle_ahead(
                                            current_position
                                        )
                                    )

                                    if remaining_obstacle is None:
                                        mode = "MANUAL_HANDOFF"
                                        handoff_start = time.time()
                                        manual_target = current_position.copy()
                                        completed_avoidances += 1
                                    else:
                                        avoidance_obstacle = remaining_obstacle
                                        avoidance_side = choose_avoidance_side(
                                            current_position,
                                            remaining_obstacle
                                        )
                                        avoidance_waypoints = build_avoidance_plan(
                                            current_position,
                                            remaining_obstacle,
                                            avoidance_side
                                        )
                                        avoidance_index = 0
                                        avoidance_start = time.time()

                                        if not avoidance_waypoints:
                                            mode = "EMERGENCY_HOLD"
                                            emergency_hold_start = time.time()


        # ====================================================
        # EMERGENCY HOLD
        # ====================================================

        if mode == "EMERGENCY_HOLD":

            # The emergency state has only one job:
            # HOLD POSITION.  It never invents a distant target.
            control_target = (
                current_position.copy()
            )

            manual_target = (
                current_position.copy()
            )

            elapsed_hold = (
                time.time()
                -
                emergency_hold_start
            )

            # Try to rebuild a route after the drone has settled.
            if elapsed_hold >= BRAKE_TIME:

                new_obstacle, new_distance = (
                    find_obstacle_ahead(
                        current_position
                    )
                )

                if new_obstacle is not None:

                    avoidance_obstacle = (
                        new_obstacle
                    )

                    avoidance_side = (
                        choose_avoidance_side(
                            current_position,
                            new_obstacle
                        )
                    )

                    avoidance_waypoints = (
                        build_avoidance_plan(
                            current_position,
                            new_obstacle,
                            avoidance_side
                        )
                    )

                    avoidance_index = 0
                    avoidance_start = time.time()

                    if avoidance_waypoints:
                        mode = "AUTONOMOUS_AVOIDANCE"

                else:
                    # No obstacle is currently blocking the path.
                    # Do not instantly restore control; use the normal
                    # stabilization handoff.
                    mode = "MANUAL_HANDOFF"
                    handoff_start = time.time()

        # ====================================================
        # MANUAL HANDOFF
        # ====================================================

        if mode == "MANUAL_HANDOFF":

            control_target = (
                current_position.copy()
            )

            manual_target = (
                current_position.copy()
            )

            elapsed = (
                time.time()
                -
                handoff_start
            )

            # Never hand control back while an obstacle is still
            # blocking the current flight corridor.
            obstacle_still_ahead, obstacle_still_distance = (
                find_obstacle_ahead(
                    current_position
                )
            )

            if (
                elapsed
                >=
                HANDOFF_TIME
                and
                obstacle_still_ahead is None
                and
                np.linalg.norm(current_velocity) <=
                BRAKE_SPEED_THRESHOLD
            ):

                mode = "MANUAL"

                manual_target = (
                    current_position.copy()
                )

                print()
                print(
                    "======================================================"
                )
                print(
                    ">>> MANUAL CONTROL RESTORED"
                )
                print(
                    "======================================================"
                )

                print(
                    "W/S/A/D/R/F are active."
                )

                print()


        # ====================================================
        # FINAL MISSION CHECK
        # ====================================================

        distance_to_goal = np.linalg.norm(
            current_position
            -
            FINAL_TARGET
        )

        if (
            distance_to_goal
            <=
            0.20
        ):

            print()
            print(
                "======================================================"
            )
            print(
                ">>> FINAL DESTINATION REACHED"
            )
            print(
                "======================================================"
            )

            print(
                "Final distance:",
                f"{distance_to_goal:.3f} m"
            )

            print()

            break


        # ====================================================
        # FINAL GLOBAL TARGET SANITY CHECK
        # ====================================================
        #
        # This is an additional last-resort guard.  In autonomous
        # modes the target is already slew-limited.  If anything
        # unexpected gets through, never give the PID a large jump.
        if mode in (
            "AUTONOMOUS_AVOIDANCE",
            "EMERGENCY_HOLD",
            "MANUAL_HANDOFF"
        ):
            control_target = limit_autonomous_target(
                current_position,
                control_target
            )

        # ====================================================
        # ALTITUDE PROTECTION
        # ====================================================

        control_target[2] = np.clip(
            control_target[2],
            MIN_ALTITUDE,
            MAX_ALTITUDE
        )

        if (
            current_position[2]
            <
            MIN_ALTITUDE
        ):

            control_target[2] = (
                MIN_ALTITUDE
                +
                0.15
            )


        # ====================================================
        # PID CONTROL
        # ====================================================

        action, pos_error, yaw_error = (
            controller.computeControl(

                control_timestep=
                env.CTRL_TIMESTEP,

                cur_pos=
                current_position,

                cur_quat=
                current_quaternion,

                cur_vel=
                current_velocity,

                cur_ang_vel=
                current_angular_velocity,

                target_pos=
                control_target
            )
        )


        # ====================================================
        # ACTION FORMAT
        # ====================================================

        action = np.asarray(
            action,
            dtype=float
        )

        if action.ndim == 1:

            action = action.reshape(
                1,
                4
            )


        # ====================================================
        # PYBULLET STEP
        # ====================================================

        env.step(
            action
        )


        # ====================================================
        # COLLISION CHECK
        # ====================================================

        try:

            contact_points = (
                p.getContactPoints(
                    bodyA=env.DRONE_IDS[0],
                    physicsClientId=CLIENT
                )
            )

            real_obstacle_collision = False

            for contact in contact_points:

                body_b = contact[2]

                if body_b in environment_objects:

                    real_obstacle_collision = True

                    break

            if real_obstacle_collision:

                collision_events += 1

                print()
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                print(
                    "!!! COLLISION DETECTED !!!"
                )
                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )
                print()

        except Exception:

            pass


        # ====================================================
        # DISPLAY
        # ====================================================

        if last_frame is not None:

            display = (
                last_frame.copy()
            )

            if mode == "MANUAL":

                mode_color = (
                    0,
                    255,
                    0
                )

            elif mode == "AUTONOMOUS_AVOIDANCE":

                mode_color = (
                    0,
                    0,
                    255
                )

            elif mode == "EMERGENCY_HOLD":

                mode_color = (
                    0,
                    0,
                    255
                )

            else:

                mode_color = (
                    0,
                    255,
                    255
                )


            cv2.putText(
                display,
                f"MODE: {mode}",
                (
                    15,
                    30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                mode_color,
                2
            )


            cv2.putText(
                display,
                (
                    f"Position: "
                    f"{current_position[0]:.2f}, "
                    f"{current_position[1]:.2f}, "
                    f"{current_position[2]:.2f}"
                ),
                (
                    15,
                    60
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    255,
                    255,
                    255
                ),
                2
            )


            if obstacle_ahead is not None:

                cv2.putText(
                    display,
                    (
                        "OBSTACLE AHEAD: "
                        f"{obstacle_distance:.2f} m"
                    ),
                    (
                        15,
                        90
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "OBSTACLE AHEAD: CLEAR",
                    (
                        15,
                        90
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


            cv2.putText(
                display,
                (
                    f"Goal distance: "
                    f"{distance_to_goal:.2f} m"
                ),
                (
                    15,
                    120
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (
                    255,
                    255,
                    0
                ),
                2
            )


            if mode == "MANUAL":

                cv2.putText(
                    display,
                    "PILOT CONTROL",
                    (
                        15,
                        150
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )

                cv2.putText(
                    display,
                    "W/S/A/D = MOVE",
                    (
                        15,
                        178
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

                cv2.putText(
                    display,
                    "R/F = ALTITUDE",
                    (
                        15,
                        204
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (
                        255,
                        255,
                        255
                    ),
                    2
                )

            elif mode == "AUTONOMOUS_AVOIDANCE":

                cv2.putText(
                    display,
                    "AUTONOMOUS AVOIDANCE",
                    (
                        15,
                        150
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

                cv2.putText(
                    display,
                    (
                        "SIDE: "
                        +
                        (
                            "LEFT"
                            if avoidance_side > 0
                            else
                            "RIGHT"
                        )
                    ),
                    (
                        15,
                        178
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (
                        0,
                        255,
                        255
                    ),
                    2
                )

            elif mode == "EMERGENCY_HOLD":

                cv2.putText(
                    display,
                    "EMERGENCY HOLD - REPLANNING",
                    (
                        15,
                        150
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "STABILIZING / HANDOFF",
                    (
                        15,
                        150
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (
                        0,
                        255,
                        255
                    ),
                    2
                )


            if last_yolo_detected:

                cv2.putText(
                    display,
                    "YOLO: CENTER OBJECT",
                    (
                        15,
                        230
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "YOLO: NO CENTER OBJECT",
                    (
                        15,
                        230
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (
                        0,
                        255,
                        0
                    ),
                    2
                )


            cv2.imshow(
                "STAGE 25 - HUMAN IN THE LOOP",
                display
            )


        # ====================================================
        # Q TO EXIT
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            print()
            print(
                "Q pressed."
            )

            break


        # ====================================================
        # STATUS
        # ====================================================

        now = time.time()

        if (
            now
            -
            last_status_time
            >=
            1.0
        ):

            print(
                f"Mode: {mode} | "
                f"Position: "
                f"[{current_position[0]:.2f}, "
                f"{current_position[1]:.2f}, "
                f"{current_position[2]:.2f}] | "
                f"Goal: "
                f"{distance_to_goal:.2f} m | "
                f"Obstacle: "
                f"{obstacle_distance:.2f} m"
                if obstacle_ahead is not None
                else
                (
                    f"Mode: {mode} | "
                    f"Position: "
                    f"[{current_position[0]:.2f}, "
                    f"{current_position[1]:.2f}, "
                    f"{current_position[2]:.2f}] | "
                    f"Goal: "
                    f"{distance_to_goal:.2f} m | "
                    f"Obstacle: CLEAR"
                )
            )

            last_status_time = now


        # ====================================================
        # TIME LIMIT
        # ====================================================

        if (
            now
            -
            start_time
            >=
            MAX_DURATION
        ):

            print()
            print(
                "Maximum duration reached."
            )

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print()
    print(
        "Shutting down Stage 25..."
    )

    try:

        cv2.destroyAllWindows()

    except Exception:

        pass


    try:

        final_state = (
            env._getDroneStateVector(0)
        )

        final_position = np.array(
            final_state[0:3],
            dtype=float
        )

    except Exception:

        final_position = np.array([
            np.nan,
            np.nan,
            np.nan
        ])


    try:

        env.close()

    except Exception:

        pass


# ============================================================
# FINAL RESULTS
# ============================================================

final_error = np.linalg.norm(
    final_position
    -
    FINAL_TARGET
)


print()
print("======================================================")
print("STAGE 25 COMPLETED")
print("======================================================")
print()

print(
    "Initial position:"
)

print(
    INITIAL_POSITION
)

print()

print(
    "Final target:"
)

print(
    FINAL_TARGET
)

print()

print(
    "Final drone position:"
)

print(
    final_position
)

print()

print(
    "Final position error:"
)

print(
    f"{final_error:.4f} m"
)

print()

print(
    "Autonomous safety takeovers:"
)

print(
    safety_takeovers
)

print()

print(
    "Completed autonomous avoidances:"
)

print(
    completed_avoidances
)

print()

print(
    "Collision events:"
)

print(
    collision_events
)

print()

print(
    "Live camera frames:"
)

print(
    camera_frames
)

print()

print(
    "Total YOLO detections:"
)

print(
    total_yolo_detections
)

print()

print(
    "YOLO center detections:"
)

print(
    yolo_center_detections
)

print()

print(
    "Manual control:"
)

print(
    "ENABLED"
)

print()

print(
    "Reactive physical obstacle safety:"
)

print(
    "ENABLED"
)

print()

print(
    "Automatic left/right decision:"
)

print(
    "ENABLED"
)

print()

print(
    "Autonomous obstacle avoidance:"
)

print(
    "ENABLED"
)

print()

print(
    "Automatic repeated avoidance:"
)

print(
    "ENABLED"
)

print()

print(
    "Stable manual handoff:"
)

print(
    "ENABLED"
)

print()

print(
    "YOLO perception:"
)

print(
    "ENABLED"
    if model is not None
    else
    "UNAVAILABLE"
)

print()

print(
    "Results directory:"
)

print(
    RESULT_DIR
)

print()

print("======================================================")