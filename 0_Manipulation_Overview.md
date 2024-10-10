# Manipulation.py: Overview

This tutorial provides an overview of all manipualtion.py helper functions and their usage context.

To better understand manipulation.py it is recomended to first take a look at [Robotic Control Interface & Manipulation Planning Library](https://marctoussaint.github.io/robotics-course/) to get a better understanding of the robotic library and all of it's components.

## Intro to manipulation.py
...

## Initial problem definition

**<span style="color:#FF5733">setup_inverse_kinematics</span>**
- <span style="color:orange">homing_scale</span>: float=1e-1
- <span style="color:orange">accumulated_collisions</span>: bool=True
- <span style="color:orange">quaternion_norms</span>: bool=False

Initialises a simple inverse kinematics optimization problem.

---

**<span style="color:#FF5733">setup_multi_phase_problem</span>**
- <span style="color:orange">phases</span>: int
- <span style="color:orange">slices_per_phase</span>: int=1
- <span style="color:orange">accumulated_collisions</span>: bool=True
- <span style="color:orange">joint_limits</span>: bool=True
- <span style="color:orange">quaternion_norms</span>: bool=False

Initialises multiple inverse kinematics problems inside of a single problem.

---

**<span style="color:#FF5733">setup_pick_and_place_waypoints</span>**
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">obj</span>: str
- <span style="color:orange">homing_scale</span>: float=1e-2
- <span style="color:orange">velocity_scale</span>: float=1e-1
- <span style="color:orange">accumulated_collisions</span>: bool=True
- <span style="color:orange">joint_limits</span>: bool=True
- <span style="color:orange">quaternion_norms</span>: bool=False

Initialises two waypoints between which an object is attached to the gripper of the robot.

---

**<span style="color:#FF5733">setup_pick_and_place_sequence</span>**
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">table</span>: str
- <span style="color:orange">objs</span>: list[str]
- <span style="color:orange">homing_scale</span>: float=1e-2
- <span style="color:orange">velocity_scale</span>: float=1e-1
- <span style="color:orange">accumulated_collisions</span>: bool=True
- <span style="color:orange">joint_limits</span>: bool=True
- <span style="color:orange">quaternion_norms</span>: bool=False

Initialises two waypoints for each of the objects defined, where one waypoint is for approaching the object and the other attaches the object to the gripper.

---

**<span style="color:#FF5733">setup_point_to_point_motion</span>**
- <span style="color:orange">q0</span>: list[float]
- <span style="color:orange">q1</span>: list[float]
- <span style="color:orange">homing_scale</span>: float=1e-2
- <span style="color:orange">acceleration_scale</span>: float=1e-1
- <span style="color:orange">accumulated_collisions</span>: bool=True
- <span style="color:orange">quaternion_norms</span>: bool=False

Initialises a problem with 32 phases in between with one initial joint state *q0* and *q1*.

---

**<span style="color:#FF5733">setup_point_to_point_rrt</span>**
- <span style="color:orange">q0</span>: list[float]
- <span style="color:orange">q1</span>: list[float]
- <span style="color:orange">explicitCollisionPairs</span>: list[str]=[]

Initialises a path-finding problem to be solved with RRT. *q0* is the starting joint state and *q1* is the desired end state.

## Objective definitions

**<span style="color:#FF5733">follow_path_on_plane_xy</span>**
- <span style="color:orange">path</span>: list[list[float]]
- <span style="color:orange">moving_frame</span>: str

---

**<span style="color:#FF5733">path_must_be_straight</span>**
- <span style="color:orange">times</span>: list[float]
- <span style="color:orange">start_frame</span>: str
- <span style="color:orange">end_frame</span>: str
- <span style="color:orange">moving_frame</span>: str
- <span style="color:orange">gotoPoints</span>: bool=False

---

**<span style="color:#FF5733">keep_distance</span>**
- <span style="color:orange">time_interval</span>: list[float]
- <span style="color:orange">obj1</span>: str
- <span style="color:orange">obj2</span>: str
- <span style="color:orange">margin</span>: float=.001

---

**<span style="color:#FF5733">keep_distances</span>**
- <span style="color:orange">time_interval</span>: list[float]
- <span style="color:orange">objs</span>: list[str]
- <span style="color:orange">margin</span>: float=.001

---

**<span style="color:#FF5733">set_relative_distance</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj1</span>: str
- <span style="color:orange">obj2</span>: str
- <span style="color:orange">distance</span>: float

---

**<span style="color:#FF5733">switch_pick</span>**
- <span style="color:orange">time_interval</span>: list[float]
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">obj</span>: str

---

**<span style="color:#FF5733">switch_place</span>**
- <span style="color:orange">time_interval</span>: list[float]
- <span style="color:orange">table</span>: str
- <span style="color:orange">obj</span>: str

---

**<span style="color:#FF5733">target_position</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">pos</span>: list[float]

---

**<span style="color:#FF5733">target_xy_position</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">pos</span>: list[float]

---

**<span style="color:#FF5733">target_relative_xy_position</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">relativeTo</span>: str
- <span style="color:orange">pos</span>: list[float]

---

**<span style="color:#FF5733">target_x_orientation</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">x_vector</span>: list[float]

---

**<span style="color:#FF5733">target_y_orientation</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">y_vector</span>: list[float]

---

**<span style="color:#FF5733">target_z_orientation</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">z_vector</span>: list[float]

## Manipulation helpers

**<span style="color:#FF5733">grasp_top_box</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">obj</span>: str
- <span style="color:orange">grasp_direction</span>: str='xz'

---

**<span style="color:#FF5733">grasp_box</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">obj</span>: str
- <span style="color:orange">palm</span>: str
- <span style="color:orange">grasp_direction</span>: str='x'
- <span style="color:orange">margin</span>: float=.02

---

**<span style="color:#FF5733">grasp_cylinder</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">obj</span>: str
- <span style="color:orange">palm</span>: str
- <span style="color:orange">margin</span>: float=.02

---

**<span style="color:#FF5733">place_box</span>**
- <span style="color:orange">time</span>: float
- <span style="color:orange">obj</span>: str
- <span style="color:orange">table</span>: str
- <span style="color:orange">palm</span>: str
- <span style="color:orange">place_direction</span>: str='z'
- <span style="color:orange">margin</span>: float=.02

---

**<span style="color:#FF5733">straight_push</span>**
- <span style="color:orange">times</span>: list[float]
- <span style="color:orange">obj</span>: str
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">table</span>: str

---

**<span style="color:#FF5733">pull</span>**
- <span style="color:orange">times</span>: list[float]
- <span style="color:orange">obj</span>: str
- <span style="color:orange">gripper</span>: str
- <span style="color:orange">table</span>: str
