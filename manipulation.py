import robotic as ry
import numpy as np
import time

class ManipulationModelling():

    def __init__(self, C: ry.Config, info: str="", helpers: list[str]=[]):
        """
        Initialize a new instance of the Manipulation class.

        Args:
            C (ry.Config): The current robotic configuration. This object represents the robot and environment setup,
                        which is used for planning and control tasks.
            info (str, optional): An optional string for providing additional information or description related to this
                                manipulation instance. Default is an empty string.
            helpers (list[str], optional): A list of helper frame names. These helper frames are used to create additional
                                        start and end frames (e.g., '_frame_start' and '_frame_end') for each entry in the list.
                                        Default is an empty list.
        """
        self.C = C
        self.info = info
        self.helpers = helpers
        for frame in helpers:
            name = f'_{frame}_end'
            f = self.C.getFrame(name, False)
            if not f:
               self.C.addFrame(name)
               
            name = f"_{frame}_start"
            f = self.C.getFrame(name, False)
            if not f:
                self.C.addFrame(name)

        self.komo: ry.KOMO=None

    def setup_inverse_kinematics(self, homing_scale: float = 1e-1, accumulated_collisions: bool = True, quaternion_norms: bool = False):
        """
        Set up a single-phase inverse kinematics problem with optional constraints.

        Args:
            homing_scale (float, optional): The scaling factor for the homing objective, which controls the strength of
                                            the objective that brings the system towards a home (default) position. 
                                            Default value is 0.1.
            accumulated_collisions (bool, optional): If True, imposes a constraint on accumulated collisions to minimize
                                                     collisions between objects. Default is True.
            quaternion_norms (bool, optional): If True, imposes a quaternion normalization constraint to ensure proper 
                                            orientation representations. Default is False.

        """
        self.komo = ry.KOMO(self.C, 1, 1, 0, accumulated_collisions)
        
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e0])

        self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, scale=[1e0])

        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def setup_multi_phase_problem(self, phases: int, slices_per_phase: int = 1, accumulated_collisions: bool = True, joint_limits: bool = True, quaternion_norms: bool = False):
        """
        Set up a motion problem with multiple phases for inverse kinematics.

        Args:
            phases (int): The number of phases in the motion problem. Each phase represents a segment of the trajectory.
            slices_per_phase (int, optional): The number of slices (time steps) per phase. Default is 1.
            accumulated_collisions (bool, optional): If True, adds a constraint on accumulated collisions to minimize  
                                                     collisions between objects. Default is True.
            joint_limits (bool, optional): If True, imposes constraints on joint limits to ensure the robot's joints 
                                            stay within their allowable range. Default is True.
            quaternion_norms (bool, optional): If True, adds a constraint to normalize quaternions to ensure valid 
                                            orientation representations. Default is False.

        """
        self.komo = ry.KOMO()
        self.komo.setConfig(self.C, accumulated_collisions)
        self.komo.setTiming(phases, slices_per_phase, 1.0, 2)

        self.komo.addControlObjective([], 1, 1e-1)  # Control order 1
        self.komo.addControlObjective([], 2, 1e-1)  # Control order 2

        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e0])

        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, scale=[1e0])

        if quaternion_norms:
            self.komo.addQuaternionNorms()

    def setup_pick_and_place_waypoints(self, gripper: str, obj: str, homing_scale: float = 1e-2, velocity_scale: float = 1e-1, accumulated_collisions: bool = True, joint_limits: bool = True, quaternion_norms: bool = False):
        """
        Set up a two-phase pick-and-place motion problem with a pick switch at timestep 1, and a place switch at timestep 2.

        Args:
            gripper (str): The name of the gripper that will pick and place the object.
            obj (str): The name of the object to be picked and placed.
            homing_scale (float, optional): The scaling factor for the homing objective, which encourages the system to 
                                            move towards a home position. Default is 0.01.
            velocity_scale (float, optional): The scaling factor for the velocity control objective, influencing the 
                                            rate of movement. Default is 0.1.
            accumulated_collisions (bool, optional): If True, adds a constraint on accumulated collisions to minimize
                                                     collisions between objects. Default is True.
            joint_limits (bool, optional): If True, imposes constraints on joint limits to ensure the robot's joints 
                                            stay within their allowable range. Default is True.
            quaternion_norms (bool, optional): If True, adds a constraint to normalize quaternions to ensure valid 
                                            orientation representations. Default is False.
        """
        self.komo = ry.KOMO(self.C, 2.0, 1, 1, accumulated_collisions)
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        self.komo.addControlObjective([], order=1, scale=velocity_scale)
        
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e0])

        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, scale=[1e0])

        if quaternion_norms:
            self.komo.addQuaternionNorms()

        self.komo.addModeSwitch([1.0, -1.0], ry.SY.stable, [gripper, obj], True)

    def setup_pick_and_place_sequence(self, gripper: str, table: str, objs: list[str], homing_scale: float=1e-2, velocity_scale: float=1e-1, accumulated_collisions: bool=True, joint_limits: bool=True, quaternion_norms: bool=False):
        """
        Set up a sequence for pick-and-place actions involving multiple objects.

        Args:
            gripper (str): The name of the gripper being used for the pick-and-place operations.
            table (str): The name of the table where objects are placed and picked from.
            objs (list[str]): A list of object names that will be picked and placed.
            homing_scale (float, optional): The scaling factor for the homing objective, which encourages the system to 
                                            move towards a home position. Default is 0.01.
            velocity_scale (float, optional): The scaling factor for the velocity control objective, influencing the 
                                            rate of movement. Default is 0.1.
            accumulated_collisions (bool, optional): If True, adds a constraint on accumulated collisions to minimize
                                                     collisions between objects. Default is True.
            joint_limits (bool, optional): If True, imposes constraints on joint limits to ensure the robot's joints 
                                            stay within their allowable range. Default is True.
            quaternion_norms (bool, optional): If True, adds a constraint to normalize quaternions to ensure valid 
                                            orientation representations. Default is False.

        """
        phases = len(objs) * 2
        self.komo = ry.KOMO(self.C, phases, 1, 1, accumulated_collisions)
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        self.komo.addControlObjective([], order=1, scale=velocity_scale)
        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e1])

        if joint_limits:
            self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, scale=[1e0])

        if quaternion_norms:
            self.komo.addQuaternionNorms()

        for i, obj in enumerate(objs):
            times = [i*2+1, i*2+2]
            self.komo.addModeSwitch(times, ry.SY.stable, [gripper, obj])
            self.komo.addModeSwitch([times[1], -1.], ry.SY.stable, [table, obj])

    def setup_point_to_point_motion(self, q0: list[float], q1: list[float], homing_scale: float = 1e-2, acceleration_scale: float = 1e-1, accumulated_collisions: bool = True, quaternion_norms: bool = False):
        """
        Set up a one-phase fine-grained motion problem with second-order (acceleration) control costs.

        Args:
            q0 (list[float]): The initial configuration of the robot, represented as a list of joint values.
            q1 (list[float]): The target configuration of the robot, represented as a list of joint values.
            homing_scale (float, optional): The scaling factor for the homing control objective, encouraging the robot 
                                            to return to a home position. Default is 0.01.
            acceleration_scale (float, optional): The scaling factor for the acceleration control objective, 
                                                influencing the robot's movement acceleration. Default is 0.1.
            accumulated_collisions (bool, optional): If True, adds a constraint on accumulated collisions to minimize 
                                                    or avoid collisions between objects. Default is True.
            quaternion_norms (bool, optional): If True, adds constraints to normalize quaternions, ensuring valid 
                                            orientation representations. Default is False.

        """
        self.C.setJointState(q1)
        for frame in self.helpers:
            f = self.C.getFrame(f'_{frame}_end', False)
            if f:
                f_org = self.C.getFrame(frame)
                f.setPosition(f_org.getPosition())
                f.setQuaternion(f_org.getQuaternion())

        self.C.setJointState(q0)
        for frame in self.helpers:
            f = self.C.getFrame(f'_{frame}_start', False)
            if f:
                f_org = self.C.getFrame(frame)
                f.setPosition(f_org.getPosition())
                f.setQuaternion(f_org.getQuaternion())
        
        self.komo = ry.KOMO(self.C, 1.0, 32, 2, accumulated_collisions)
        self.komo.addControlObjective([], order=0, scale=homing_scale)
        self.komo.addControlObjective([], order=2, scale=acceleration_scale)
        self.komo.initWithWaypoints([q1], 1, interpolate=True, qHomeInterpolate=0.5, verbose=0)
        
        if quaternion_norms:
            self.komo.addQuaternionNorms()

        if accumulated_collisions:
            self.komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, scale=[1e0])

        # ensure zero velocity at the end of the motion
        self.komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, scale=[1e0], order=1)

        # ensure the end configuration matches the target configuration
        self.komo.addObjective([1.0], ry.FS.qItself, [], ry.OT.eq, scale=[1e0], target=q1)

    def setup_point_to_point_rrt(self, q0: list[float], q1: list[float], explicitCollisionPairs: list[str] = []):
        """
        Set up a point-to-point Rapidly-exploring Random Tree (RRT) motion planning problem between two configurations (q0 and q1).

        Args:
            q0 (list[float]): The initial configuration of the robot represented as a list of joint values.
            q1 (list[float]): The target configuration of the robot represented as a list of joint values.
            explicitCollisionPairs (list[str], optional): A list of pairs of objects that should be treated as colliding.
                                                        This can help the planner avoid known collision pairs.
                                                        Default is an empty list, meaning no explicit collision pairs are set.
        """
        self.rrt = ry.PathFinder()
        self.rrt.setProblem(self.C, q0, q1)
        
        if len(explicitCollisionPairs):
            self.rrt.setExplicitCollisionPairs(explicitCollisionPairs)

    def add_helper_frame(self, type: ry.JT, parent: str, name: str, initFrame: int):
        """
        Add a helper frame to the KOMO problem.

        Args:
            type (ry.JT): The type of joint associated with the frame (e.g., hingeX, transXY).
            parent (str): The name of the parent frame to which this helper frame will be attached.
            name (str): The unique name for the helper frame.
            initFrame (int): The initial time frame at which the helper frame is introduced.

        Details:
            - Adds a stable frame to the motion problem, which maintains its position relative to the parent frame over time.
            - Sets a marker shape with a size of 0.2 for visualization purposes.
            - Colors the frame marker magenta (RGB: [1, 0, 1]) for distinction.
        """
        f = self.komo.addStableFrame(name, parent, type, True, initFrame)
        f.setShape(ry.ST.marker, [.2])
        f.setColor([1., 0., 1.])
        #f.joint.sampleSdv=1.
        #f.joint.setRandom(self.komo.timeSlices.d1, 0)

    def grasp_top_box(self, time: float, gripper: str, obj: str, grasp_direction: str='xz'):
        """
        Grasp a box using a top-centered grasp with the gripper's axes fully aligned with the object's axes.

        Args:
            time (float): The time at which the grasp action will be executed.
            gripper (str): The name of the gripper that will perform the grasp.
            obj (str): The name of the object (box) to be grasped.
            grasp_direction (str, optional): Specifies the grasp direction by aligning specific axes of the gripper 
                                            and the object. Default is 'xz'.
                                            Possible values:
                                            - 'xz': Aligns XY, XZ, and YZ axes.
                                            - 'yz': Aligns YY, XZ, and YZ axes.
                                            - 'xy': Aligns XY, XZ, and ZZ axes.
                                            - 'zy': Aligns XX, XZ, and ZZ axes.
                                            - 'yx': Aligns YY, YZ, and ZZ axes.
                                            - 'zx': Aligns YX, YZ, and ZZ axes.

        Raises:
            Exception: If an invalid grasp_direction is provided.
        """

        if grasp_direction == 'xz':
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif grasp_direction == 'yz':
            align = [ry.FS.scalarProductYY, ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif grasp_direction == 'xy':
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'zy':
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'yx':
            align = [ry.FS.scalarProductYY, ry.FS.scalarProductYZ, ry.FS.scalarProductZZ]
        elif grasp_direction == 'zx':
            align = [ry.FS.scalarProductYX, ry.FS.scalarProductYZ, ry.FS.scalarProductZZ]
        else:
            raise Exception('pickDirection not defined:', grasp_direction)

        # position: centered
        self.komo.addObjective([time], ry.FS.positionDiff, [gripper, obj], ry.OT.eq, [1e1])

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], align[0], [obj, gripper], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [obj, gripper], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[2], [obj, gripper], ry.OT.eq, [1e0])

    def grasp_box(self, time: float, gripper: str, obj: str, palm: str, grasp_direction: str='x', margin: float=.02):
        """
        Perform a general grasp of a box by squeezing along the specified grasp axis (resulting in three possible grasps of a box),
        ensuring no collision with the palm. 
        Args:
            time (float): The time at which the grasp action will be executed.
            gripper (str): The name of the gripper that will perform the grasp.
            obj (str): The name of the object (box) to be grasped.
            palm (str): The name of the palm or part of the robot that should avoid collisions with the object.
            grasp_direction (str, optional): The axis along which the gripper should grasp the box. Default is 'x'.
                                            Possible values:
                                            - 'x': Grasp along the X-axis.
                                            - 'y': Grasp along the Y-axis.
                                            - 'z': Grasp along the Z-axis.
            margin (float, optional): The margin for no-collision constraints between the box and the palm. Default is 0.02.

        Raises:
            Exception: If an invalid grasp_direction is provided.

        Details:
            - The Angle of the grasp is decided by inequalities on the grasp plan.
            - The position and orientation objectives ensure that the gripper is centered on the box, and the grasp axis is orthogonal to the target plane.
            - The margin parameter is used to add tolerance to the no-collision constraints between the box and palm.

        """
        if grasp_direction == 'x':
            xLine = np.array([[1, 0, 0]])
            yzPlane = np.array([[0, 1, 0],[0, 0, 1]])
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductXZ]
        elif grasp_direction == 'y':
            xLine = np.array([[0, 1, 0]])
            yzPlane = np.array([[1, 0, 0],[0, 0, 1]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXZ]
        elif grasp_direction == 'z':
            xLine = np.array([[0, 0, 1]])
            yzPlane = np.array([[1, 0, 0],[0, 1, 0]])
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductXY]
        else:
            raise Exception('grasp_direction not defined:', grasp_direction)

        boxSize = self.C.frame(obj).getSize()[:3]

        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, xLine*1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane*1e1, .5*boxSize-margin)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, yzPlane*(-1e1), -.5*boxSize+margin)

        self.komo.addObjective([time-.2,time], align[0], [gripper, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [gripper, obj], ry.OT.eq, [1e0])

        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])

    def grasp_cylinder(self, time: float, gripper: str, obj: str, palm: str, margin: float=.02):
        """
        Perform a grasp of a cylinder by squeezing normally to the cylinder's axis, ensuring no collision with the palm.
        
        Args:
            time (float): The time at which the grasp action will be executed.
            gripper (str): The name of the gripper that will perform the grasp.
            obj (str): The name of the cylindrical object to be grasped.
            palm (str): The name of the palm or part of the robot that should avoid collisions with the object.
            margin (float, optional): The margin for no-collision constraints between the cylinder and the palm. Default is 0.02.

        Details:
            - Inequality constraint along the z-axis for positioning.

        """

        size = self.C.frame(obj).getSize()[:2]

        # position: center along axis, stay within z-range
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.eq, np.array([[1, 0, 0],[0, 1, 0]])*1e1)
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, np.array([[0, 0, 1]])*1e1, np.array([0.,0.,.5*size[0]-margin]))
        self.komo.addObjective([time], ry.FS.positionRel, [gripper, obj], ry.OT.ineq, np.array([[0, 0, 1]])*(-1e1), np.array([0.,0.,-.5*size[0]+margin]))

        # orientation: grasp axis orthoginal to target plane X-specific
        self.komo.addObjective([time-.2,time], ry.FS.scalarProductXZ, [gripper, obj], ry.OT.eq, [1e0])

        # no collision with palm
        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, obj], ry.OT.ineq, [1e1], [-.001])

    def place_box(self, time: float, obj: str, table: str, palm: str, place_direction: str='z', margin: float=.02):
        """
        Placement of one box onto another (named table) in a specific direction

        Args:
            time (float): The time at which the placement is executed.
            obj (str): The name of the object (box) to be placed.
            table (str): The name of the surface (table or another box) where the box will be placed.
            palm (str): The name of the palm or part of the robot that should avoid collisions during placement.
            place_direction (str, optional): The axis along which the box is placed on the surface. Default is 'z'.
                                            Possible values:
                                            - 'x': Place along the X-axis.
                                            - 'y': Place along the Y-axis.
                                            - 'z': Place along the Z-axis (default).
                                            - 'xNeg': Place along the negative X-axis.
                                            - 'yNeg': Place along the negative Y-axis.
                                            - 'zNeg': Place along the negative Z-axis.
            margin (float, optional): The margin to avoid collisions between the box and the table or other objects. Default is 0.02.
        """

        zVectorTarget = np.array([0.,0.,1.])
        boxSize = self.C.getFrame(obj).getSize()[:3]
        tableSize = self.C.getFrame(table).getSize()[:3]
        if place_direction == 'x':
            relPos = .5*(boxSize[0]+tableSize[2])
            zVector = ry.FS.vectorX
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductYX]
        elif place_direction == 'y':
            relPos = .5*(boxSize[1]+tableSize[2])
            zVector = ry.FS.vectorY
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductYY]
        elif place_direction == 'z':
            relPos = .5*(boxSize[2]+tableSize[2])
            zVector = ry.FS.vectorZ
            align = [ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]
        elif place_direction == 'xNeg':
            relPos = .5*(boxSize[0]+tableSize[2])
            zVector = ry.FS.vectorX
            zVectorTarget *= -1.
            align = [ry.FS.scalarProductXX, ry.FS.scalarProductYX]
        elif place_direction == 'yNeg':
            relPos = .5*(boxSize[1]+tableSize[2])
            zVector = ry.FS.vectorY
            zVectorTarget *= -1.
            align = [ry.FS.scalarProductXY, ry.FS.scalarProductYY]
        elif place_direction == 'zNeg':
            relPos = .5*(boxSize[2]+tableSize[2])
            zVector = ry.FS.vectorZ
            zVectorTarget *= -1.
            align = [ry.FS.scalarProductXZ, ry.FS.scalarProductYZ]

        self.komo.addObjective([time], ry.FS.positionDiff, [obj, table], ry.OT.eq, 1e1*np.array([[0, 0, 1]]), np.array([.0, .0, relPos]))
        self.komo.addObjective([time], ry.FS.positionRel, [obj, table], ry.OT.ineq, 1e1*np.array([[1, 0, 0],[0, 1, 0]]), .5*tableSize-margin)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, table], ry.OT.ineq, -1e1*np.array([[1, 0, 0],[0, 1, 0]]), -.5*tableSize+margin)

        self.komo.addObjective([time-.2, time], zVector, [obj], ry.OT.eq, [0.5], zVectorTarget)
        self.komo.addObjective([time-.2,time], align[0], [table, obj], ry.OT.eq, [1e0])
        self.komo.addObjective([time-.2,time], align[1], [table, obj], ry.OT.eq, [1e0])

        self.komo.addObjective([time-.3,time], ry.FS.distance, [palm, table], ry.OT.ineq, [1e1], [-.001])

    def straight_push(self, times: list[float], obj: str, gripper: str, table: str):
        """
        Define a straight push motion for the gripper to push an object across a table.

        Args:
            times (list[float]): A list of two time points specifying the start and end times for the pushing motion.
            obj (str): The name of the object being pushed.
            gripper (str): The name of the gripper that will perform the push.
            table (str): The name of the table where the push occurs.

        Behavior:
            - Adds two helper frames ('_push_start' and '_push_end') attached to the table and object to define the
            start and end points of the pushing motion.
            - Ensures the start and end frames are aligned in both orientation and position, imposing constraints on
            their alignment and ensuring a minimum distance between them.
            - Ensures the gripper is in contact with the object and aligns it with the start position at the beginning
            of the motion.
            - The object is constrained to follow a straight path and maintain its orientation at the end of the push.
        """ 
        
        #start & end helper frames
        self.add_helper_frame(ry.JT.hingeZ, table, '_push_start', obj)
        self.add_helper_frame(ry.JT.transXYPhi, table, '_push_end', obj)

        #-- couple both frames symmetricaly
        #aligned orientation
        self.komo.addObjective([times[0]], ry.FS.vectorYDiff, ['_push_start', '_push_end'], ry.OT.eq, [1e1])
        #aligned position
        self.komo.addObjective([times[0]], ry.FS.positionRel, ['_push_end', '_push_start'], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        self.komo.addObjective([times[0]], ry.FS.positionRel, ['_push_start', '_push_end'], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        #at least 2cm appart, positivenot !not  direction
        self.komo.addObjective([times[0]], ry.FS.positionRel, ['_push_end', '_push_start'], ry.OT.ineq, -1e2*np.array([[0., 1., 0.]]), [.0, .02, .0])
        self.komo.addObjective([times[0]], ry.FS.positionRel, ['_push_start', '_push_end'], ry.OT.ineq, 1e2*np.array([[0., 1., 0.]]), [.0, -.02, .0])

        #gripper touch
        self.komo.addObjective([times[0]], ry.FS.negDistance, [gripper, obj], ry.OT.eq, [1e1], [-.02])
        #gripper start position
        self.komo.addObjective([times[0]], ry.FS.positionRel, [gripper, '_push_start'], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 0., 1.]]))
        self.komo.addObjective([times[0]], ry.FS.positionRel, [gripper, '_push_start'], ry.OT.ineq, 1e1*np.array([[0., 1., 0.]]), [.0, -.02, .0])
        #gripper start orientation
        self.komo.addObjective([times[0]], ry.FS.scalarProductYY, [gripper, '_push_start'], ry.OT.ineq, [-1e1], [.2])
        self.komo.addObjective([times[0]], ry.FS.scalarProductYZ, [gripper, '_push_start'], ry.OT.ineq, [-1e1], [.2])
        self.komo.addObjective([times[0]], ry.FS.vectorXDiff, [gripper, '_push_start'], ry.OT.eq, [1e1])

        #obj end position
        self.komo.addObjective([times[1]], ry.FS.positionDiff, [obj, '_push_end'], ry.OT.eq, [1e1])
        #obj end orientation: unchanged
        self.komo.addObjective([times[1]], ry.FS.quaternion, [obj], ry.OT.eq, [1e1], [], 1); #qobjPose.rot.getArr4d())

    def pull(self, times: list[float], obj: str, gripper: str, table: str):
        """
        Define a pulling motion where the gripper pulls an object along the table surface while maintaining a fixed downward orientation.

        Args:
            times (list[float]): A list of two time points specifying the start and end times for the pulling motion.
            obj (str): The name of the object being pulled.
            gripper (str): The name of the gripper performing the pull.
            table (str): The name of the table or surface on which the object is being pulled.

        Behavior:
            - Adds a helper frame (`_pull_end`) attached to the table and object to define the endpoint of the pulling motion.
            - Ensures both the gripper and object maintain a fixed downward orientation (along the Z-axis) throughout the motion.
            - Ensures the gripper is in contact with the object at the start of the pulling motion.
            - Constrains the object to follow the gripper and reach the target position (`_pull_end`) at the end of the motion.
        """

        self.add_helper_frame(ry.JT.transXYPhi, table, '_pull_end', obj)
        
        self.komo.addObjective([times[0]], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[1]], ry.FS.vectorZ, [gripper], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[0]], ry.FS.vectorZ, [obj], ry.OT.eq, [1e1], np.array([0,0,1]))
        self.komo.addObjective([times[1]], ry.FS.vectorZ, [obj], ry.OT.eq, [1e1], np.array([0,0,1]))

        self.komo.addObjective([times[0]], ry.FS.positionRel, [gripper, obj], ry.OT.eq, 1e1*np.array([[1., 0., 0.], [0., 1., 0.]]), np.array([0, 0, 0]))
        self.komo.addObjective([times[0]], ry.FS.negDistance, [gripper, obj], ry.OT.eq, [1e1], [-.005])
        self.komo.addObjective([times[1]], ry.FS.positionDiff, [obj, '_pull_end'], ry.OT.eq, [1e1])

    def follow_path_on_plane_xy(self, path: list[list[float]], moving_frame: str):
        """
        This function assumes the the robot is already at the starting position!
        Move through the 2D points defined in the path while staying on the plane specified.

        TODO: Take direction of plane normal into account.
        """
        start_pos = self.C.getFrame(moving_frame).getPosition()
        plane_pos = np.array([0, 0, start_pos[2]])
        threed_path = []
        for p in path:
            threed_path.append([p[0], p[1], 0])
        
        imp_axis = np.array([0., 0., 1.])
        
        phases = len(path)-1

        self.komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
        self.komo.addObjective([1, phases], ry.FS.vectorZ, [moving_frame], ry.OT.eq, [1e1], imp_axis)
        self.komo.addObjective([1, phases], ry.FS.position, [moving_frame], ry.OT.eq, imp_axis*1e1, plane_pos)

        for i in range(1, len(path)):
            self.target_xy_position(i, moving_frame, threed_path[i])

    def path_must_be_straight(self, times: list[float], start_frame: str, end_frame: str, moving_frame: str, gotoPoints: bool=False):
        """
        Constrain the motion of a moving frame to follow a straight path between start and end during the.

        Args:
            times (list[float]): A list of times that define the duration of the motion constraint.
            start_frame (str): The name of the starting frame for the motion.
            end_frame (str): The name of the target frame for the motion.
            moving_frame (str): The name of the frame that is moving and must follow the straight path.
            gotoPoints (bool, optional): If True, additional constraints are added to ensure that the moving frame is exactly
                                        at the start and end points at the beginning and end of the time period.
                                        Default is False.
        """

        delta = self.C.getFrame(end_frame).getPosition() - \
            self.C.getFrame(start_frame).getPosition()
        
        delta /= np.linalg.norm(delta)
        mat = np.eye(3) - np.outer(delta, delta)

        self.komo.addObjective(times, ry.FS.positionDiff, [
                          moving_frame, start_frame], ry.OT.eq, mat)
        
        if gotoPoints:
            self.komo.addObjective([times[0]], ry.FS.positionDiff, [
                                moving_frame, start_frame], ry.OT.eq, [1e1])
            self.komo.addObjective([times[1]], ry.FS.positionDiff, [
                                moving_frame, end_frame], ry.OT.eq, [1e1])

    def keep_distance(self, time_interval: list[float], obj1: str, obj2: str, margin: float=.001):
        """
        Impose an inequality constraint on the distance between two objects over a given time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times during which
                                        the distance constraints between objects are enforced.
            objs (list[str]): A list of object names or identifiers. The method ensures that all pairs of objects
                            maintain a minimum distance from each other.
            margin (float, optional): The minimum allowed distance between the objects. Default is 0.001 meters.
        """
        self.komo.addObjective(time_interval, ry.FS.negDistance, [obj1, obj2], ry.OT.ineq, [1e1], [-margin])

    def keep_distances(self, time_interval: list[float], objs: list[str], margin: float = 0.001):
        """
        Impose an inequality constraint on the distance between multiple objects over a given time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times during which
                                        the distance constraints between objects are enforced.
            objs (list[str]): A list of object names or identifiers. The method ensures that all pairs of objects
                            maintain a minimum distance from each other.
            margin (float, optional): The minimum allowed distance between the objects. Default is 0.001 meters.
        """
        while len(objs) > 1:
            comp = objs[0]
            del objs[0]
            for obj in objs:
                self.komo.addObjective(time_interval, ry.FS.negDistance, [comp, obj], ry.OT.ineq, [1e1], [-margin])

    def set_relative_distance(self, time: float, obj1: str, obj2: str, distance: float):
        """
        Impose an inequality constraint on the relative distance between two objects at a specific time.

        Args:
            time (float): The time at which the distance constraint between the two objects is enforced.
            obj1 (str): The name of the first object.
            obj2 (str): The name of the second object.
            distance (float): The desired distance between the two objects, where the actual distance must not exceed this value.
        """
        self.komo.addObjective([time], ry.FS.negDistance, [obj1, obj2], ry.OT.eq, [1e1], [-distance])

    def switch_pick(self, time_interval: list[float], gripper: str, obj: str):
        """
        Perform a kinematic mode switch, attaching the object to the gripper with a freely parameterized but stable (i.e. constant)
        relative pose during a specified time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times during which the mode switch occurs.
            gripper (str): The name of the gripper that will pick up the object.
            obj (str): The name of the object to be picked up and attached to the gripper.
        """
        self.komo.addModeSwitch(time_interval, ry.SY.stable, [gripper, obj])

    def switch_place(self, time_interval: list[float], table: str, obj: str):
        """
        TODO: a kinematic mode switch, where obj becomes attached to table, with a 3D parameterized (XYPhi) stable relative pose
        this requires obj and table to be boxes and assumes default placement alone z-axis
        more general placements have to be modelled with switch_pick (table picking the object) and additinal user-defined geometric constraints
        """
        self.komo.addModeSwitch(time_interval, ry.SY.stable, [table, obj])

    def target_position(self, time: float, obj: str, pos: list[float]):
        """
        Impose a specific 3D target position on the specified object at a given time.

        Args:
            time (float): The time at which the position constraint is applied.
            obj (str): The name of the object whose position is being constrained.
            pos (list[float]): A list of three floats representing the target 3D position (x, y, z) of the object.
        """
        self.komo.addObjective([time], ry.FS.position, [obj], ry.OT.eq, [1e1], pos)

    def target_xy_position(self, time: float, obj: str, pos: list[float]):
        """
        Impose a specific 2D target position (x, y) on the specified object at a given time.

        Args:
            time (float): The time at which the position constraint is applied.
            obj (str): The name of the object whose position is being constrained.
            pos (list[float]): A list of two or three floats representing the target 2D position (x, y).
                            If only two values are provided, the z-component is set to 0.
                            
        """
        if len(pos)==2:
            pos.append(0.)
        self.komo.addObjective([time], ry.FS.position, [obj], ry.OT.eq, 1e1*np.array([[1,0,0],[0,1,0]]), pos)
    
    def target_relative_xy_position(self, time: float, obj: str, relativeTo: str, pos: list[float]):
        """
        Impose a specific 3D target position on an object relative to another frame at a given time.

        Args:
            time (float): The time at which the position constraint is applied.
            obj (str): The name of the object whose position is being constrained.
            relativeTo (str): The name of the reference frame relative to which the object's position is defined.
            pos (list[float]): A list of two or three floats representing the target position relative to `relativeTo`.
                            If only two values are provided, the z-component is set to 0.
        """
        if len(pos)==2:
            pos.append(0.)
        self.komo.addObjective([time], ry.FS.positionRel, [obj, relativeTo], ry.OT.eq, scale=1e1*np.array([[1,0,0],[0,1,0]]), target=pos)
    
    def target_x_orientation(self, time: float, obj: str, x_vector: list[float]):
        """
        Align the x-axis orientation of a specified object with a target vector at a given time.

        Args:
            time (float): The specific time at which the orientation constraint is applied.
            obj (str): The name of the object whose z-axis orientation is being constrained.
            z_vector (list[float]): A 3D vector representing the desired orientation of the object's x-axis.
                                    This vector defines the target direction for the object's x-axis.
        """
        self.komo.addObjective([time], ry.FS.vectorX, [obj], ry.OT.eq, scale=[1e1], target=x_vector)

    def target_y_orientation(self, time: float, obj: str, y_vector: list[float]):
        """
        Align the y-axis orientation of a specified object with a target vector at a given time.

        Args:
            time (float): The specific time at which the orientation constraint is applied.
            obj (str): The name of the object whose z-axis orientation is being constrained.
            z_vector (list[float]): A 3D vector representing the desired orientation of the object's y-axis.
                                    This vector defines the target direction for the object's y-axis.
        """
        self.komo.addObjective([time], ry.FS.vectorY, [obj], ry.OT.eq, scale=[1e1], target=y_vector)

    def target_z_orientation(self, time: float, obj: str, z_vector: list[float]):
        """
        Align the z-axis orientation of a specified object with a target vector at a given time.

        Args:
            time (float): The specific time at which the orientation constraint is applied.
            obj (str): The name of the object whose z-axis orientation is being constrained.
            z_vector (list[float]): A 3D vector representing the desired orientation of the object's z-axis.
                                    This vector defines the target direction for the object's z-axis.
        """
        self.komo.addObjective([time], ry.FS.vectorZ, [obj], ry.OT.eq, scale=[1e1], target=z_vector)

    def bias(self, time: float, qBias: list[float], scale: float = 1.0):
        """
        Apply a square potential bias directly in joint space.

        Args:
            time (float): The time at which to impose the bias.
            qBias (list[float]): A list of target joint angles or positions. This represents the desired configuration for the robot's joints.
            scale (float, optional): The scaling factor for the bias. This controls the strength of the imposed bias.
                                    Default value is 1.0.

        Details:
            - This method adds an objective to the optimization problem to steer the system towards the specified joint configuration `qBias`.
        """
        self.komo.addObjective([time], ry.FS.qItself, [], ry.OT.sos, scale=scale, target=qBias)

    def retract(self, time_interval: list[float], gripper: str, dist: float = 0.05):
        """
        Define a retract motion for a specified gripper over a time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times of the retract motion.
            gripper (str): The name of the gripper that will perform the retract motion.
            dist (float, optional): The distance to retract relative to the grippers z-axis. 
                                    Default value is 0.05 meters.
        """
        helper = f'_{gripper}_start'

        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1, 0, 0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, 
                            -1e2 * np.array([[0, 0, 1]]), target=[0., 0., dist])

    def approach(self, time_interval: list[float], gripper: str, dist: float = 0.05):
        """
        Define an approach motion for a specified gripper over a time interval.

        Args:
            time_interval (list[float]): A list with two elements specifying the start and end times of the approach motion.
            gripper (str): The name of the gripper that will perform the approach motion.
            dist (float, optional): The distance to approach along the z-axis (relative to the gripper's final position).
                                    Default value is 0.05 meters.
        """
        helper = f'_{gripper}_end'

        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, 1e2 * np.array([[1, 0, 0]]))
        self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, 
                            -1e2 * np.array([[0, 0, 1]]), target=[0., 0., dist])

    def retractPush(self, time_interval: list[float], gripper: str, dist: float=.03):
        helper = f'_{gripper}_start'
        #  self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[1,3]),{1,0,0]})
        #  self.komo.addObjective(time_interval, ry.FS.quaternionDiff, [gripper, helper], ry.OT.eq, [1e2])
        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[1, 0, 0]]))
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, * np.array([[0, 1, 0]]), [0., -dist, 0.])
        self.komo.addObjective([time_interval[1]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), [0., 0., dist])

    def approachPush(self, time_interval: list[float], gripper: str, dist: float=.03):
        #  if not helper.N) helper = STRING("_push_start":
        #  self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[2,3]),{1,0,0,0,0,1]})
        #  self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, * np.array([[1,3]),{0,1,0]}, [0., -dist, 0.])
        helper = f'_{gripper}_end'
        self.komo.addObjective(time_interval, ry.FS.positionRel, [gripper, helper], ry.OT.eq, * np.array([[1, 0, 0]]))
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, * np.array([[0, 1, 0]]), [0., -dist, 0.])
        self.komo.addObjective([time_interval[0]], ry.FS.positionRel, [gripper, helper], ry.OT.ineq, -1e2 * np.array([[0, 0, 1]]), [0., 0., dist])
        
    def solve(self, verbose: int=1) -> list[list[float]]:
        """   
        Finding a feasible path or trajectory by solving a nonlinear optimization problem formulated in KOMO, or using RRT, respectively.
        
        Args:
            verbose (int, optional): The verbosity level for logging the process.
                                    - 0: No output.
                                    - 1: Basic output about feasibility.
                                    - 2: Detailed output, including solver reports and failure visualization.
                                    - 3: Full output, including real-time path playback. Default is 1.
        
        Returns:
            list[list[float]]: The computed path or trajectory as a list of 3D points, if a solution is found. 
                            Returns `None` if the optimization fails or no problem is defined.
        """
        if self.komo:
            sol = ry.NLP_Solver()
            sol.setProblem(self.komo.nlp())
            sol.setOptions(damping=1e-3, verbose=verbose-1, stopTolerance=1e-3, maxLambda=100., stopEvals=200)
            self.ret = sol.solve()
            if self.ret.feasible:
                self.path = self.komo.getPath()
            else:
                self.path = None
            if verbose>0:
                if not self.ret.feasible:
                    print(f'  -- infeasible:{self.info}\n     {self.ret}')
                    if verbose>1:
                        print(self.komo.report(False, True))
                        self.komo.view(True, f"failed: {self.info}\n{self.ret}")
                    if verbose>2:
                        while(self.komo.view_play(True, 1.)):
                            pass
                else:
                    print(f'  -- feasible:{self.info}\n     {self.ret}')
                    if verbose>2:
                        self.komo.view(True, f"success: {self.info}\n{self.ret}")
                    if verbose>3:
                        while(self.komo.view_play(True, 1.)):
                            pass

        elif self.rrt:
            self.ret = self.rrt.solve()
            if self.ret.feasible:
                self.path = self.ret.x
            else:
                self.path = None

        else:
            print('no problem defined')
            
        return self.path

    def play(self, C: ry.Config, duration: float = 1.):
        """
        Play back a trajectory by setting joint states at each step.

        Args:
            C (ry.Config): The configuration object representing the robot's state.
            duration (float, optional): The total duration for playing back the trajectory.
                                        The default is 1 second.
        """
        for t in range(self.path.shape[0]):
            C.setJointState(self.path[t])
            C.view(False, f'step {t}\n{self.info}')
            time.sleep(duration / self.path.shape[0])

    def sub_motion(self, phase: int, homing_scale: float=1e-2, acceleration_scale: float=1e-1, accumulated_collisions: bool=True, quaternion_norms: bool=False) -> 'ManipulationModelling':
        """
        Create a sub-motion plan for a specific phase using KOMO and return a ManipulationModelling instance.

        Args:
            phase (int): The phase number for which the sub-motion is to be planned. .
            homing_scale (float, optional): The scaling factor for the homing (returning to a predefined posture) objective. Default is 1e-2.
            acceleration_scale (float, optional): The scaling factor for the acceleration minimization objective. Default is 1e-1.
            accumulated_collisions (bool, optional): If True, enables accumulated collision constraints during the sub-motion planning. Default is True.
            quaternion_norms (bool, optional): If True, enforces quaternion normalization during the motion (useful for rotational consistency). Default is False.

        Returns:
            ManipulationModelling: A new instance of the ManipulationModelling class, configured for the sub-motion plan of the given phase.
        """
        (C, q0, q1) = self.komo.getSubProblem(phase)
        manip = ManipulationModelling(C, f'sub_motion_{phase}--{self.info}', self.helpers)
        manip.setup_point_to_point_motion(q0, q1, homing_scale, acceleration_scale, accumulated_collisions, quaternion_norms)
        return manip

    def sub_rrt(self, phase: int, explicitCollisionPairs: list[str]=[]) -> 'ManipulationModelling':
        """
        Create a sub-motion plan for a specific phase using RRT (Rapidly-exploring Random Trees) and return a ManipulationModelling instance.

        Args:
            phase (int): The phase number for which the sub-motion is to be planned.
            explicitCollisionPairs (list[str], optional): A list of object pairs for which explicit collision avoidance should be enforced. Default is an empty list.

        Returns:
            ManipulationModelling: A new instance of the ManipulationModelling class, configured for the sub-motion plan of the given phase.
        """
        (C, q0, q1) = self.komo.getSubProblem(phase)
        manip = ManipulationModelling(C, f'sub_rrt_{phase}--{self.info}', self.helpers)
        manip.setup_point_to_point_rrt(q0, q1, explicitCollisionPairs)
        return manip
    
    @property
    def feasible(self) -> bool:
        return self.ret.feasible
    