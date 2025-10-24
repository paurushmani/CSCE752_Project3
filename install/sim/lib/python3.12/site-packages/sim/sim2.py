# sim2
#    A simple simulator that tracks a robot's position as it changes in
#    response to twist commands, publishing a marker array showing its past
#    locations and tracking which of a collection of segments the robot has
#    crossed.

__author__ = "Jason M. O'Kane"
__copyright__ = "Copyright 2025"

import math
import os
import random

import numpy as np
import yaml

import rclpy.node
import rclpy.qos
import sensor_msgs.msg
import std_msgs.msg
import std_srvs.srv
import tf2_ros
import ament_index_python.packages 
import geometry_msgs.msg
import visualization_msgs.msg

PACKAGE_NAME = 'sim'
ROBOT_RADIUS = 0.5
TARGET_RADIUS = 0.05

def euler_to_quaternion(r, p, y):
    # This is the standard formula, which seems not to be included directly in ROS2 for some reason.
    # See also: https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/Using-URDF-with-Robot-State-Publisher.html
    return geometry_msgs.msg.Quaternion(x=math.sin(r/2)*math.cos(p/2)*math.cos(y/2) - math.cos(r/2)*math.sin(p/2)*math.sin(y/2),
                                        y=math.cos(r/2)*math.sin(p/2)*math.cos(y/2) + math.sin(r/2)*math.cos(p/2)*math.sin(y/2),
                                        z=math.cos(r/2)*math.cos(p/2)*math.sin(y/2) - math.sin(r/2)*math.sin(p/2)*math.cos(y/2),
                                        w=math.cos(r/2)*math.cos(p/2)*math.cos(y/2) + math.sin(r/2)*math.sin(p/2)*math.sin(y/2))

def yaml_contents_from_parameter(node, parameter_name, default_parameter=''):
    # Read a YAML file named in a given parameter and return its parsed
    # contents.
    #
    # The file to read can be specified in any of three ways:
    # - Directly as a relative or absolute path.
    # - In the form "<package_name>:<file_name>", to read from the share
    # directory of the named package.
    # - As the name of a file in the share directory of the 'sim' package.
    #

    # Get the parameter value.
    node.declare_parameter(parameter_name, default_parameter)
    value = node.get_parameter(parameter_name).get_parameter_value().string_value

    if value == '':
        raise ValueError(f'No value given for parameter: {parameter_name}')
    
    if os.path.exists(value):
        fullpath = value
    else:
        if ':' in value:
            package, filename = value.split(':')
        else:
            package = 'sim'
            filename = value

        dir = ament_index_python.packages.get_package_share_directory(package)
        fullpath = os.path.join(dir, filename)

    node.get_logger().info(f'Reading {parameter_name} from {fullpath}')
    with open(fullpath) as f:
        yaml_text = f.read()
    return yaml.safe_load(yaml_text)

def point_segment_distance(p, s):
    """How far is the given point from the given segment?"""
    d = (s[1][0] - s[0][0], s[1][1] - s[0][1])
    try:
        a = min(1, max(0, ((p[0] - s[0][0]) * d[0] + (p[1] - s[0][1]) * d[1]) / (d[0] * d[0] + d[1] * d[1])))
    except ZeroDivisionError:
        a = 0
    q = (s[0][0] + a * d[0], s[0][1] + a * d[1])
    return math.hypot(p[0]-q[0], p[1]-q[1])

def cw(a, b, c):
    """Are the three points given in clockwise order?"""
    return (b[1]-a[1])*(c[0]-b[0])-(b[0]-a[0])*(c[1]-b[1]) > 0

def segments_intersect(s1, s2):
    """Do the segments intersect?"""
    return (cw(s1[0], s1[1], s2[0]) != cw(s1[0], s1[1], s2[1]) 
        and cw(s2[0], s2[1], s1[0]) != cw(s2[0], s2[1], s1[1]))
 
class Sim2_Node(rclpy.node.Node):
    def __init__(self):
        # Initialize the node itself.
        super().__init__('sim2')

        # Acquire a list of the target positions that the robot should visit.
        target_list = yaml_contents_from_parameter(self, 'targets')
        try:
            self.targets = { ((target[0][0], target[0][1]),(target[1][0], target[1][1])) : False for target in target_list }
        except (TypeError, IndexError):
            raise ValueError(f'Target list is not valid: {target_list}')

        # We'll need this to broadcast tf transforms.
        qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.transform_broadcaster = tf2_ros.TransformBroadcaster(self, 10)
        self.static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster(self, qos) 

        # Get ready to publish the robot description, but only very
        # occasionally.  Also publish it once right at the start, to eliminate
        # delays in showing the robot model when we're starting up.
        self.robot_description_publisher = self.create_publisher(std_msgs.msg.String, 'robot_description', qos)
        self.robot_description_timer = self.create_timer(0.5, self.publish_robot_description)
        self.publish_robot_description()

        # Get ready to publish a marker array showing the robot's path and the targets.
        self.markers_publisher = self.create_publisher(visualization_msgs.msg.MarkerArray, '/sim_markers', 1)

        # Initialize the position, history, commands, etc.
        self.reset()

        # Listen for twists to be published.
        self.create_subscription(geometry_msgs.msg.Twist, 'cmd_vel', self.receive_twist, 1)

        # Arrange to update the position periodically.
        self.create_timer(0.1, self.update_pose)

        # Arrange to publish the updated pose periodically.
        self.create_timer(0.1, self.broadcast_pose_transform)

        self.past_locations_timer = self.create_timer(0.2, self.publish_markers)

        # Get ready to publish the robot's pose.
        self.pose_publisher = self.create_publisher(geometry_msgs.msg.Pose2D, '/pose', 1)

        # Get ready to publish the targets that remain to be visited.
        self.target_publisher = self.create_publisher(sensor_msgs.msg.PointCloud, '/unvisited_targets', 1)
        self.create_timer(1, self.publish_unvisited_targets)

        # Start services to reset the simulation and toggle the recording.
        self.reset_srv = self.create_service(std_srvs.srv.Empty, 'reset', self.reset)
        self.set_pen_srv = self.create_service(std_srvs.srv.SetBool, 'set_pen', self.set_pen)


    def receive_twist(self, msg):
        # A twist has been published.  Keep track of it for simulating the
        # robot's future motion.
        self.twist = msg
        self.time_of_last_twist_command = self.get_clock().now()

    def update_pose(self):
        # Change the robot's pose based on the commanded velocity and the
        # elapsed time.
        verbose = False
        
        # Expire the movement command if it is too old.
        if self.get_clock().now()-self.time_of_last_twist_command > rclpy.duration.Duration(seconds = 1.0):
            self.twist = geometry_msgs.msg.Twist()

        # Compute the linear and angular velocities.
        linear = self.twist.linear.x * self.linear_bias
        angular = self.twist.angular.z * self.angular_bias
        linear = max(-2.0, min(2.0, linear))
        angular = max(-2.0, min(2.0, angular))

        # How much time are we simulating?
        now = self.get_clock().now()
        seconds_since_start = (now - self.start_time).nanoseconds * 1e-9
        dt = seconds_since_start - self.total_simulated_time
        if verbose:
            print(f"seconds_since_start={seconds_since_start} total_simulated_time={self.total_simulated_time} dt={dt}")
        self.total_simulated_time += dt

        # Move by this amount of time.
        position_before = (self.pose[0], self.pose[1])
        self.pose[0] += math.cos(self.pose[2]) * linear * dt
        self.pose[1] += math.sin(self.pose[2]) * linear * dt
        self.pose[2] += angular * dt
        position_after = (self.pose[0], self.pose[1])

        # # Check whether we've reached any targets.
        if position_before != position_after:
            for target in self.targets.keys():
                if self.targets[target]: continue
                if (segments_intersect((position_before, position_after), target)
                    or point_segment_distance(position_after, target) < 0.05):
                    self.targets[target] = True

        # Record the movement for publishing later.
        if self.recording and self.twist.linear.x != 0:
            self.past_locations[-1].append(self.pose[0:2].copy())

        # Publish the new pose.
        msg = geometry_msgs.msg.Pose2D()
        msg.x = self.pose[0]
        msg.y = self.pose[1]
        msg.theta = self.pose[2]
        self.pose_publisher.publish(msg)

        if verbose:
            print('twist:', self.twist)
            print('new pose:', self.pose)


    def broadcast_pose_transform(self):
        # Broadcast the current pose in the form of a transform from the world
        # frame to the base_link frame.  Used, for example, by rviz for
        # visualizing the robot.
        transform = geometry_msgs.msg.Transform()
        transform.translation.x = self.pose[0]
        transform.translation.y = self.pose[1]
        transform.rotation = euler_to_quaternion(0, 0, self.pose[2])

        msg = tf2_ros.TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.child_frame_id = 'base_link'
        msg.transform = transform
        self.transform_broadcaster.sendTransform(msg)
    
    def publish_robot_description(self):
        # Do the work needed to let rviz know how to show the robot in the
        # correct place.  This includes publishing the URDF for the robot on
        # /robot_description and broadcasting the (static) transform for the
        # heading box.  This is all work usually done by robot_state_publisher,
        # but doing it here keeps the simulator within a single node.

        radius = ROBOT_RADIUS
        height = 0.1
        offset = 0.45*radius

        urdf = f"""<?xml version="1.0"?>
                   <robot name="disc">
                       <material name="light_blue"><color rgba="0.5 0.5 1 1"/></material>
                       <material name="dark_blue"><color rgba="0.1 0.1 1 1"/></material>
                       <material name="dark_red"><color rgba="1 0.1 0.1 1"/></material>
                       <link name="base_link">
                           <visual>
                               <geometry><cylinder length="{height}" radius="{radius}"/></geometry>
                               <material name="light_blue"/>
                           </visual>
                       </link>
                       <link name="heading_box">
                           <visual>
                               <geometry><box size="{0.9*radius} {0.2*radius} {1.2*height}"/></geometry>
                               <material name="dark_blue"/>
                           </visual>
                       </link>
                       <joint name="base_to_heading_box" type="fixed">
                           <parent link="base_link"/>
                           <child link="heading_box"/>
                           <origin xyz="{offset} 0.0 0.0"/>
                       </joint>
                   </robot>
                   """
        msg = std_msgs.msg.String(data=urdf)
        self.robot_description_publisher.publish(msg)

        transform = geometry_msgs.msg.Transform()
        transform.translation.x = offset
        transform.rotation = euler_to_quaternion(0, 0, 0)

        msg = tf2_ros.TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.child_frame_id = 'heading_box'
        msg.transform = transform
        self.static_transform_broadcaster.sendTransform(msg)

    def publish_markers(self):
        # Create and publish a MarkerArray of things we want to visualize.

        # - Past locations.
        markers = visualization_msgs.msg.MarkerArray()

        path_color = std_msgs.msg.ColorRGBA(r=80/255, g=0.0, b=0.0, a=1.0)

        for i, location_chain in enumerate(self.past_locations):
            path_marker = visualization_msgs.msg.Marker()
            path_marker.header.frame_id = 'world'
            path_marker.ns = 'trace'
            path_marker.scale = geometry_msgs.msg.Vector3(x=0.15, y=0.0, z=0.0)
            path_marker.id = 100+i
            path_marker.type = visualization_msgs.msg.Marker.LINE_STRIP
            for location in location_chain:
                path_marker.points.append(geometry_msgs.msg.Point(x=location[0], y=location[1], z=-0.1))
                path_marker.colors.append(path_color)
            markers.markers.append(path_marker)

        # - Targets
        target_colors = { False: std_msgs.msg.ColorRGBA(r=0.0, g=180/255, b=180/255, a=1.0),
                          True:  std_msgs.msg.ColorRGBA(r=180/255, g=180/255, b=0.0, a=1.0) }

        target_marker = visualization_msgs.msg.Marker()
        target_marker.header.frame_id = 'world'
        target_marker.ns = 'targets'
        target_marker.scale = geometry_msgs.msg.Vector3(x=2*TARGET_RADIUS, y=0.0, z=0.0)
        target_marker.id = 20000
        target_marker.type = visualization_msgs.msg.Marker.LINE_LIST
        for i, (target, reached) in enumerate(self.targets.items()):
            target_marker.points.append(geometry_msgs.msg.Point(x=target[0][0], y=target[0][1], z=0.2))
            target_marker.points.append(geometry_msgs.msg.Point(x=target[1][0], y=target[1][1], z=0.2))
            target_marker.colors.append(target_colors[reached])
            target_marker.colors.append(target_colors[reached])
        markers.markers.append(target_marker)

        self.markers_publisher.publish(markers)

    def reset(self, req=None, resp=None):
        # Put everything into an initial state.

        # The current pose of the robot, expressed as (x, y, theta).
        self.pose = np.array([0, 0, 0], dtype=float)

        # The current commanded twist, i.e. angular and linear velocities.
        self.twist = geometry_msgs.msg.Twist()

        # When did the last commanded twist arrive? Used to timeout each
        # command after about a second.
        self.time_of_last_twist_command = self.get_clock().now()

        # When did we start, in real time? How much time have we simulated so
        # far?  The difference between these will tell us the amount of time to
        # use for each step of the simulation.
        self.start_time = self.get_clock().now()
        self.total_simulated_time = 0.0
        
        # Where have we been before?
        self.past_locations = [ ]

        # We want to record the movement commands.
        self.set_pen(std_srvs.srv.SetBool.Request(data=True), std_srvs.srv.SetBool.Response())

        # None of the targets have been reached yet.
        for target in self.targets.keys():
            self.targets[target] = False

        # Clear any markers that RViz might be displaying.
        markers = visualization_msgs.msg.MarkerArray()
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = 'world'
        marker.action = marker.DELETEALL
        markers.markers.append(marker)
        self.markers_publisher.publish(markers)

        # Small random biases on the robot's motion.
        self.linear_bias = 0.9 + 0.2*random.random()
        self.angular_bias = 0.9 + 0.2*random.random()

        # Done.  Send the response object back.
        return resp

    def publish_unvisited_targets(self):
        cloud = sensor_msgs.msg.PointCloud()
        cloud.header.frame_id = 'world'
        for target, visited in self.targets.items():
            if not visited:
                cloud.points.append(geometry_msgs.msg.Point32(x=target[0][0], y=target[0][1], z=0.0))
                cloud.points.append(geometry_msgs.msg.Point32(x=target[1][0], y=target[1][1], z=0.0))
        self.target_publisher.publish(cloud)

    def set_pen(self, req, resp):
        # Start or stop appending to the history.
        self.recording = req.data
        if self.recording:
            self.past_locations.append([ self.pose[0:2].copy() ])
        resp.success=True
        return resp

def main():
    rclpy.init()
    x = Sim2_Node()
    rclpy.spin(x)

if __name__ == '__main__':
    main()

