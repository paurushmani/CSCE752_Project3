# sim1
#    A simple simulator that tracks a robot's position as it changes in
#    response to twist commands, publishing a marker array showing its past
#    locations.

__author__ = "Jason M. O'Kane"
__copyright__ = "Copyright 2024"

import geometry_msgs.msg
import math
import numpy as np
import rclpy.node
import rclpy.qos
import std_msgs.msg
import tf2_ros
import visualization_msgs.msg
import std_srvs.srv

def euler_to_quaternion(r, p, y):
    # This is the standard formula, which seems not to be included directly in ROS2 for some reason.
    # See also: https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/Using-URDF-with-Robot-State-Publisher.html
    return geometry_msgs.msg.Quaternion(x=math.sin(r/2)*math.cos(p/2)*math.cos(y/2) - math.cos(r/2)*math.sin(p/2)*math.sin(y/2),
                                        y=math.cos(r/2)*math.sin(p/2)*math.cos(y/2) + math.sin(r/2)*math.cos(p/2)*math.sin(y/2),
                                        z=math.cos(r/2)*math.cos(p/2)*math.sin(y/2) - math.sin(r/2)*math.sin(p/2)*math.cos(y/2),
                                        w=math.cos(r/2)*math.cos(p/2)*math.cos(y/2) + math.sin(r/2)*math.sin(p/2)*math.sin(y/2))

class Sim1_Node(rclpy.node.Node):
    def __init__(self):
        # Initialize the node itself.
        super().__init__('sim1')

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

        # Get ready to publish a marker array showing the robot's path.
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

        # How much time are we simulating?
        now = self.get_clock().now()
        seconds_since_start = (now - self.start_time).nanoseconds * 1e-9
        dt = seconds_since_start - self.total_simulated_time
        if verbose:
            print(f"seconds_since_start={seconds_since_start} total_simulated_time={self.total_simulated_time} dt={dt}")
        self.total_simulated_time += dt

        self.pose[0] += math.cos(self.pose[2]) * self.twist.linear.x * dt
        self.pose[1] += math.sin(self.pose[2]) * self.twist.linear.x * dt
        self.pose[2] += self.twist.angular.z * dt

        if self.recording and self.twist.linear.x != 0:
            self.past_locations[-1].append(self.pose[0:2].copy())

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

        radius = 0.5
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
        # Assemble the past locations we've stored into a MarkerArray and
        # publish it.
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

        self.markers_publisher.publish(markers)

    def reset(self, req=None, resp=None):
        # Put everything into an initial state.

        # The current pose of the robot, expressed in (x, y, theta).
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

        markers = visualization_msgs.msg.MarkerArray()
        marker = visualization_msgs.msg.Marker()
        marker.header.frame_id = 'map'
        marker.action = marker.DELETEALL
        markers.markers.append(marker)
        self.markers_publisher.publish(markers)

        # Done.  Send the response object back.
        return resp


    def set_pen(self, req, resp):
        # Start or stop appending to the history.
        self.recording = req.data
        if self.recording:
            self.past_locations.append([ self.pose[0:2].copy() ])
        resp.success=True
        return resp

def main():
    rclpy.init()
    x = Sim1_Node()
    rclpy.spin(x)

if __name__ == '__main__':
    main()

