import os
from launch import LaunchDescription
# RegisterEventHandler was missing from your imports
from launch.actions import DeclareLaunchArgument, ExecuteProcess, Shutdown, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    rviz_path = "/home/maria-visinescu/ros2_ws/src/CSCE752_Project3/src/sim_configured.rviz"

    bag_in_arg = DeclareLaunchArgument(
        'bag_in',
        description='Full path to the input bag file.'
    )

    bag_out_arg = DeclareLaunchArgument(
        'bag_out',
        description='Path to the directory to record the output bag file in.'
    )

    # 2. Get argument values
    bag_in = LaunchConfiguration('bag_in')
    bag_out = LaunchConfiguration('bag_out')

    # 3. Define your two nodes
    detector_node = Node(
        package='proj3',
        executable='detector_node',
        name='detector_node'
    )

    tracker_node = Node(
        package='proj3',
        executable='tracker_node',
        name='tracker_node'
    )

    # 4. Define bag play and record (no --clock)
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_in],
        output='screen'
    )

    bag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-a', '-o', bag_out],
        output='screen'
    )

    # 5. Define RViz node, loading the hardcoded config
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_path],
        output='screen'
    )

    # 6. Define the event handler object
    shutdown_on_play_exit_handler = OnProcessExit(
        target_action=bag_play,
        on_exit=[
            Shutdown()
        ]
    )
    
    # 7. Create the RegisterEventHandler action to wrap the handler
    register_shutdown_hook = RegisterEventHandler(
        event_handler=shutdown_on_play_exit_handler
    )

    # 8. Return the launch description
    return LaunchDescription([
        bag_in_arg,
        bag_out_arg,
        
        detector_node,
        tracker_node,
        bag_play,
        bag_record,
        rviz_node,
        
        # Add the RegisterEventHandler action here, NOT the OnProcessExit object
        register_shutdown_hook 
    ])