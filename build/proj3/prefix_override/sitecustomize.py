import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/maria-visinescu/ros2_ws/src/CSCE752_Project3/install/proj3'
