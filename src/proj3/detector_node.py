#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import numpy as np
from sklearn.cluster import DBSCAN

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(Float32MultiArray, '/detected_people', 10)
        self.get_logger().info("Detector node started. Listening to /scan...")

        # Parameters
        self.declare_parameter('cluster_eps', 0.4)   # meters
        self.declare_parameter('min_points', 5)

    def scan_callback(self, scan):
        # Convert to (x,y)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.isfinite(ranges)
        xs = ranges[mask] * np.cos(angles[mask])
        ys = ranges[mask] * np.sin(angles[mask])
        points = np.vstack((xs, ys)).T
        print(points)

        # Cluster points
        eps = self.get_parameter('cluster_eps').value
        min_samples = self.get_parameter('min_points').value
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        centroids = []
        for lbl in set(labels):
            if lbl == -1:
                continue  # noise
            cluster = points[labels == lbl]
            if len(cluster) > 0:
                centroid = np.mean(cluster, axis=0)
                centroids.append(centroid)

        # Publish
        msg = Float32MultiArray()
        if len(centroids) > 0:
            msg.data = np.array(centroids).flatten().tolist()
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
