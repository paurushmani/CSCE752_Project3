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
        self.declare_parameter('cluster_eps', 1.0)   # meters
        self.declare_parameter('min_points', 3)
        self.declare_parameter('min_width', 0.07)
        self.declare_parameter('max_width', 2.0)
        self.declare_parameter('min_height', 0.07)
        self.declare_parameter('max_height', 2.0)

        # Background initialization
        self.background_points = None
        self.bg_frames = 5
        self.bg_data = []

    def scan_callback(self, scan):
        # Convert to (x,y)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)
        mask = np.isfinite(ranges)
        xs = ranges[mask] * np.cos(angles[mask])
        ys = ranges[mask] * np.sin(angles[mask])
        points = np.vstack((xs, ys)).T
        print(points)

        # Build static background from first couple of frames
        if self.background_points is None:
            self.bg_data.append(points)
            if len(self.bg_data) >= self.bg_frames:
                self.background_points = np.vstack(self.bg_data)
            return

        # Remove static background points
        dists = np.min(np.linalg.norm(points[:, None, :] - self.background_points[None, :, :], axis=2), axis=1)
        points = points[dists > 0.05]

        if len(points) == 0:
            return

        # Cluster points
        eps = self.get_parameter('cluster_eps').value
        min_samples = self.get_parameter('min_points').value
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        min_width = self.get_parameter('min_width').value
        max_width = self.get_parameter('max_width').value
        min_height = self.get_parameter('min_height').value
        max_height= self.get_parameter('max_height').value

        centroids = []
        for lbl in set(labels):
            if lbl == -1:
                continue  # noise
            cluster = points[labels == lbl]
            width = cluster[:,0].max() - cluster[:,0].min()
            height = cluster[:,1].max() - cluster[:,1].min()
            # Don't count any clusters that are not the right size
            if not (min_width <= width <= max_width and min_height <= height <= max_height):
                continue
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
