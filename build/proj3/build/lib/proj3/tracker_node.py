#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
from scipy.spatial import distance


class TrackedPerson:
    def __init__(self, person_id, pos):
        self.id = person_id
        self.positions = [pos]
        self.last_seen = 0
        self.confirmed = False  # only publish once confirmed
        self.confirmation_dist = 0.2  # must move this much total to confirm
        self.min_velocity = 0.02
        self.max_velocity_jump = 0.2

    def update(self, pos):
        self.positions.append(pos)
        self.last_seen = 0  # reset timeout

    def has_moved_enough(self):
        """Check if this track has moved enough to be considered real."""
        if len(self.positions) < 2:
            return False
        start = np.array(self.positions[0])
        end = np.array(self.positions[-1])
        return np.linalg.norm(end - start) > self.confirmation_dist

    def has_consistent_motion(self):
        """Check if the motion flows and is consistent"""
        if len(self.positions) < 3:
            return False
        diffs = np.diff(self.positions, axis=0)
        norms = np.linalg.norm(diffs, axis=1)
        return np.mean(norms) > self.min_velocity and np.std(norms) < self.max_velocity_jump


class TrackerNode(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.subscription = self.create_subscription(Float32MultiArray, '/detected_people', self.detected_callback, 10)
        self.publisher = self.create_publisher(MarkerArray, '/person_markers', 10)
        self.tracked = {}
        self.next_id = 0
        self.timeout = 15  # frames before deleting a person
        self.association_dist = 0.7  # meters
        self.get_logger().info("Tracker node started. Listening to /detected_people...")

        # periodic maintenance
        self.timer = self.create_timer(0.1, self.update_tracks)

    def detected_callback(self, msg):
        detections = np.array(msg.data).reshape(-1, 2) if len(msg.data) > 0 else np.empty((0, 2))
        self.associate_detections(detections)
        self.publish_markers()

    def associate_detections(self, detections):
        track_ids = list(self.tracked.keys())
        track_positions = np.array([self.tracked[i].positions[-1] for i in track_ids]) if track_ids else np.empty((0, 2))

        # if no tracks exist, start new ones
        if len(track_positions) == 0:
            for det in detections:
                self.tracked[self.next_id] = TrackedPerson(self.next_id, det)
                self.next_id += 1
            return

        # if no detections, nothing to do
        if len(detections) == 0:
            return

        # match detections to tracks
        dist_matrix = distance.cdist(track_positions, detections)
        assigned_tracks = set()
        assigned_detections = set()

        for i, track_id in enumerate(track_ids):
            min_j = np.argmin(dist_matrix[i])
            if dist_matrix[i, min_j] < self.association_dist:
                self.tracked[track_id].update(detections[min_j])
                assigned_tracks.add(track_id)
                assigned_detections.add(min_j)

        # new detections (unmatched)
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                self.tracked[self.next_id] = TrackedPerson(self.next_id, det)
                self.next_id += 1

    def update_tracks(self):
        # increment unseen counters and clean up old tracks
        for track_id in list(self.tracked.keys()):
            track = self.tracked[track_id]
            track.last_seen += 1

            # confirm if it moved enough
            if not track.confirmed and track.has_moved_enough() and track.has_consistent_motion():
                track.confirmed = True
                self.get_logger().info(f"Confirmed moving person {track_id}")

            # delete if unseen for too long
            if track.last_seen > self.timeout:
                del self.tracked[track_id]

    def publish_markers(self):
        markers = MarkerArray()
        for track_id, track in self.tracked.items():
            # only publish confirmed tracks (moving people)
            if not track.confirmed:
                continue

            marker = Marker()
            marker.header.frame_id = "laser"
            marker.ns = "people"
            marker.id = track_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.0, 1.0, 0.0, 1.0)

            marker.points = []
            for pos in track.positions:
                p = Point()
                p.x, p.y, p.z = float(pos[0]), float(pos[1]), 0.0
                marker.points.append(p)
            markers.markers.append(marker)

        self.publisher.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
