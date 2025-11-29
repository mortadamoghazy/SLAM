#!/usr/bin/env python3
import os
import time
import json
import csv
import sys

import numpy as np
import psutil

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, LookupException

class SLAMEvaluator(Node):
    def __init__(self):
        super().__init__('slam_evaluator')

        # --- PARAMETERS ---
        self.declare_parameter('est_topic', '/pose')  # ✅ use actual published topic
        self.declare_parameter('world_frame', 'odom')  # ✅ use odom, not "world"
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('output_dir', os.path.expanduser('~/slam_eval'))
        self.declare_parameter('duration_sec', 60.0)  # ✅ Optional shutdown timer

        self.est_topic   = self.get_parameter('est_topic').value
        self.world_frame = self.get_parameter('world_frame').value
        self.base_frame  = self.get_parameter('base_frame').value
        self.output_dir  = self.get_parameter('output_dir').value
        self.duration_sec = self.get_parameter('duration_sec').value

        os.makedirs(self.output_dir, exist_ok=True)

        self.get_logger().info(f"✅ SLAM evaluator running for {self.duration_sec:.0f} seconds")
        self.get_logger().info(f"⏺ Saving to: {self.output_dir}")
        self.get_logger().info(f"📡 Listening to topic: {self.est_topic}")
        self.get_logger().info(f"🧭 TF: {self.world_frame} → {self.base_frame}")

        # --- BUFFERS ---
        self.gt_buf   = []
        self.est_buf  = []
        self.perf_buf = []

        self.proc = psutil.Process()
        self._last_cb = None

        # TF listener
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscribe to SLAM pose
        self.create_subscription(
            PoseWithCovarianceStamped,
            self.est_topic,
            self._est_cb,
            10
        )

        # shutdown timer
        self.shutdown_timer = self.create_timer(self.duration_sec, self.shutdown)

        # TF warmup (allow 3s for TFs to populate)
        self.tf_ready = False
        self.create_timer(3.0, self._check_tf_ready)

    def _check_tf_ready(self):
        try:
            self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, rclpy.time.Time())
            self.tf_ready = True
            self.get_logger().info("✅ TF buffer ready.")
        except LookupException:
            self.get_logger().warn("⚠️ Waiting for TF buffer...")

    def _est_cb(self, msg: PoseWithCovarianceStamped):
        if not self.tf_ready:
            return

        # timestamp
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # record estimate
        p = msg.pose.pose.position
        self.est_buf.append((t, p.x, p.y, p.z))
        self.get_logger().info(f"[✓] Pose@{t:.2f} → Est=({p.x:.2f}, {p.y:.2f})")

        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.base_frame,
                Time(seconds=msg.header.stamp.sec, nanoseconds=msg.header.stamp.nanosec)
            )
            gt_p = trans.transform.translation
            self.gt_buf.append((t, gt_p.x, gt_p.y, gt_p.z))
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed at t={t:.3f}: {e}")
            return

        now = time.time()
        dt  = now - self._last_cb if self._last_cb else 0.0
        cpu = psutil.cpu_percent(interval=None)
        mem = self.proc.memory_info().rss / (1024**2)
        self.perf_buf.append((t, cpu, mem, dt))
        self._last_cb = now

    def shutdown(self):
        self.get_logger().info("🔻 Shutting down and saving files...")

        with open(os.path.join(self.output_dir, 'ground_truth.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['time','x','y','z']); w.writerows(self.gt_buf)
        with open(os.path.join(self.output_dir, 'estimated.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['time','x','y','z']); w.writerows(self.est_buf)
        with open(os.path.join(self.output_dir, 'performance.csv'), 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['time','cpu_percent','mem_MB','callback_dt']); w.writerows(self.perf_buf)

        metrics = self._compute_metrics()
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        self.get_logger().info("✅ Evaluation complete.")
        rclpy.shutdown()

    def _compute_metrics(self):
        gt   = np.array(self.gt_buf)
        est  = np.array(self.est_buf)
        perf = np.array(self.perf_buf)

        if gt.shape[0]<2 or est.shape[0]<2:
            return {'error':'insufficient data'}

        t_gt, pts_gt = gt[:,0], gt[:,1:]
        t_e,  pts_e  = est[:,0], est[:,1:]

        errs = np.linalg.norm(pts_e - pts_gt, axis=1)
        ate  = float(np.sqrt(np.mean(errs**2)))

        dgt  = pts_gt[1:]-pts_gt[:-1]
        dest = pts_e[1:]-pts_e[:-1]
        rpe  = float(np.sqrt(np.mean(np.linalg.norm(dest-dgt,axis=1)**2)))

        pl_gt  = float(np.sum(np.linalg.norm(dgt,axis=1)))
        pl_est = float(np.sum(np.linalg.norm(dest,axis=1)))

        drift_per_m  = ate/pl_gt  if pl_gt>0 else None
        scale_drift  = pl_est/pl_gt if pl_gt>0 else None

        cpu = perf[:,1]; mem = perf[:,2]; dt = perf[:,3][1:] if perf.shape[0]>1 else []
        perf_stats = {
            'cpu_avg': float(np.mean(cpu)),
            'cpu_peak':float(np.max(cpu)),
            'mem_avg_MB':float(np.mean(mem)),
            'mem_peak_MB':float(np.max(mem)),
            'avg_callback_interval_s': float(np.mean(dt)) if len(dt) else None,
            'max_callback_interval_s': float(np.max(dt))  if len(dt) else None,
        }

        return {
            'ATE_m': ate,
            'RPE_m': rpe,
            'gt_path_length_m': pl_gt,
            'est_path_length_m': pl_est,
            'drift_per_meter': drift_per_m,
            'scale_drift': scale_drift,
            'performance': perf_stats
        }


def main(args=None):
    rclpy.init(args=args)
    node = SLAMEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()