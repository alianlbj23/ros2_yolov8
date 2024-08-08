import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
import numpy as np
import math
import os
from ament_index_python.packages import get_package_share_directory

class GetFaceDistanceFromCamera(Node):

    def __init__(self):
        super().__init__('get_face_distance_from_camera')

        self.bridge = CvBridge()

        self.camera_info_sub = Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.image_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.camera_info_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.callback)

        self.pub = self.create_publisher(Image, '/unibas_face_distance_calculator/faces', 1)

    def callback(self, rgb_data, depth_data, camera_info):
        try:
            camera_info_K = np.array(camera_info.k)

            # Intrinsic camera matrix for the raw (distorted) images.
            m_fx = camera_info_K[0]
            m_fy = camera_info_K[4]
            m_cx = camera_info_K[2]
            m_cy = camera_info_K[5]
            inv_fx = 1. / m_fx
            inv_fy = 1. / m_fy

            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
            depth_array = np.array(depth_image, dtype=np.float32)
            cv2.normalize(depth_array, depth_array, 0, 1, cv2.NORM_MINMAX)
            depth_8 = (depth_array * 255).round().astype(np.uint8)
            cv_depth = np.zeros_like(cv_rgb)
            cv_depth[:,:,0] = depth_8
            cv_depth[:,:,1] = depth_8
            cv_depth[:,:,2] = depth_8
            package_share_directory = get_package_share_directory('face_detect')
            model_path = os.path.join(package_share_directory, 'resource', 'haarcascade_frontalface_default.xml')

            face_cascade = cv2.CascadeClassifier('/home/bloisi/catkin_ws/src/unibas_face_distance_calculator/haarcascade/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            rgb_height, rgb_width, rgb_channels = cv_rgb.shape

            for (x, y, w, h) in faces:
                cv_rgb, cv_depth = self.draw_faces(cv_rgb, cv_depth, x, y, w, h, depth_image, m_fx, m_fy, m_cx, m_cy, inv_fx, inv_fy)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

        rgbd = np.concatenate((cv_rgb, cv_depth), axis=1)

        try:
            faces_message = self.bridge.cv2_to_imgmsg(rgbd, "bgr8")
            self.pub.publish(faces_message)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def draw_faces(self, cv_rgb, cv_depth, x, y, w, h, depth_image, m_fx, m_fy, m_cx, m_cy, inv_fx, inv_fy):
        cv2.rectangle(cv_rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(cv_depth, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(cv_rgb, (x + 30, y + 30), (x + w - 30, y + h - 30), (0, 0, 255), 2)
        cv2.rectangle(cv_depth, (x + 30, y + 30), (x + w - 30, y + h - 30), (0, 0, 255), 2)
        roi_depth = depth_image[y + 30:y + h - 30, x + 30:x + w - 30]

        n = 0
        sum_depth = 0
        for i in range(roi_depth.shape[0]):
            for j in range(roi_depth.shape[1]):
                value = roi_depth.item(i, j)
                if value > 0.:
                    n += 1
                    sum_depth += value

        mean_z = sum_depth / n if n > 0 else 0

        point_z = mean_z * 0.001  # distance in meters
        point_x = ((x + w / 2) - m_cx) * point_z * inv_fx
        point_y = ((y + h / 2) - m_cy) * point_z * inv_fy

        x_str = f"X: {point_x:.2f}"
        y_str = f"Y: {point_y:.2f}"
        z_str = f"Z: {point_z:.2f}"

        cv2.putText(cv_rgb, x_str, (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cv_rgb, y_str, (x + w, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cv_rgb, z_str, (x + w, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

        dist = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)
        dist_str = f"dist: {dist:.2f}m"
        cv2.putText(cv_rgb, dist_str, (x + w, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

        return cv_rgb, cv_depth


def main(args=None):
    rclpy.init(args=args)
    node = GetFaceDistanceFromCamera()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
