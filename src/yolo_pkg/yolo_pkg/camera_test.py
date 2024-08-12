import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory
import yaml

class DepthCenterTester(Node):

    def __init__(self):
        super().__init__('depth_center_tester')
        self.get_logger().info("Depth Center Tester Node is running")

        # 读取相机内参
        package_share_directory = get_package_share_directory('yolo_pkg')
        camera_calib_path = os.path.join(package_share_directory, 'resource', 'ost.yaml')
        with open(camera_calib_path, 'r') as file:
            calib_data = yaml.safe_load(file)
            self.camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape((3, 3))
            self.dist_coeffs = np.array(calib_data['distortion_coefficients']['data'])

        self.bridge = CvBridge()

        # 订阅深度图像话题
        self.create_subscription(CompressedImage, "/camera/depth/image_raw/compressed", self.depth_callback, 10)

    def depth_callback(self, msg):
        # 解压缩图像数据
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if depth_image is not None:
            # 获取图像中心点坐标
            height, width = depth_image.shape
            center_x = width // 2
            center_y = height // 2

            # 获取中心点的深度值
            depth_value = depth_image[center_y, center_x]

            # 使用内参矩阵将中心点从像素坐标转换为相机坐标
            if depth_value > 0:  # 确保深度值有效
                point_camera = np.dot(np.linalg.inv(self.camera_matrix), np.array([center_x, center_y, 1.0])) * depth_value
                self.get_logger().info(f"Center depth (z): {depth_value:.2f} mm, Camera coordinates: x={point_camera[0]:.2f}, y={point_camera[1]:.2f}, z={point_camera[2]:.2f}")
            else:
                self.get_logger().warn(f"Invalid depth value at the center: {depth_value}")

def main():
    rclpy.init()
    node = DepthCenterTester()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
