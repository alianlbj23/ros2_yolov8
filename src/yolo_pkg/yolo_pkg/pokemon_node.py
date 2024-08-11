import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, Imu
from yolo_msgs.msg import Detection  # 自定义的消息类型
from geometry_msgs.msg import Point  # 用于发布坐标
import os
import numpy as np
import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import yaml
import tf_transformations
import matplotlib.pyplot as plt
from collections import deque
import roslibpy

class PokemonNode(Node):

    def __init__(self):
        super().__init__("pokemon_yolo_node")
        self.get_logger().info("Pokemon Node is running")
        self.bridge = CvBridge()
        self.orientation_data = deque(maxlen=100)

        package_share_directory = get_package_share_directory('yolo_pkg')
        model_path = os.path.join(package_share_directory, 'resource', 'best.pt')
        self.model = YOLO(model_path)

        # ROSBridge client to connect to Jetson
        self.rosbridge_client = roslibpy.Ros(host='192.168.0.207', port=9090)
        self.rosbridge_client.run()
        self.coordinate_topic = roslibpy.Topic(self.rosbridge_client, '/object_coordinates', 'geometry_msgs/Point')

        # 读取相机内参
        camera_calib_path = os.path.join(package_share_directory, 'resource', 'ost.yaml')
        with open(camera_calib_path, 'r') as file:
            calib_data = yaml.safe_load(file)
            self.camera_matrix = np.array(calib_data['camera_matrix']['data']).reshape((3, 3))
            self.dist_coeffs = np.array(calib_data['distortion_coefficients']['data'])

        # 初始外参矩阵（假设为单位矩阵）
        self.extrinsics_matrix = np.eye(4)

        # 创建发布者
        self.detection_image_publisher_ = self.create_publisher(Image, "yolo_detection_image", 10)
        self.detection_publisher_ = self.create_publisher(Detection, "yolo_detection_topic", 10)
        self.coordinate_publisher_ = self.create_publisher(Point, "object_coordinates", 10)  # 新的坐标发布者

        self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", self.image_callback, 10)
        self.create_subscription(CompressedImage, "/camera/depth/image_raw/compressed", self.depth_callback, 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        self.create_subscription(Imu, "/imu/data_raw", self.imu_callback, 10)

        self.depth_image = None

    def imu_callback(self, msg):
        orientation_q = msg.orientation
        # 将四元数转换为欧拉角
        euler = tf_transformations.euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])
        self.orientation_data.append(euler)
        # self.plot_orientation()

    def plot_orientation(self):
        plt.clf()
        plt.plot(range(len(self.orientation_data)), [d[0] for d in self.orientation_data], label='Roll')
        plt.plot(range(len(self.orientation_data)), [d[1] for d in self.orientation_data], label='Pitch')
        plt.plot(range(len(self.orientation_data)), [d[2] for d in self.orientation_data], label='Yaw')
        plt.legend()
        plt.pause(0.01)

    def image_callback(self, msg):
        if isinstance(msg, CompressedImage):
            # 解压缩图像数据
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 使用YOLO进行物体检测，显式指定使用GPU
        results = self.model.predict(source=frame, verbose=False, device='cuda')

        if results and len(results[0].boxes) > 0:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0]
                    cls = box.cls[0]
                    depth, point = self.get_depth_and_point_for_box(x1, y1, x2, y2)

                    if depth is not None and point is not None:
                        # 使用内参矩阵将像素坐标转换为相机坐标
                        point_camera = np.dot(np.linalg.inv(self.camera_matrix), np.array([point[0], point[1], 1.0])) * depth

                        # 计算物体在世界坐标系中的位置
                        point_world = np.dot(self.extrinsics_matrix, np.array([point_camera[0], point_camera[1], point_camera[2], 1.0]))

                        # 发布检测消息
                        detection_msg = Detection()
                        detection_msg.class_name = str(cls)
                        detection_msg.confidence = float(confidence)
                        detection_msg.xmin = float(x1)
                        detection_msg.ymin = float(y1)
                        detection_msg.xmax = float(x2)
                        detection_msg.ymax = float(y2)
                        detection_msg.depth = float(depth)
                        detection_msg.point_x = float(point_world[0])
                        detection_msg.point_y = float(point_world[1])
                        detection_msg.point_z = float(point_world[2])
                        self.detection_publisher_.publish(detection_msg)

                        # 发布坐标消息
                        point_msg = Point()
                        point_msg.x = float(point_world[0])
                        point_msg.y = float(point_world[1])
                        point_msg.z = float(point_world[2])
                        self.coordinate_publisher_.publish(point_msg)
                        roslibpy_point_msg = roslibpy.Message({
                            'x': float(point_world[0]),
                            'y': float(point_world[1]),
                            'z': float(point_world[2])
                        })
                        self.coordinate_topic.publish(roslibpy_point_msg)
                        # print("position : ", point_msg.x, point_msg.y, point_msg.z)

        annotated_frame = results[0].plot()
        msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.detection_image_publisher_.publish(msg)

    def depth_callback(self, msg):
        if isinstance(msg, CompressedImage):
            # 解压缩图像数据
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        else:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth_and_point_for_box(self, x1, y1, x2, y2):
        if self.depth_image is not None:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            center_x = min(max(center_x, 0), self.depth_image.shape[1] - 1)
            center_y = min(max(center_y, 0), self.depth_image.shape[0] - 1)
            depth = self.depth_image[center_y, center_x]
            if depth > 0:
                point = (center_x, center_y, depth)
                return depth, point
        return None, None

def main():
    rclpy.init()
    node = PokemonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
