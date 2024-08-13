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
from std_msgs.msg import String

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
        self.direction_topic = roslibpy.Topic(self.rosbridge_client, '/object_direction', 'std_msgs/String')
        self.depth_topic = roslibpy.Topic(self.rosbridge_client, '/object_depth', 'std_msgs/String')

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
        self.color_image_publisher_ = self.create_publisher(Image, "object_color_image", 10)
        self.coordinate_publisher_ = self.create_publisher(Point, "object_coordinates", 10)  # 新的坐标发布者

        self.create_subscription(CompressedImage, "/camera/color/image_raw/compressed", self.image_callback, 10)
        self.create_subscription(CompressedImage, "/camera/depth/image_raw/compressed", self.depth_callback, 10)
        self.create_subscription(Imu, "/imu/data_raw", self.imu_callback, 10)
        self.color_background = None
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

    def image_callback(self, msg):
        target_tag = "pikachu"  # 指定需要处理的标签
        tolerance = 0.1  # 容忍度

        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.predict(source=frame, verbose=False, device='cuda')

        color_background = np.zeros_like(frame)

        if results and len(results[0].boxes) > 0:
            for result in results:
                for box in result.boxes:
                    label = self.model.names[int(box.cls[0])]
                    if label == target_tag:  # 只处理特定标签的物体
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        object_region = frame[y1:y2, x1:x2]
                        avg_color = cv2.mean(object_region)[:3]

                        color_block = np.full(object_region.shape, avg_color, dtype=np.uint8)
                        color_background[y1:y2, x1:x2] = color_block

                        depth, point = self.get_depth_and_point_for_box(x1, y1, x2, y2)
                        if depth is not None:
                            depth_in_meters = depth / 1000.0
                            print(depth_in_meters)

                            # 发布深度信息到新的topic
                            depth_msg = roslibpy.Message({'data': str(depth_in_meters)})
                            self.depth_topic.publish(depth_msg)

                            if depth_in_meters < 0.2:
                                rect_width = 100
                                rect_height = 100
                                rect_center_x = frame.shape[1] // 2 + 20
                                rect_center_y = frame.shape[0] // 2 + 100

                                box_center_x = (x1 + x2) // 2
                                box_center_y = (y1 + y2) // 2

                                if abs(box_center_x - rect_center_x) < rect_width // 2 and abs(box_center_y - rect_center_y) < rect_height // 2:
                                    direction = "front"
                                elif box_center_x < rect_center_x - rect_width // 2:
                                    direction = "left"
                                elif box_center_x > rect_center_x + rect_width // 2:
                                    direction = "right"
                                elif box_center_y < rect_center_y - rect_height // 2:
                                    direction = "up"
                                else:
                                    direction = "down"
                            else:
                                frame_center_x = frame.shape[1] / 2
                                frame_center_y = frame.shape[0] / 2
                                up_tolerance_factor = 2.0
                                front_offset_factor = 0.5

                                if abs((x1 + x2) / 2 - frame_center_x) / frame.shape[1] < tolerance:
                                    if abs((y1 + y2) / 2 - (frame_center_y - frame.shape[0] * front_offset_factor)) / frame.shape[0] < tolerance:
                                        direction = "front"
                                    elif (y1 + y2) / 2 < frame_center_y - frame.shape[0] * (up_tolerance_factor * tolerance):
                                        direction = "up"
                                    else:
                                        direction = "down"
                                elif (x1 + x2) / 2 < frame_center_x:
                                    direction = "left"
                                else:
                                    direction = "right"

                            direction_msg = roslibpy.Message({'data': direction})
                            self.direction_topic.publish(direction_msg)

                            if point is not None:
                                point_camera = np.dot(np.linalg.inv(self.camera_matrix), np.array([point[0], point[1], 1.0])) * depth_in_meters
                                point_world = np.dot(self.extrinsics_matrix, np.array([point_camera[0], point_camera[1], point_camera[2], 1.0]))
                                point_bullet = np.array([point_world[2], -point_world[0], -point_world[1]])

                                point_msg = Point()
                                point_msg.x = float(point_bullet[0])
                                point_msg.y = float(point_bullet[1])
                                point_msg.z = float(point_bullet[2])
                                self.coordinate_publisher_.publish(point_msg)

                                roslibpy_point_msg = roslibpy.Message({
                                    'x': float(point_bullet[0]),
                                    'y': float(point_bullet[1]),
                                    'z': float(point_bullet[2])
                                })
                                self.coordinate_topic.publish(roslibpy_point_msg)

        annotated_frame = results[0].plot()

        rect_width = 100
        rect_height = 100
        rect_center_x = annotated_frame.shape[1] // 2 + 20
        rect_center_y = annotated_frame.shape[0] // 2 + 100

        top_left = (rect_center_x - rect_width // 2, rect_center_y - rect_height // 2)
        bottom_right = (rect_center_x + rect_width // 2, rect_center_y + rect_height // 2)

        cv2.rectangle(annotated_frame, top_left, bottom_right, color=(0, 255, 0), thickness=2)

        detection_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.detection_image_publisher_.publish(detection_image)

        color_image = self.bridge.cv2_to_imgmsg(color_background, encoding="bgr8")
        self.color_image_publisher_.publish(color_image)

    def depth_callback(self, msg):
        if isinstance(msg, CompressedImage):
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
