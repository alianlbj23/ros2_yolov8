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
        model_path = os.path.join(package_share_directory, 'resource', 'best_large.pt')
        self.model = YOLO(model_path)

        # ROSBridge client to connect to Jetson
        self.rosbridge_client = roslibpy.Ros(host='192.168.0.52', port=9090)
        self.rosbridge_client.run()
        self.coordinate_topic = roslibpy.Topic(self.rosbridge_client, '/object_coordinates', 'geometry_msgs/Point')
        self.direction_topic = roslibpy.Topic(self.rosbridge_client, '/object_direction', 'std_msgs/String')
        self.depth_topic = roslibpy.Topic(self.rosbridge_client, '/object_depth', 'std_msgs/String')
        self.tag_exist_topic = roslibpy.Topic(self.rosbridge_client, '/tag_exist', 'std_msgs/String')

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
        self.create_subscription(String, "/tag_name", self.tag_name_callback, 10)
        self.color_background = None
        self.depth_image = None
        self.tag_exist = None
        self.tag_name_from_node = None

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

    # 取得 tag
    def tag_name_callback(self, msg):
        self.tag_name_from_node = msg.data


    def image_callback(self, msg):
        if self.tag_name_from_node != "None":
            target_tag = self.tag_name_from_node
        else:
            target_tag = "None"
        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.predict(source=frame, verbose=False, device='cuda')

        color_background = np.zeros_like(frame)
        found_target = False
        if results and len(results[0].boxes) > 0:
            for result in results:
                for box in result.boxes:
                    label = self.model.names[int(box.cls[0])]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    if label == target_tag:  # 只处理特定标签的物体
                        found_target = True
                        tag_exist_msg = roslibpy.Message({'data': "1"})
                        self.tag_exist_topic.publish(tag_exist_msg)

                        object_region = frame[y1:y2, x1:x2]
                        avg_color = cv2.mean(object_region)[:3]

                        color_block = np.full(object_region.shape, avg_color, dtype=np.uint8)
                        color_background[y1:y2, x1:x2] = color_block

                        depth, point = self.get_depth_and_point_for_box(x1, y1, x2, y2)
                        rect_width = 100
                        rect_height = 100
                        rect_center_x = frame.shape[1] // 2
                        rect_center_y = frame.shape[0] // 2
                        box_center_x = (x1 + x2) // 2
                        box_center_y = (y1 + y2) // 2
                        offset_x = (box_center_x - rect_center_x) / (frame.shape[1] // 10)
                        offset_y = (rect_center_y - box_center_y) / (frame.shape[0] // 10)

                        # 使用 x 和 y 轴刻度判断方向，并调整 y 轴的高度到 -1
                        if abs(offset_x) < 1 and abs(offset_y +1) < 1:
                            direction = "front"  # 物体在方格中心且 y 轴高度接近 -1
                        elif abs(offset_x) > abs(offset_y + 1):
                            if offset_x > 1:
                                direction = "right"
                            else:
                                direction = "left"
                        else:
                            if offset_y < 0:
                                direction = "down"
                            else:
                                direction = "up"
                        direction_msg = roslibpy.Message({'data': direction})
                        self.direction_topic.publish(direction_msg)

                        if depth is not None:
                            depth_in_meters = depth / 1000.0
                            depth_msg = roslibpy.Message({'data': str(depth_in_meters)})
                            self.depth_topic.publish(depth_msg)

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
        if not found_target:
            tag_exist_msg = roslibpy.Message({'data': "0"})
            self.tag_exist_topic.publish(tag_exist_msg)
            direction_msg = roslibpy.Message({'data': "stop"})
            self.direction_topic.publish(direction_msg)

        # 绘制中心的y轴和x轴刻度
        rect_center_x = frame.shape[1] // 2
        rect_center_y = frame.shape[0] // 2
        y_interval = frame.shape[0] // 10  # y轴每个刻度间隔
        x_interval = frame.shape[1] // 10  # x轴每个刻度间隔

        # 绘制Y轴刻度
        for i in range(-5, 6):  # 从-5到+5
            y_position = rect_center_y - i * y_interval
            cv2.line(frame, (rect_center_x - 10, y_position), (rect_center_x + 10, y_position), (0, 255, 0), 2)
            cv2.putText(frame, f"{i}", (rect_center_x + 15, y_position + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # 绘制X轴刻度
        for i in range(-5, 6):  # 从-5到+5
            x_position = rect_center_x + i * x_interval
            cv2.line(frame, (x_position, rect_center_y - 10), (x_position, rect_center_y + 10), (0, 255, 0), 2)
            cv2.putText(frame, f"{i}", (x_position - 10, rect_center_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # 发布处理后的图像
        detection_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
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
