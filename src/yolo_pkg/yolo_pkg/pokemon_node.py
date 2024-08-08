import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from yolo_msgs.msg import Detection  # 自定义的消息类型
import os
import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import torch
import numpy as np

class ObjectDetectionNode(Node):

    def __init__(self):
        super().__init__("object_detection_node")
        self.get_logger().info("Object Detection Node is running")
        self.bridge = CvBridge()
        package_share_directory = get_package_share_directory('yolo_pkg')
        model_path = os.path.join(package_share_directory, 'resource', 'best.pt')
        self.model = YOLO(model_path)

        # 检查CUDA是否可用并设置模型设备
        if torch.cuda.is_available():
            self.get_logger().info(f"CUDA is available. Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
            self.model.to('cuda')
        else:
            self.get_logger().info("CUDA is not available, using CPU.")
            self.model.to('cpu')

        # Create publishers
        self.detection_image_publisher_ = self.create_publisher(Image, "yolo_detection_image", 10)
        self.detection_publisher_ = self.create_publisher(Detection, "yolo_detection_topic", 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_image_callback, 10)
        self.create_subscription(CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 10)

        self.depth_image = None
        self.camera_info = None

    def image_callback(self, msg):
        # 将图像消息转换为OpenCV格式
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 使用YOLO进行物体检测
        results = self.model.predict(source=frame, verbose=False)

        if results and len(results[0].boxes) > 0:
            # 只在有检测结果时输出日志，并发布检测结果
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框坐标
                    confidence = box.conf[0]  # 获取置信度
                    cls = box.cls[0]  # 获取类别

                    if self.depth_image is not None and self.camera_info is not None:
                        depth, point = self.get_point_and_depth_for_box(x1, y1, x2, y2)
                        if depth is not None and point is not None:
                            # self.get_logger().info(f"Detected object: {cls}, Depth={depth:.2f} meters, Point={point}")
                            detection_msg = Detection()
                            detection_msg.class_name = str(cls)
                            detection_msg.confidence = float(confidence)
                            detection_msg.xmin = float(x1)
                            detection_msg.ymin = float(y1)
                            detection_msg.xmax = float(x2)
                            detection_msg.ymax = float(y2)
                            detection_msg.depth = float(depth)
                            detection_msg.point_x = float(point[0])
                            detection_msg.point_y = float(point[1])
                            detection_msg.point_z = float(point[2])
                            self.detection_publisher_.publish(detection_msg)
                    #     else:
                    #         self.get_logger().info(f"Depth data not available for detected object: {cls}")
                    # else:
                    #     self.get_logger().info("Depth image or camera info not available")

        # 可视化识别结果
        annotated_frame = results[0].plot()

        # 发布识别结果图像
        msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.detection_image_publisher_.publish(msg)

    def depth_image_callback(self, msg):
        print(msg)
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def get_point_and_depth_for_box(self, x1, y1, x2, y2):
        # 获取检测框中心点
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # 获取中心点的深度值
        depth = self.depth_image[center_y, center_x]

        if depth > 0 and self.camera_info:
            # 提取相机内参
            fx = self.camera_info.k[0]
            fy = self.camera_info.k[4]
            cx = self.camera_info.k[2]
            cy = self.camera_info.k[5]

            # 计算三维坐标
            point_x = (center_x - cx) * depth / fx
            point_y = (center_y - cy) * depth / fy
            point_z = depth

            return depth, (point_x, point_y, point_z)

        return None, None

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
