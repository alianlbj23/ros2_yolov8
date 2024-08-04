import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from yolo_msgs.msg import Detection  # 自定义的消息类型
import os
import numpy as np
import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

class PokemonNode(Node):

    def __init__(self):
        super().__init__("pokemon_yolo_node")
        self.get_logger().info("Pokemon Node is running")
        self.bridge = CvBridge()
        package_share_directory = get_package_share_directory('yolo_pkg')
        model_path = os.path.join(package_share_directory, 'resource', 'best.pt')
        self.model = YOLO(model_path)

        # Create publishers
        self.detection_image_publisher_ = self.create_publisher(Image, "yolo_detection_image", 10)
        self.detection_publisher_ = self.create_publisher(Detection, "yolo_detection_topic", 10)
        self.create_subscription(Image, "/camera/color/image_raw", self.image_callback, 10)
        self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)

        self.depth_image = None

    def image_callback(self, msg):
        # 将图像消息转换为OpenCV格式
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 使用YOLO进行物体检测
        results = self.model.predict(source=frame, verbose=False)  # 使用 YOLO 模型进行识别，禁用详细日志

        if results and len(results[0].boxes) > 0:
            # 只在有检测结果时输出日志，并发布检测结果
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框坐标
                    confidence = box.conf[0]  # 获取置信度
                    cls = box.cls[0]  # 获取类别
                    depth = self.get_depth_for_box(x1, y1, x2, y2)  # 获取深度信息

                    if depth is not None:
                        self.get_logger().info(f"Depth={depth}")

                        detection_msg = Detection()
                        detection_msg.class_name = str(cls)
                        detection_msg.confidence = float(confidence)
                        detection_msg.xmin = float(x1)
                        detection_msg.ymin = float(y1)
                        detection_msg.xmax = float(x2)
                        detection_msg.ymax = float(y2)
                        detection_msg.depth = float(depth)  # 加入深度信息
                        self.detection_publisher_.publish(detection_msg)
                    else:
                        self.get_logger().info(
                            f"Detected object: Class={cls}, Confidence={confidence}, Box=[{x1}, {y1}, {x2}, {y2}], Depth=None"
                        )

        # 可视化识别结果
        annotated_frame = results[0].plot()

        # 发布识别结果图像
        msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.detection_image_publisher_.publish(msg)

    def depth_callback(self, msg):
        # 将深度图像转换为OpenCV格式
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_depth_for_box(self, x1, y1, x2, y2):
        if self.depth_image is not None:
            # 提取检测框内的深度信息
            depth_region = self.depth_image[int(y1):int(y2), int(x1):int(x2)]
            # 计算平均深度，忽略为0的深度值（无效深度值）
            valid_depths = depth_region[depth_region > 0]
            self.get_logger().info(f"Depth region shape: {depth_region.shape}, valid depth count: {valid_depths.size}")
            if valid_depths.size > 0:
                return np.mean(valid_depths)
        return None

def main():
    rclpy.init()
    node = PokemonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
