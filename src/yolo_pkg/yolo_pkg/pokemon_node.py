import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from ultralytics import YOLO
import cv2
import os
from yolo_msgs.msg import Detection  # 自定义的消息类型
import numpy as np

class PokemonNode(Node):

    def __init__(self):
        super().__init__("pokemon_yolo_node")
        self.get_logger().info("Pokemon Node is running")
        self.bridge = CvBridge()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = YOLO(
            "/workspaces/src/yolo_pkg/resource/best.pt"
        )

        self.detection_publisher_ = self.create_publisher(Image, "yolo_detection_topic", 10)
        self.create_subscription(CompressedImage, "/out/compressed", self.image_callback, 10)

    def image_callback(self, msg):
        # 将压缩图像转换为OpenCV格式
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 使用YOLO进行物体检测
        results = self.model.predict(
            source=frame, verbose=False
        )  # 使用 YOLO 模型进行识别，禁用详细日志

        if results and len(results[0].boxes) > 0:
            # 只在有检测结果时输出日志，并发布检测结果
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框坐标
                    confidence = box.conf[0]  # 获取置信度
                    cls = box.cls[0]  # 获取类别
                    self.get_logger().info(
                        f"Detected object: Class={cls}, Confidence={confidence}, Box=[{x1}, {y1}, {x2}, {y2}]"
                    )
                    detection_msg = Detection()
                    detection_msg.class_name = str(cls)
                    detection_msg.confidence = float(confidence)
                    detection_msg.xmin = float(x1)
                    detection_msg.ymin = float(y1)
                    detection_msg.xmax = float(x2)
                    detection_msg.ymax = float(y2)
                    # self.detection_publisher_.publish(detection_msg)

        # 可视化识别结果
        annotated_frame = results[0].plot()

        # 发布识别结果图像
        msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.detection_publisher_.publish(msg)

def main():
    rclpy.init()
    node = PokemonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
