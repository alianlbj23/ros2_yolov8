import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2
import os
from yolo_msgs.msg import Detection  # 自定义的消息类型
from threading import Thread, Event

class PokemonNode(Node):

    def __init__(self):
        super().__init__("pokemon_yolo_node")
        self.get_logger().info("Pokemon Node is running")
        self.bridge = CvBridge()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model = YOLO(
            "/workspaces/src/yolo_pkg/resource/best.pt"
        )

        self.publisher_ = self.create_publisher(Image, "yolo_output_topic", 10)
        self.detection_publisher_ = self.create_publisher(Detection, "yolo_detection_topic", 10)
        self.cap = cv2.VideoCapture(0)  # 使用设备ID 0 打开默认相机
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            rclpy.shutdown()
            return

        self.stop_event = Event()
        self.thread = Thread(target=self.process_frames)
        self.thread.start()

    def process_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().info("Failed to capture frame from camera.")
                continue

            # 缩小图像分辨率
            frame = cv2.resize(frame, (320, 240))  # 缩小到320x240分辨率

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
                        self.detection_publisher_.publish(detection_msg)

            # 可视化识别结果
            annotated_frame = results[0].plot()

            # 压缩图像质量
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]  # 调整质量参数，值越小，压缩率越高
            result, compressed_image = cv2.imencode('.jpg', annotated_frame, encode_param)
            compressed_frame = cv2.imdecode(compressed_image, 1)

            # 发布识别结果图像
            msg = self.bridge.cv2_to_imgmsg(compressed_frame, encoding="bgr8")
            self.publisher_.publish(msg)

            self.stop_event.wait(0.2)  # 调整发布频率

    def destroy_node(self):
        self.stop_event.set()
        self.thread.join()
        self.cap.release()
        super().destroy_node()
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

def main():
    rclpy.init()
    node = PokemonNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
