import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2


class PokemonNode(Node):

    def __init__(self):
        super().__init__("pokemon_yolo_node")
        self.get_logger().info("Pokemon Node is running")
        self.bridge = CvBridge()
        self.model = YOLO(
            "/home/user/workspace/ros2_yolov8/src/yolo_pkg/resource/best.pt"
        )
        self.publisher_ = self.create_publisher(Image, "yolo_output_topic", 10)
        self.cap = cv2.VideoCapture(0)  # 使用设备ID 0 打开默认相机
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            rclpy.shutdown()
            return
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Failed to capture frame from camera.")
            return

        results = self.model.predict(
            source=frame, verbose=False
        )  # 使用 YOLO 模型进行识别，禁用详细日志

        if results and len(results[0].boxes) > 0:
            # 只在有检测结果时输出日志
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取边界框坐标
                    confidence = box.conf[0]  # 获取置信度
                    cls = box.cls[0]  # 获取类别
                    self.get_logger().info(
                        f"Detected object: Class={cls}, Confidence={confidence}, Box=[{x1}, {y1}, {x2}, {y2}]"
                    )

        # 可视化识别结果
        annotated_frame = results[0].plot()

        # 发布识别结果图像
        msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.publisher_.publish(msg)

        # 显示识别结果
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()


def main():
    rclpy.init()
    node = PokemonNode()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


if __name__ == "__main__":
    main()
