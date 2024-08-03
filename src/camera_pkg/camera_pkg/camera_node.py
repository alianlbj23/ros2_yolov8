import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraNode(Node):

    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Camera Node is running")
        self.publisher_ = self.create_publisher(Image, "camera_topic", 10)
        self.cap = cv2.VideoCapture(0)  # 尝试更改设备ID，如 0 或 1
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            rclpy.shutdown()
            return
        self.bridge = CvBridge()
        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Failed to capture frame from camera.")
            self.cap.release()
            rclpy.shutdown()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(msg)
        self.get_logger().info("Publishing frame captured")

        # 调试：显示捕获的画面
        cv2.imshow("Camera Frame", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口


if __name__ == "__main__":
    main()
