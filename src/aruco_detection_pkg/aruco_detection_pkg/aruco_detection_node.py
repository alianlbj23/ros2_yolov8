import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32, String
import cv2
from cv2 import aruco
import numpy as np
import roslibpy


class ArucoDepthDetectionNode(Node):

    def __init__(self):
        super().__init__('ArucoDepthDetection_node')
        self.get_logger().info('ArucoDepthDetection Node is running')
        self.bridge = CvBridge()

        # Declare the parameters
        self.declare_parameter('id', -1)
        self.declare_parameter('rosbridge_ip', '192.168.0.52')
        self.declare_parameter('rosbridge_port', 9090)
        self.marker_id_to_detect = self.get_parameter('id').get_parameter_value().integer_value

        # Create the predefined dictionary for marker detection
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)
        self.param_markers = aruco.DetectorParameters()

        # ROSBridge connection
        rosbridge_ip = self.get_parameter('rosbridge_ip').get_parameter_value().string_value
        rosbridge_port = self.get_parameter('rosbridge_port').get_parameter_value().integer_value
        self.rosbridge_client = roslibpy.Ros(host=rosbridge_ip, port=rosbridge_port)
        self.rosbridge_client.run()

        # Topics for publishing via roslibpy
        self.roslibpy_depth_topic = roslibpy.Topic(self.rosbridge_client, '/arucode_depth', 'std_msgs/Float32')
        self.roslibpy_direction_topic = roslibpy.Topic(self.rosbridge_client, '/arucode_direction', 'std_msgs/String')

        # Subscription to the camera color image topic
        self.color_subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.color_image_callback,
            10
        )

        # Subscription to the camera depth image topic
        self.depth_subscription = self.create_subscription(
            CompressedImage,
            '/camera/depth/image_raw/compressed',
            self.depth_image_callback,
            10
        )

        # Publisher to publish the ArUco marker depth and direction (ROS 2)
        self.depth_publisher = self.create_publisher(Float32, 'arucode_depth', 10)
        self.direction_publisher = self.create_publisher(String, 'arucode_direction', 10)

        self.depth_image = None  # To store the latest depth image
        self.detected_markers = {}  # To store detected ArUco marker information

    def color_image_callback(self, msg):
        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        image_width = frame.shape[1]

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        marker_corners, marker_IDs, _ = aruco.detectMarkers(
            gray_frame, self.marker_dict, parameters=self.param_markers
        )

        # If markers are detected, store their positions
        if marker_corners:
            for ids, corners in zip(marker_IDs, marker_corners):
                if self.marker_id_to_detect == -1 or ids[0] == self.marker_id_to_detect:
                    # Calculate the center of the ArUco marker
                    corners = corners.reshape(4, 2)
                    center_x = int(corners[:, 0].mean())
                    center_y = int(corners[:, 1].mean())

                    # Store marker information
                    self.detected_markers[ids[0]] = (center_x, center_y)

                    # Determine position (left, right, center)
                    position = self.determine_position(center_x, image_width)
                    # Publish the position
                    self.publish_direction(position)

                    # Calculate depth if depth image is available
                    if self.depth_image is not None:
                        depth = self.depth_image[center_y, center_x]
                        if depth > 0:
                            depth_in_meters = depth / 1000.0  # Assuming the depth image is in millimeters
                            self.publish_depth(float(depth_in_meters))

    def depth_image_callback(self, msg):
        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            self.depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        else:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def publish_depth(self, depth):
        # Publish the depth to the 'arucode_depth' topic (ROS 2)
        depth_msg = Float32()
        depth_msg.data = depth
        self.depth_publisher.publish(depth_msg)

        # Publish the depth to the 'arucode_depth' topic (ROSBridge)
        roslibpy_depth_msg = roslibpy.Message({'data': depth})
        self.roslibpy_depth_topic.publish(roslibpy_depth_msg)

    def publish_direction(self, direction):
        # Publish the direction to the 'arucode_direction' topic (ROS 2)
        direction_msg = String()
        direction_msg.data = direction
        self.direction_publisher.publish(direction_msg)

        # Publish the direction to the 'arucode_direction' topic (ROSBridge)
        roslibpy_direction_msg = roslibpy.Message({'data': direction})
        self.roslibpy_direction_topic.publish(roslibpy_direction_msg)

    def determine_position(self, center_x, image_width):
        # Divide the image into three sections: left, center, and right
        if center_x < image_width * 0.33:
            return "left"
        elif center_x > image_width * 0.66:
            return "right"
        else:
            return "front"

def main():
    rclpy.init()
    node = ArucoDepthDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
