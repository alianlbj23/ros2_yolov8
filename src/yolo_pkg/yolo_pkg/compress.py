import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageCompressionNode(Node):
    def __init__(self):
        super().__init__('image_compression_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribers
        self.color_subscriber = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)

        self.depth_subscriber = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)

        # Publishers
        self.color_compressed_publisher = self.create_publisher(
            CompressedImage, '/camera/color/image_raw/compressed1', 10)

        self.depth_compressed_publisher = self.create_publisher(
            CompressedImage, '/camera/depth/image_raw/compressed1', 10)

    def color_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Compress the image using optimized JPEG settings
        compressed_image = self.compress_image(cv_image, '.jpg', [cv2.IMWRITE_JPEG_QUALITY, 70])

        # Create CompressedImage message
        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header
        compressed_msg.format = "jpeg"
        compressed_msg.data = compressed_image

        # Publish compressed image
        self.color_compressed_publisher.publish(compressed_msg)

    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Compress the image using PNG to preserve depth accuracy
        compressed_image = self.compress_image(cv_image, '.png', [cv2.IMWRITE_PNG_COMPRESSION, 3])

        # Create CompressedImage message
        compressed_msg = CompressedImage()
        compressed_msg.header = msg.header
        compressed_msg.format = "png"
        compressed_msg.data = compressed_image

        # Publish compressed image
        self.depth_compressed_publisher.publish(compressed_msg)

    def compress_image(self, cv_image, extension, compression_params):
        # Compress the image using the specified format and parameters
        result, encimg = cv2.imencode(extension, cv_image, compression_params)
        return encimg.tobytes()

def main(args=None):
    rclpy.init(args=args)

    image_compression_node = ImageCompressionNode()

    try:
        rclpy.spin(image_compression_node)
    except KeyboardInterrupt:
        pass

    image_compression_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
