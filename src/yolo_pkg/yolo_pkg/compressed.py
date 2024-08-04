import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ImageCompressor(Node):
    def __init__(self):
        super().__init__('image_compressor')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # 订阅原始图像话题
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(CompressedImage, 'compressed_image_topic', 10)  # 发布压缩图像话题
        self.br = CvBridge()
        self.get_logger().info('Image compressor node has been started.')

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info('Image received and converted to OpenCV format.')

            # Compress the image using OpenCV
            success, compressed_image = cv2.imencode('.jpg', cv_image)

            if success:
                # Create a CompressedImage message
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "jpeg"
                compressed_msg.data = compressed_image.tobytes()

                # Publish the compressed image
                self.publisher.publish(compressed_msg)
                self.get_logger().info('Compressed image published.')
            else:
                self.get_logger().error('Failed to compress image.')

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unexpected Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    image_compressor = ImageCompressor()
    try:
        rclpy.spin(image_compressor)
    except KeyboardInterrupt:
        pass
    finally:
        image_compressor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
