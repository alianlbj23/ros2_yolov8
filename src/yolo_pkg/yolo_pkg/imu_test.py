import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt
from collections import deque
import tf_transformations  # 添加此导入

class ImuPlotter(Node):
    def __init__(self):
        super().__init__('imu_plotter')

        # 用于存储原始和滤波后的数据
        self.raw_data = {'roll': deque(maxlen=200), 'pitch': deque(maxlen=200), 'yaw': deque(maxlen=200)}
        self.filtered_data = {'roll': deque(maxlen=200), 'pitch': deque(maxlen=200), 'yaw': deque(maxlen=200)}

        # 订阅原始IMU数据和滤波后的IMU数据
        self.create_subscription(Imu, '/imu/data_raw', self.raw_imu_callback, 10)
        self.create_subscription(Imu, '/imu/data', self.filtered_imu_callback, 10)

        # 设置图表
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(10, 8))
        self.fig.suptitle('IMU Data: Raw vs. Filtered')

        # 初始化绘图定时器
        self.timer = self.create_timer(0.1, self.plot_data)

    def raw_imu_callback(self, msg):
        # 将四元数转换为欧拉角
        roll, pitch, yaw = self.quaternion_to_euler(msg.orientation)
        self.raw_data['roll'].append(roll)
        self.raw_data['pitch'].append(pitch)
        self.raw_data['yaw'].append(yaw)

    def filtered_imu_callback(self, msg):
        # 将四元数转换为欧拉角
        roll, pitch, yaw = self.quaternion_to_euler(msg.orientation)
        self.filtered_data['roll'].append(roll)
        self.filtered_data['pitch'].append(pitch)
        self.filtered_data['yaw'].append(yaw)

    def quaternion_to_euler(self, orientation):
        q = orientation
        roll, pitch, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return roll, pitch, yaw

    def plot_data(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        self.ax1.plot(self.raw_data['roll'], label='Raw Roll')
        self.ax1.plot(self.filtered_data['roll'], label='Filtered Roll')
        self.ax1.set_ylabel('Roll')
        self.ax1.legend()

        self.ax2.plot(self.raw_data['pitch'], label='Raw Pitch')
        self.ax2.plot(self.filtered_data['pitch'], label='Filtered Pitch')
        self.ax2.set_ylabel('Pitch')
        self.ax2.legend()

        self.ax3.plot(self.raw_data['yaw'], label='Raw Yaw')
        self.ax3.plot(self.filtered_data['yaw'], label='Filtered Yaw')
        self.ax3.set_ylabel('Yaw')
        self.ax3.legend()

        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = ImuPlotter()
    plt.ion()  # 开启交互式模式
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
