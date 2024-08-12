from setuptools import find_packages, setup
import os

package_name = 'yolo_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), [os.path.join('resource', 'best.pt')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ocar1053',
    maintainer_email='ocar1053@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_pokemon_node = yolo_pkg.pokemon_node:main',
            'compressed_node = yolo_pkg.compress:main',
            'imu_test_node = yolo_pkg.imu_test:main',
            'camera_depth_test_node = yolo_pkg.camera_test:main'
        ],
    },
)
