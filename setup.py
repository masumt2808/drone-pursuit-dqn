from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'drone_pursuit'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='masum',
    maintainer_email='masum@todo.todo',
    description='Vision-Based Autonomous Drone Pursuit Using Deep Q-Networks',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'evader_node = drone_pursuit.evader_node:main',
            'train = drone_pursuit.train:main',
            'evaluate = drone_pursuit.evaluate:main',
        ],
    },
)
