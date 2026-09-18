from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'mml_guidance'
setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py') + glob('launch/*.launch')),
    ],
    install_requires=['setuptools', 'mml-core==0.1.0'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='MML Guidance Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'guidance = mml_guidance.main_guidance_node:main',
            'markov_goal_pose = mml_guidance.markov_goal_pose:main',
            'turtlebot_pid_controller = mml_guidance.turtlebot_pid_controller:main',
            'mml_pf_visualization = mml_guidance.mml_pf_visualization:main',
            'rl_tracker = mml_guidance.rl_offboard_tracker:main',
            'boundary_safety_test = mml_guidance.boundary_safety_test:main',
            'boundary_visualization = mml_guidance.boundary_visualization:main',
            'simulated_uav_tracker = mml_guidance.sim_uav_tracker:main',
            'simulated_turtlebot = mml_guidance.Sim_turtle:main',
            'Sim_turtle = mml_guidance.Sim_turtle:main',
            'go_here = mml_guidance.go_here:main',
            'go_here_square = mml_guidance.go_here_square:main',
            'rl_policy_node = mml_guidance.rl_policy_node:main',
            'drone_commander = mml_guidance.drone_commander:main',
            'monitor_tracking = mml_guidance.monitor_tracking:main',
        ],
    },                                                                                      
)
