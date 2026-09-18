import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from pupil_apriltags import Detector

class RealSensePoseNode(Node):
    def __init__(self):
        super().__init__('realsense_pose_node')

        # --- CONFIGURATION ---
        self.topic_name = '/camera/camera/color/image_raw'
        self.tag_family = 'tag36h11'

        # 1. PHYSICAL TAG SIZE (Meters)
        # Measure the black square edge. Example: 5cm = 0.05
        self.tag_size = 0.05 

        # 2. CAMERA INTRINSICS [fx, fy, cx, cy]
        # Replace these with values from 'ros2 topic echo /camera/camera/color/camera_info'
        # Approximate values for RealSense at 640x480:
        self.camera_params = [386.0, 386.0, 320.0, 240.0] 
        # ---------------------

        self.bridge = CvBridge()
        self.detector = Detector(families=self.tag_family)
        
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            qos_policy
        )
        
        self.get_logger().info(f"Looking for {self.tag_family} with pose estimation...")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # --- MODIFIED DETECTION ---
            # We pass estimate_tag_pose=True and the camera parameters
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=self.camera_params,
                tag_size=self.tag_size
            )

            for tag in detections:
                # 1. Extract Pose (Position of Tag in Camera Frame)
                # pose_t is a translation vector [x, y, z] in meters
                # pose_R is a 3x3 rotation matrix
                t_vec = tag.pose_t
                r_mat = tag.pose_R
                
                # 'z' is the distance forward from the camera
                dist = t_vec[2][0] 
                
                # --- VISUALIZATION ---
                # Draw the bounding box
                for i in range(4):
                    start = tuple(tag.corners[i].astype(int))
                    end = tuple(tag.corners[(i + 1) % 4].astype(int))
                    cv2.line(frame, start, end, (0, 255, 0), 2)

                # Draw ID and Position
                center = tuple(tag.center.astype(int))
                label = f"ID:{tag.tag_id} Z:{dist:.2f}m"
                cv2.putText(frame, label, (center[0], center[1]-15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Print full coordinates to terminal
                print(f"ID: {tag.tag_id} | X: {t_vec[0][0]:.3f} | Y: {t_vec[1][0]:.3f} | Z: {t_vec[2][0]:.3f}")

                # (Optional) Draw 3D Axes
                self.draw_axes(frame, self.camera_params, tag.pose_R, tag.pose_t, self.tag_size)

            cv2.imshow("RealSense Pose Estimation", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

    def draw_axes(self, img, camera_params, R, t, tag_size):
        """
        Helper function to draw 3D axes on the tag
        """
        fx, fy, cx, cy = camera_params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Define axis length (half the tag size)
        len = tag_size / 2
        
        # 3D points of the axes (Origin, X, Y, Z)
        points_3d = np.array([
            [0, 0, 0],       # Origin
            [len, 0, 0],     # X axis
            [0, len, 0],     # Y axis
            [0, 0, -len]     # Z axis (pointing up out of tag)
        ])
        
        # Project 3D points to 2D image plane
        points_2d, _ = cv2.projectPoints(points_3d, R, t, K, distCoeffs=None)
        points_2d = points_2d.astype(int)
        
        origin = tuple(points_2d[0].ravel())
        img = cv2.line(img, origin, tuple(points_2d[1].ravel()), (0, 0, 255), 2) # X = Red
        img = cv2.line(img, origin, tuple(points_2d[2].ravel()), (0, 255, 0), 2) # Y = Green
        img = cv2.line(img, origin, tuple(points_2d[3].ravel()), (255, 0, 0), 2) # Z = Blue (Actually points into tag in this lib usually)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(RealSensePoseNode())
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
