#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from tf2_ros import TransformBroadcaster
from tf_conversions import transformations 
from tf.transformations import quaternion_from_matrix

from geometry_msgs.msg import TransformStamped, Vector3Stamped, PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CameraInfo

from hololens_py.hl2ss_py.viewer import hl2ss
from hololens_py.hl2ss_py.viewer import hl2ss_lnm
from hololens_py.hl2ss_py.viewer import hl2ss_mp
from hololens_py.hl2ss_py.viewer import hl2ss_3dcv
from hololens_py.hl2ss_py.viewer import hl2ss_utilities

from scipy.spatial.transform import Rotation as R

# ------------------------- Prepare msg data-----------------------------
def prepare_Image(image, frame_id, desired_encoding='passthrough', stamp=None) -> Image:
    """
    Prepares an Image message from an image array.
    
    Parameters:
        image (np.array): The image data array.
        frame_id (str): The frame ID to set in the message header.
        desired_encoding (str): The encoding type for the image (e.g., 'bgr8', 'mono8').
        stamp (rospy.Time, optional): The timestamp for the header. Defaults to current time if None.

    Returns:
        sensor_msgs/Image: A ROS Image message.
    """
    # Define data
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding=desired_encoding)

    # Define header
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    image_msg.header.stamp = stamp
    image_msg.header.frame_id = frame_id
   
    return image_msg

def prepare_CameraInfo(frame_id, cam_K, cam_P, cam_height, cam_width, stamp=None) -> CameraInfo:
    """
    Prepares a CameraInfo message.

    Parameters:
        cam_K (list): The camera matrix [K] as a list of 9 elements.
        cam_P(list): The projection matrix [P] as a list of 12 elements.
        frame_id (str): The frame ID to set in the message header.
        image_height (int): The height of the image.
        image_width (int): The width of the image.
        stamp (rospy.Time, optional): The timestamp for the header. Defaults to current time if None.

    Returns:
        CameraInfo: A ROS CameraInfo message.
    """
    camera_info = CameraInfo()
    # Define header
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    camera_info.header.stamp = stamp
    camera_info.header.frame_id = frame_id

    # Define other info
    camera_info.height = cam_height
    camera_info.width = cam_width
    camera_info.K = cam_K
    if cam_P is None:
        cam_P = create_projection_matrix(cam_K)
    camera_info.P = cam_P
    # camera_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] # rectification (only for stereo)
    return camera_info

def create_projection_matrix(cam_K):
    """
    Creates a projection matrix from the camera's intrinsic matrix assuming zero translation.

    Parameters:
        cam_K (list): The camera intrinsic matrix [K], provided as a list of 9 elements in row-major order.
                      Format: [K11, K12, K13, K21, K22, K23, K31, K32, K33]

    Returns:
        list: The camera projection matrix [P], returned as a list of 12 elements in row-major order.
              The matrix is a 3x4 matrix with zero translation.
              Format: [K11, K12, K13, 0, K21, K22, K23, 0, K31, K32, K33, 0]
    """
    # Reshape K into a 3x3 matrix
    K = np.array(cam_K).reshape((3, 3))
    
    # Create a 3x4 projection matrix P by appending a zero column to K
    zero_column = np.zeros((3, 1))
    P = np.hstack((K, zero_column))
    
    # Flatten P to a list and return
    return P.flatten().tolist()
    
# ------------------------- Prepare gaze msg (for SAM@)-----------------------------
def prepare_gaze_prompt_msg(stamp, gaze_prompt: np.ndarray) -> Vector3Stamped:
    """
    Prepares a Float32MultiArray message from a gaze prompt array.
    NOTE: The gaze prompt array should be a 1D array without header information.
    """
    gaze_prompt_msg = Vector3Stamped()
    # Flatten the array
    gaze_prompt = gaze_prompt.flatten()
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    gaze_prompt_msg.header.stamp = stamp
    gaze_prompt_msg.header.frame_id = "gaze_prompt"
    
    gaze_prompt_msg.vector.x = gaze_prompt[0]
    gaze_prompt_msg.vector.y = gaze_prompt[1]
    gaze_prompt_msg.vector.z = 0 # no z value

    return gaze_prompt_msg  # Corrected syntax here

# ------------------------- Prepare gaze point msg (for Unity display) -----------------------------
def prepare_gaze_point_msg(stamp, gaze_point: np.ndarray) -> PoseStamped:
    """
    Prepares a Float32MultiArray message from a gaze prompt array.
    NOTE: The gaze prompt array should be a 1D array without header information.
    """
    gaze_point_msg = PoseStamped()
    # Flatten the array
    gaze_point = gaze_point.flatten()
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    gaze_point_msg.header.stamp = stamp
    gaze_point_msg.header.frame_id = "gaze_point"
    
    gaze_point_msg.pose.position.x = gaze_point[0]
    gaze_point_msg.pose.position.y = gaze_point[1]
    gaze_point_msg.pose.position.z = gaze_point[2]

    return gaze_point_msg  # Corrected syntax here

# ------------------------- Prepare gaze msg -----------------------------
def prepare_hand_msg(stamp, hand_prompt: np.ndarray, frame_id) -> Vector3Stamped:
    """
    Prepares a Float32MultiArray message from a hand prompt array.
    NOTE: The gaze prompt array should be a 1D array without header information.
    """
    fingertip_prompt_msg = Vector3Stamped()
    # Flatten the array
    fingertip_prompt = hand_prompt.flatten()
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    fingertip_prompt_msg.header.stamp = stamp
    fingertip_prompt_msg.header.frame_id = frame_id
    
    fingertip_prompt_msg.vector.x = fingertip_prompt[0]
    fingertip_prompt_msg.vector.y = fingertip_prompt[1]
    fingertip_prompt_msg.vector.z = 0 # no z value

    return fingertip_prompt_msg  # Corrected syntax here

# ------------------------- Prepare tf msg -----------------------------
def prepare_tf_msg(frame_id: str, child_frame_id: str, stamp, *args) -> TransformStamped:
    """
    Prepares a TransformStamped message from various input types.
    
    :param frame_id: The ID of the reference frame.
    :param child_frame_id: The ID of the child frame.
    :param args: Variable arguments which can be:
        - (translation, rotation) where translation is [x, y, z] and rotation is 3x3 matrix or quaternion [x, y, z, w].
        - (transformation_matrix) where transformation_matrix is a 4x4 matrix.
    :return: TransformStamped message.
    """
    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = frame_id
    tf_msg.child_frame_id = child_frame_id
    
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].shape == (4, 4):
        # Case 1: 4x4 transformation matrix
        matrix = args[0]
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        quaternion = quaternion_from_matrix(matrix)  # Directly use 4x4 matrix
    elif len(args) == 2:
        translation = args[0]
        if isinstance(args[1], np.ndarray) and args[1].shape == (3, 3):
            # Case 2: translation + rotation matrix
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = args[1]
            quaternion = quaternion_from_matrix(rotation_matrix)  # Use 4x4 matrix
        elif isinstance(args[1], list) or isinstance(args[1], np.ndarray) and len(args[1]) == 4:
            # Case 3: translation + quaternion
            quaternion = args[1]
        else:
            raise ValueError("Invalid rotation data type or size.")
    else:
        raise ValueError("Invalid arguments for TF preparation.")
    
    # Set translation
    tf_msg.transform.translation.x = translation[0]
    tf_msg.transform.translation.y = translation[1]
    tf_msg.transform.translation.z = translation[2]
    
    # Set rotation
    tf_msg.transform.rotation.x = quaternion[0]
    tf_msg.transform.rotation.y = quaternion[1]
    tf_msg.transform.rotation.z = quaternion[2]
    tf_msg.transform.rotation.w = quaternion[3]
    
    return tf_msg

# ------------------------ change camera frame to OpenCV -----------------------------
# pv (OpenGl) to OpenCV frame
def camera_pose_gl_to_cv(camera_pose):
    """
    Converts a camera pose from OpenGL coordinates to OpenCV coordinates.

    Parameters:
        camera_pose (np.array): The camera pose in the world frame in OpenGL coordinates.

    Returns:
        np.array: The camera pose in the world frame in OpenCV coordinates.
    """
    T_w_gl = camera_pose  # Camera pose in OpenGL coordinates to world frame (OpenGL)
    T_gl_cv = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])  # Transformation matrix from OpenGL to OpenCV coordinates
    T_w_cv = T_w_gl @ T_gl_cv  # Convert to OpenCV coordinates
    return T_w_cv

# ------------------------ Remap color and depth -----------------------------
# get pv_uv which is important for remapping depth and color
def get_remap_factor(calibration_lt, calibration_pv, data_lt, data_pv, depth):
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    # Get color calibration
    color_intrinsics, color_extrinsics = get_color_calibration(calibration_pv)
    # Remap color image
    lt_points         = hl2ss_3dcv.rm_depth_to_points(xy1, depth)
    lt_to_world       = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
    world_to_lt       = hl2ss_3dcv.world_to_reference(data_lt.pose) @ hl2ss_3dcv.rignode_to_camera(calibration_lt.extrinsics)
    world_to_pv_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)
    world_points      = hl2ss_3dcv.transform(lt_points, lt_to_world)
    pv_uv             = hl2ss_3dcv.project(world_points, world_to_pv_image)

    return pv_uv

def remap_pv(remap_factor, image_pv):
    pv_uv = remap_factor
    color_remap = cv2.remap(image_pv, pv_uv[:, :, 0], pv_uv[:, :, 1], cv2.INTER_LINEAR)
    return color_remap

def remap_lt(remap_factor, depth, pv_width, pv_height):
    pv_uv = remap_factor
    mask_uv = hl2ss_3dcv.slice_to_block((pv_uv[:, :, 0] < 0) | (pv_uv[:, :, 0] >= pv_width) | (pv_uv[:, :, 1] < 0) | (pv_uv[:, :, 1] >= pv_height))
    depth[mask_uv] = 0
    return depth

# helper
def get_color_calibration(calibration_pv):
    calibration_pv.intrinsics, calibration_pv.extrinsics = hl2ss_3dcv.pv_fix_calibration(calibration_pv.intrinsics, np.eye(4, 4, dtype=np.float32))
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(calibration_pv.intrinsics, calibration_pv.extrinsics)
    return color_intrinsics, color_extrinsics

# ------------------------ Get gaze prompt -----------------------------
# TODO: use spatial mapping for gaze 
def get_gaze_prompt(data_eet, data_pv, extrinsics, intrinsics, d=5):
    eet = hl2ss.unpack_eet(data_eet.payload)
    world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(extrinsics) @ hl2ss_3dcv.camera_to_image(intrinsics)
    local_combined_ray = hl2ss_utilities.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
    combined_ray = hl2ss_utilities.si_ray_transform(local_combined_ray, data_eet.pose)
    # d = 5
    combined_point = hl2ss_utilities.si_ray_to_point(combined_ray, d) 
    combined_image_point = hl2ss_3dcv.project(combined_point, world_to_image)
    gaze_prompt = np.array([[combined_image_point[0][0], combined_image_point[0][1]]])
    # Return gaze prompt and gaze point
    return gaze_prompt, combined_point

# ------------------------ Get fingertip prompt -----------------------------

# def get_offset_fingertip_point(data_fingertip, offset = 0.015):
#     finger_tip_x = data_fingertip.pose.position.x
#     finger_tip_y = data_fingertip.pose.position.y
#     finger_tip_z = data_fingertip.pose.position.z
#     finger_tip_quaternion = [data_fingertip.pose.orientation.x, data_fingertip.pose.orientation.y, data_fingertip.pose.orientation.z, data_fingertip.pose.orientation.w]
#     finger_tip_rotation_matrix = R.from_quat(finger_tip_quaternion).as_matrix()
#     finger_tip_y_axis = finger_tip_rotation_matrix[:, 1]
#     finger_tip_extended_normal = np.array([[finger_tip_x, finger_tip_y, finger_tip_z]]) - offset* finger_tip_y_axis
#     return finger_tip_extended_normal

# def get_hand_prompt(fingertip_point, si, data_pv, pv_extrinsics, pv_intrinsics):
#     right_hand = si.get_hand_right()
#     right_joints = hl2ss_utilities.si_unpack_hand(right_hand)
#     world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
#     if fingertip_point == "index_tip":
#         fingertip_point = right_joints.positions[10]
#     elif fingertip_point == "middle_tip":
#         fingertip_point = right_joints.positions[15]
#     elif fingertip_point == "thumb_tip":
#         fingertip_point = right_joints.positions[5]
#     else:
#         return None
#     hand_prompt = hl2ss_3dcv.project(fingertip_point, world_to_image)
#     return hand_prompt
   
   
def get_fingertip_prompt(finger_name:str, si, data_pv, pv_extrinsics, pv_intrinsics, hand ='right'):
    if hand == 'right':
        hand = si.get_hand_right()
    elif hand == 'left':
        hand = si.get_hand_left()

    world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
    
    if finger_name == "index_tip":
        fingertip_pt = hand.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
    elif finger_name == "middle_tip":
        fingertip_pt = hand.get_joint_pose(hl2ss.SI_HandJointKind.MiddleTip).position
    elif finger_name == "thumb_tip":
        fingertip_pt = hand.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip).position
    else:
        return None
    finger_prompt = hl2ss_3dcv.project(fingertip_pt, world_to_image)
    return finger_prompt