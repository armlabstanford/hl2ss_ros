#!/usr/bin/env python3

import multiprocessing as mp
import numpy as np
from numpy.linalg import inv
import open3d as o3d
import cv2
# from hololens_py.hl2ss_py.viewer import hl2ss_imshow
from hololens_py.hl2ss_py.viewer import hl2ss
from hololens_py.hl2ss_py.viewer import hl2ss_lnm
from hololens_py.hl2ss_py.viewer import hl2ss_mp
from hololens_py.hl2ss_py.viewer import hl2ss_3dcv
from hololens_py.hl2ss_py.viewer import hl2ss_sa,hl2ss_utilities

import rospy
import hl2ss_ros_utils as utils
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from tf2_ros import TransformBroadcaster

HANDOVER_OFFSET_POSITION = [0, -0.15, 0]
HANDOVER_OFFSET_ORIENTATION = [-1/np.sqrt(2), 0, 0, 1/np.sqrt(2)]

# Settings --------------------------------------------------------------------
rospy.init_node("hl2ss_node")
rospy.loginfo("(Hololens_client) Hololens2 Client Node Initialized")
host = rospy.get_param('host') # HoloLens address (global)
calibration_path = rospy.get_param('~calibration_path') # within this node
lt_calibration_path = rospy.get_param('~lt_calibration_path') # within this node
# Front RGB camera parameters
pv_focus = 1000 # In mm
pv_width = 640
pv_height = 360
pv_framerate = 30

# Gaze (EET) parameters
eet_framerate = 30  

buffer_length = 10 # [s]
max_depth = 3.0 # [m]

# Hand settings
default_hand = 'right'

# Spatial Mapping manager settings
triangles_per_cubic_meter = 200
mesh_threads = 2
sphere_center = [0, 0, 0]
sphere_radius = 2

if __name__ == '__main__':
    # Init ROS related
    pub_img_lt = rospy.Publisher("/hololens2/image_lt", Image, queue_size=10)
    pub_img_pv_remap = rospy.Publisher("/hololens2/image_pv_remap", Image, queue_size=10)
    pub_info_lt = rospy.Publisher("/hololens2/camerainfo_lt", CameraInfo, queue_size=10)

    pub_img_pv = rospy.Publisher("/camera/image_rect", Image, queue_size=10)
    pub_info_pv = rospy.Publisher("/camera/camera_info", CameraInfo, queue_size=10)

    pub_gaze_eet = rospy.Publisher("/hololens2/gaze_eet", Vector3Stamped, queue_size=10)
    pub_gaze_point = rospy.Publisher("/hololens2/gaze_point_raw", PoseStamped, queue_size=10)

    pub_remap_factor = rospy.Publisher("/hololens2/remap_factor", Image, queue_size=10)

    pub_index_tip = rospy.Publisher("/hololens2/index_tip", Vector3Stamped, queue_size=10)
    pub_middle_tip = rospy.Publisher("/hololens2/middle_tip", Vector3Stamped, queue_size=10)
    pub_thumb_tip = rospy.Publisher("/hololens2/thumb_tip", Vector3Stamped, queue_size=10)
    
    br = TransformBroadcaster()

    # Start Spatial Mapping subsystem ------------------------------------------
    # Set region of 3D space to sample
    volumes = hl2ss.sm_bounding_volume()
    volumes.add_sphere(sphere_center, sphere_radius)
    # Download observed surfaces
    sm_manager = hl2ss_sa.sm_manager(host, triangles_per_cubic_meter, mesh_threads)
    sm_manager.open()
    sm_manager.set_volumes(volumes)
    sm_manager.get_observed_surfaces()
        
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Fix PV focus -------------------------------------------------------
    rc_client = hl2ss_lnm.ipc_rc(host, hl2ss.IPCPort.REMOTE_CONFIGURATION)
    rc_client.open()
    rc_client.wait_for_pv_subsystem(True)
    rc_client.set_pv_focus(hl2ss.PV_FocusMode.Manual, hl2ss.PV_AutoFocusRange.Normal, hl2ss.PV_ManualFocusDistance.Infinity, pv_focus, hl2ss.PV_DriverFallback.Disable)
    rc_client.close()

    # Get calibration (focus is fixed so intrinsics don't change between frames)
    calibration_pv = hl2ss_3dcv.get_calibration_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, calibration_path, pv_focus, pv_width, pv_height, pv_framerate)
    calibration_pv.intrinsics, calibration_pv.extrinsics = hl2ss_3dcv.pv_fix_calibration(calibration_pv.intrinsics, np.eye(4, 4, dtype=np.float32))
    pv_K = calibration_pv.intrinsics.T[:3, :3].flatten().tolist()
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(calibration_pv.intrinsics, calibration_pv.extrinsics)
    
    # Start PV and RM Depth Long Throw streams --------------------------------
    # 1. PRODUCER
    producer = hl2ss_mp.producer()
    # Start PV
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO, hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height, framerate=pv_framerate, decoded_format='rgb24'))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_framerate * buffer_length)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    # Start LT (depth)
    producer.configure(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss_lnm.rx_rm_depth_longthrow(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW))
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS * buffer_length)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)
    # Start EET (gaze)
    producer.configure(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss_lnm.rx_eet(host, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=eet_framerate))
    producer.initialize(hl2ss.StreamPort.EXTENDED_EYE_TRACKER, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.EXTENDED_EYE_TRACKER)
    # Start SI (hand)
    producer.configure(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss_lnm.rx_si(host, hl2ss.StreamPort.SPATIAL_INPUT))
    producer.initialize(hl2ss.StreamPort.SPATIAL_INPUT, hl2ss.Parameters_SI.SAMPLE_RATE * buffer_length)
    producer.start(hl2ss.StreamPort.SPATIAL_INPUT)

    # 2. MANAGER
    manager = mp.Manager()
    # 3. CONSUMER
    consumer = hl2ss_mp.consumer()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_pv.get_attach_response()

    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, ...)
    sink_depth.get_attach_response()

    sink_eet = consumer.create_sink(producer, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, manager, ...)
    sink_eet.get_attach_response()

    sink_si = consumer.create_sink(producer, hl2ss.StreamPort.SPATIAL_INPUT, manager, None)
    sink_si.get_attach_response()
    rospy.loginfo("(Hololens_client) Starting Hololens2 communication for RGBD and gaze streams (pv + lt + eet)")

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, lt_calibration_path)
    # DEPTH CAMERA PARAMETERS (for publisher)
    lt_K = calibration_lt.intrinsics.T[:3, :3].flatten().tolist()
    lt_height = hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT
    lt_width = hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH
 
    uv2xy = hl2ss_3dcv.compute_uv2xy(calibration_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    # Main Loop ---------------------------------------------------------------
    while not rospy.is_shutdown():
        # (Spatial Mapping) Download observed surfaces ------------------------------------------
        sm_manager.get_observed_surfaces()

        # Wait for RM Depth Long Throw frame ----------------------------------
        sink_depth.acquire()

        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        _, data_lt = sink_depth.get_most_recent_frame()
        if ((data_lt is None) or (not hl2ss.is_valid_pose(data_lt.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_lt.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue

        _, data_eet = sink_eet.get_nearest(data_lt.timestamp)
        # print("Got eet frame")
        eet = hl2ss.unpack_eet(data_eet.payload)
        if ((data_eet is None) or (not hl2ss.is_valid_pose(data_eet.pose)) or (not eet.combined_ray_valid)):
            rospy.loginfo_once("(Hololens_client) No valid EET data, waiting...")
            continue

        _, data_si = sink_si.get_nearest(data_pv.timestamp)
        # print("Got si frame")
        si = hl2ss.unpack_si(data_si.payload)
        if default_hand == 'right':
            if (data_si is None) or (not si.is_valid_hand_right()):
                rospy.loginfo_once("(Hololens_client) No valid SI data, waiting...")
                continue
        elif default_hand == 'left':
            if (data_si is None) or (not si.is_valid_hand_left()):
                rospy.loginfo_once("(Hololens_client) No valid SI data, waiting...")
                continue

        # Preprocess frames ---------------------------------------------------
        # Raw color and depth
        depth = hl2ss_3dcv.rm_depth_undistort(data_lt.payload.depth, calibration_lt.undistort_map)
        depth = hl2ss_3dcv.rm_depth_normalize(depth, scale)
        color = data_pv.payload.image

        # Remap color and depth
        remap_factor = utils.get_remap_factor(calibration_lt=calibration_lt, data_lt=data_lt, 
                                            calibration_pv=calibration_pv, data_pv=data_pv,
                                            depth = depth) # image but two channels
        color_remap = utils.remap_pv(remap_factor=remap_factor, image_pv=color)
        depth = utils.remap_lt(remap_factor=remap_factor, depth=depth, pv_height=pv_height, pv_width=pv_width)

        # Gaze point prompt
        # TODO: Need clean up
        local_combined_ray = hl2ss_utilities.si_ray_to_vector(eet.combined_ray.origin, eet.combined_ray.direction)
        combined_ray = hl2ss_utilities.si_ray_transform(local_combined_ray, data_eet.pose)
        d = sm_manager.cast_rays(combined_ray)
        if not (np.isfinite(d)):
            d = 0.9

        gaze_prompt, gaze_point = utils.get_gaze_prompt(data_eet=data_eet, data_pv=data_pv, 
                                                        extrinsics=color_extrinsics, 
                                                        intrinsics=color_intrinsics,
                                                        d=d)

        # Right hand prompts
        index_tip_prompt = utils.get_fingertip_prompt(finger_name="index_tip", si = si, 
                                                 data_pv = data_pv, 
                                                 pv_extrinsics = color_extrinsics, 
                                                 pv_intrinsics = color_intrinsics)
        middle_tip_prompt = utils.get_fingertip_prompt(finger_name="middle_tip", si = si, 
                                                 data_pv = data_pv, 
                                                 pv_extrinsics = color_extrinsics, 
                                                 pv_intrinsics = color_intrinsics)
        thumb_tip_prompt = utils.get_fingertip_prompt(finger_name="thumb_tip", si = si, 
                                                 data_pv = data_pv, 
                                                 pv_extrinsics = color_extrinsics, 
                                                 pv_intrinsics = color_intrinsics)
        hand_right = si.get_hand_right()
        palm_pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Palm)

         
        # PREPARE MSG FOR PUBLISHER -------------------------------------------------------------
        # Define the TimeStamp (same across all pub per cycle)
        stamp = rospy.Time.now()
        Image_pv_remap = utils.prepare_Image(stamp=stamp,
                                             image=color_remap,
                                             frame_id='lt',
                                             desired_encoding='rgb8')
        Image_lt = utils.prepare_Image(stamp=stamp,
                                       image=depth,
                                       frame_id='lt',
                                       desired_encoding='32FC1')
        CameraInfo_lt = utils.prepare_CameraInfo(stamp=stamp,
                                                 frame_id='lt',
                                                 cam_K=lt_K,
                                                 cam_P=None,
                                                 cam_height=lt_height,
                                                 cam_width=lt_width)
        # RGB (pv frame)
        Image_pv = utils.prepare_Image(stamp=stamp,
                                       image=color,
                                       frame_id='pv',
                                       desired_encoding='rgb8')
        CameraInfo_pv = utils.prepare_CameraInfo(stamp=stamp,
                                                 frame_id='camera',
                                                 cam_K=pv_K,
                                                 cam_P=None,
                                                 cam_height=pv_height,
                                                 cam_width=pv_width)
        
        # Remap pv_uv -> for grasp mask
        pv_uv_msg = utils.prepare_Image(stamp=stamp,
                                        image=remap_factor, 
                                        frame_id='pv_uv', 
                                        desired_encoding='passthrough')
        # Gaze prompt (eet)
        Vector3Stamped_gaze = utils.prepare_gaze_prompt_msg(stamp=stamp,
                                                            gaze_prompt=gaze_prompt)
        PoseStamped_gaze_point = utils. prepare_gaze_point_msg(stamp=stamp,
                                                               gaze_point=gaze_point)
    
        # Fingertip prompts
        index_msg = utils.prepare_hand_msg(stamp=stamp, hand_prompt=index_tip_prompt, frame_id='index_tip')
        middle_msg = utils.prepare_hand_msg(stamp=stamp, hand_prompt=middle_tip_prompt, frame_id='middle_tip')
        thumb_msg = utils.prepare_hand_msg(stamp=stamp, hand_prompt=thumb_tip_prompt, frame_id='thumb_tip')

        # PUBLISH -------------------------------------------------------------------
        rospy.loginfo_once("(Hololens_client) All data valid, publishing... (NOTE: sometime EET data is not valid)")
        # Aligned RGBD, using lt camera info
        pub_img_pv_remap.publish(Image_pv_remap)
        pub_img_lt.publish(Image_lt)
        pub_info_lt.publish(CameraInfo_lt)

        # raw RGB, using pv camera info
        pub_img_pv.publish(Image_pv)
        pub_info_pv.publish(CameraInfo_pv)

        # Gaze prompt
        pub_gaze_eet.publish(Vector3Stamped_gaze)
        pub_gaze_point.publish(PoseStamped_gaze_point)

        # Remap factor
        pub_remap_factor.publish(pv_uv_msg)

        # Fingertip prompts
        pub_index_tip.publish(index_msg)
        pub_middle_tip.publish(middle_msg)
        pub_thumb_tip.publish(thumb_msg)

        # IMPORTANT: This is how the lt frame /tf problem is fixed
        lt_extrinsics = calibration_lt.extrinsics.T
        lt_pose = data_lt.pose.T
        pv_pose = data_pv.pose.T
        
        tf_lt = utils.prepare_tf_msg('hl_world', 'lt', stamp, lt_pose @ inv(lt_extrinsics)) # lt_pose is rignode in world frame, extrinsic is camera in rignode frame
        tf_pv = utils.prepare_tf_msg('hl_world', 'pv', stamp, utils.camera_pose_gl_to_cv(pv_pose)) 
    
        tf_palm = utils.prepare_tf_msg('hl_world', 'palm', stamp, palm_pose.position, palm_pose.orientation) # check camera frame
        tf_handover = utils.prepare_tf_msg('palm', 'handover_tmp', stamp, HANDOVER_OFFSET_POSITION, HANDOVER_OFFSET_ORIENTATION) # offset & flip from plam to handover

        br.sendTransform([tf_palm, tf_handover, tf_lt, tf_pv])

    # Stop Data streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO) # pv
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW) # lt
    producer.stop(hl2ss.StreamPort.EXTENDED_EYE_TRACKER) # eet
    producer.stop(hl2ss.StreamPort.SPATIAL_INPUT) # si

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

