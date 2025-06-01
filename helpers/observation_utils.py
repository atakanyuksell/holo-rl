import torch
import numpy as np

def process_observation(auv):
    camera_img = auv['Camera'][..., :3] / 255.0
    camera_img = camera_img.transpose(2, 0, 1)
    camera_img = torch.from_numpy(camera_img).float()
    assert camera_img.shape == (3, 256, 256), f"camera_img shape hatalı: {camera_img.shape}"
    pose = np.array(auv['PoseSensor'])
    gps = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])
    imu = auv['IMUSensor']
    return camera_img, gps, imu

def pose_to_gps(auv):
    """
    PoseSensor'dan GPS koordinatlarını çıkarır.
    """
    pose = np.array(auv['PoseSensor'])
    gps = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])
    return gps

def pose_to_rotation(auv):
    """
    PoseSensor'dan rotasyon matrisini çıkarır.
    """
    pose = np.array(auv['PoseSensor'])
    rotation = pose[:3, :3]
    return rotation