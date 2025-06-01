import numpy as np
import time

def detect_collision_by_diff(imu, threshold, cooldown=1):
    """
    IMU sensöründen gelen her satır için bir önceki adımla farkın normu threshold'u aşarsa çarpışma tespit eder.
    Collision tespitinden sonra cooldown süresi boyunca tekrar tespit yapmaz.
    """
    if not hasattr(detect_collision_by_diff, "prev_imu"):
        detect_collision_by_diff.prev_imu = None
    if not hasattr(detect_collision_by_diff, "last_collision_time"):
        detect_collision_by_diff.last_collision_time = 0

    now = time.time()
    if now - detect_collision_by_diff.last_collision_time < cooldown:
        detect_collision_by_diff.prev_imu = np.copy(imu)
        return 0

    if detect_collision_by_diff.prev_imu is None:
        detect_collision_by_diff.prev_imu = np.copy(imu)
        return 0

    prev = detect_collision_by_diff.prev_imu
    curr = np.array(imu)
    for i in range(curr.shape[0]):
        diff = curr[i, :3] - prev[i, :3]
        diff_norm = np.linalg.norm(diff)
        if diff_norm > threshold:
            detect_collision_by_diff.prev_imu = np.copy(curr)
            detect_collision_by_diff.last_collision_time = now
            return 1
    detect_collision_by_diff.prev_imu = np.copy(curr)
    return 0

def detect_flip_by_imu(imu, angle_threshold=2.5):
    """
    IMU'dan gelen orientation (roll, pitch, yaw) değerlerine bakarak takla tespiti yapar.
    angle_threshold (radyan cinsinden): Roll veya pitch bu değeri aşarsa takla kabul edilir.
    """
    # imu'nun orientation'ı quaternion veya euler olabilir. Burada euler (roll, pitch, yaw) bekleniyor.
    # Eğer quaternion ise, önce euler'a çevrilmeli.
    # imu['orientation'] = [roll, pitch, yaw] veya benzeri bir yapı olmalı.
    orientation = imu
    if orientation is None:
        return False
    roll, pitch, _ = orientation
    if abs(roll) > angle_threshold or abs(pitch) > angle_threshold:
        return True
    return False