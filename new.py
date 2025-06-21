import holoocean
import numpy as np
from pynput import keyboard
from helpers.observation_utils import process_observation, pose_to_euler, get_pose_from_auv,pose_to_gps, pose_to_rotation, velocity_to_speed_and_direction

pressed_keys = list()
force = 25
target_location = [-547,107,-257]
def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    return command

with holoocean.make("OpenWater-HoveringCamera") as env:
    while True:
        if 'q' in pressed_keys:
            break
        command = parse_keys(pressed_keys, force)

        #send to holoocean
        env.act("auv0", command)
        state = env.tick()

        if 'PoseSensor' in state and 'IMUSensor' in state:
            pose = state['PoseSensor']
            imu = state['IMUSensor']
            gps = pose_to_gps(state)
            yaw,pitch,roll = pose_to_euler(pose)
            
            # Target location ile GPS arasındaki vektörü hesapla
            target_vector = np.array(target_location) - np.array(gps)
            
            # Hedef yaw değerini hesapla (target'a bakmak için gereken yaw)
            target_yaw = np.arctan2(target_vector[1], target_vector[0]) * 180.0 / np.pi
            
            # Yaw farkını hesapla
            yaw_diff = target_yaw - yaw
            
            # -180 ile +180 arasına normalize et
            while yaw_diff > 180:
                yaw_diff -= 360
            while yaw_diff < -180:
                yaw_diff += 360
            
            
            # Mesafe hesapla
            distance = np.linalg.norm(target_vector)
            command = target_location + [0,0,target_yaw]
            print(f"Current Yaw: {yaw:.2f}° | Target Yaw: {target_yaw:.2f}° | Yaw Diff: {yaw_diff:.2f}° | Distance: {distance:.2f}")
         