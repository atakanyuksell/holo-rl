from pynput import keyboard
import numpy as np

pressed_keys = list()

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        if key.char not in pressed_keys:
            pressed_keys.append(key.char)

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        if key.char in pressed_keys:
            pressed_keys.remove(key.char)

def start_listener():
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.daemon = True  # Program kapanınca thread de kapansın
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

def keys_to_onehot(keys):
    key_list = ['i', 'k', 'j', 'l', 'w', 's', 'a', 'd']
    onehot = np.zeros(len(key_list), dtype=np.float32)
    for idx, k in enumerate(key_list):
        if k in keys:
            onehot[idx] = 1.0
    return onehot