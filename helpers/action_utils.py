import numpy as np

ALLOWED_ACTIONS = [
    np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    np.array([0, 0, 0, 0, 1, -1, 1, -1]),
    np.array([0, 0, 0, 0, -1, -1, -1, -1]),
    np.array([0, 0, 0, 0, -1, 1, -1, 1]),
    np.array([0, 0, 0, 0, 1, -1, -1, 1]),
    np.array([-1, -1, -1, -1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, -1, 1, 1, -1]),
    np.array([1, 1, 1, 1, 0, 0, 0, 0]),
    # #combines
    # np.array([0, 0, 0, 0, 2, 0, 2, 0]),
    # np.array([0, 0, 0, 0, 0, 0, 0, 0]),
    # np.array([0, 0, 0, 0, 0, 2, 0, 2]),
    # np.array([0, 0, 0, 0, 2, 0, 0, 2]),
    # np.array([-1, -1, -1, -1, 1, 1, 1, 1]),
    # np.array([0, 0, 0, 0, 0, 2, 2, 0]),
    # np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    # np.array([0, 0, 0, 0, 0, -2, 0, -2]),
    # np.array([0, 0, 0, 0, 2, -2, 0, 0]),
    # np.array([-1, -1, -1, -1, 1, -1, 1, -1]),
    # np.array([0, 0, 0, 0, 0, 0, 2, -2]),
    # np.array([1, 1, 1, 1, 1, -1, 1, -1]),
    # np.array([0, 0, 0, 0, -2, 0, -2, 0]),
    # np.array([0, 0, 0, 0, 0, -2, -2, 0]),
    # np.array([-1, -1, -1, -1, -1, -1, -1, -1]),
    # np.array([0, 0, 0, 0, -2, 0, 0, -2]),
    # np.array([1, 1, 1, 1, -1, -1, -1, -1]),
    # np.array([0, 0, 0, 0, 0, 0, -2, 2]),
    # np.array([-1, -1, -1, -1, -1, 1, -1, 1]),
    # np.array([0, 0, 0, 0, -2, 2, 0, 0]),
    # np.array([1, 1, 1, 1, -1, 1, -1, 1]),
    # np.array([-1, -1, -1, -1, 1, -1, -1, 1]),
    # np.array([1, 1, 1, 1, 1, -1, -1, 1]),
    # np.array([-1, -1, -1, -1, -1, 1, 1, -1]),
    # np.array([1, 1, 1, 1, -1, 1, 1, -1]),

]

def filter_to_allowed_action(action):
    """
    3 elemanlı 0/1 action vektörünü base_actions'daki 8 elemanlı array'lere eşler.
    [0,0,0] -> [0,0,0,0,0,0,0,0]
    [1,0,0] -> base_actions[0], [0,1,0] -> base_actions[1], [0,0,1] -> base_actions[2],
    [1,1,0] -> base_actions[3], [1,0,1] -> base_actions[4], [0,1,1] -> base_actions[5], [1,1,1] -> base_actions[6]
    """
    action = np.array(action).astype(int)
    base_actions = [
        np.array([0,0,0,0,1,1,1,1]),    # 0: ileri
        np.array([0,0,0,0,-1,-1,-1,-1]), # 1: geri
        np.array([0,0,0,0,-1,1,-1,1]),   # 2: sağa
        np.array([0,0,0,0,1,-1,-1,1]),   # 3: sola
        np.array([1,1,1,1,0,0,0,0]),     # 4: yukarı
        np.array([-1,-1,-1,-1,0,0,0,0]), # 5: aşağı
        np.array([0,0,0,0,-1,1,1,-1]),     # 6: sol rotasyon
        np.array([1,1,1,1,0,0,0,0]),     # 7: sağ rotasyon
    ]
    # 3 bitlik binary sayıyı index olarak kullan
    idx = (action[0] << 2) | (action[1] << 1) | action[2]
    if idx == 0:
        return np.zeros(8, dtype=int)
    if 1 <= idx <= 7:
        return base_actions[idx-1]
    return np.zeros(8, dtype=int)

def get_command(action_idx, action_to_key, force, parse_keys):
    """
    Verilen eylem indeksine karşilik gelen komutu, anahtara eşleyip ayriştirarak döndürür.

    Argümanlar:
        action_idx (int): Komutun alinacaği eylem indeksi.
        action_to_key (dict veya list): Eylem indekslerini anahtar temsillerine eşleyen bir yapi.
        force (Herhangi): parse_keys fonksiyonuna iletilecek ek parametre.
        parse_keys (callable): Bir anahtar ve force parametresini alip, ayriştirilmiş komutu döndüren fonksiyon.

    Döndürür:
        Herhangi: Eylem indeksine karşilik gelen anahtarin parse_keys fonksiyonu ile ayriştirilmiş sonucu.
    """
    pressed = action_to_key[action_idx]
    return parse_keys(pressed, force)

def filter_action_salvo(current_action, prev_action, max_delta=0.5):
    """
    Prevents rapid, unrealistic changes (salvos) in the action vector.
    Limits the change in each action dimension to max_delta per step.
    Args:
        current_action (np.ndarray): The action proposed by the agent.
        prev_action (np.ndarray): The action taken in the previous step.
        max_delta (float): Maximum allowed change per action dimension.
    Returns:
        np.ndarray: The filtered action vector.
    """
    if prev_action is None:
        return current_action
    current_action = np.array(current_action)
    prev_action = np.array(prev_action)
    delta = current_action - prev_action
    delta = np.clip(delta, -max_delta, max_delta)
    filtered_action = prev_action + delta
    return filtered_action
