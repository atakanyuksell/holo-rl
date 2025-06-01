import numpy as np

def compute_reward(gps_4ago, next_gps, collision, out_of_bounds, next_dist, TARGET, flipped=False):
    pos0_4ago = np.linalg.norm(gps_4ago - TARGET)
    reward = -1.0
    # Hedefe yaklaşma ödülü (4 adım önceki konum ile şimdiki konum farkı)
    if next_dist > pos0_4ago + 1:
        reward += (pos0_4ago - next_dist) * 1000
    
    # Sınır dışı cezası
    if out_of_bounds:
        reward -= 1000
        print ("Out of bounds! Reward penalized.")
    # Çarpışma cezası
    if collision:
        reward -= 1000
        print ("Collision detected! Reward penalized.")
    # Takla cezası
    if flipped:
        reward -= 20000
    # Hedefe ulaşma ödülü
    if next_dist < 2.0:
        reward += 10000
    return reward

def follow_reward(gps0, gps1, collision, min_dist=5.0, collision_penalty=-5.0, close_bonus=0.5, approach_bonus=1.0, pos0_4ago = None , still_penalty=-0.05,pos1_4ago=None):
    """
    gps0: takipçi AUV'nin (auv0) konumu (numpy array veya torch tensor)
    gps1: hedef AUV'nin (auv1) konumu (numpy array veya torch tensor)
    collision: bool
    prev_gps: iki adım önceki takipçi AUV'nin konumu (numpy array veya torch tensor)
    min_dist: Takip için ideal minimum mesafe (metre)
    collision_penalty: Çarpışma olursa verilecek ceza
    close_bonus: Çok yakın takipte ekstra ödül
    approach_bonus: Yaklaşıyorsa ekstra ödül
    pos0_4ago: 4 saniye önceki takipçi AUV'nin konumu (numpy array veya torch tensor)
    still_penalty: Hareketsizlik cezası
    """
    gps0 = np.array(gps0)
    gps1 = np.array(gps1)
    dist = np.linalg.norm(gps0 - gps1)
    
    reward = -dist/100
    print(f"Distance to target: {dist:.2f} m")
    # Çok yakınsa ekstra ödül
    if dist < min_dist:
        reward += close_bonus
        print("Very close to target! Bonus awarded.")
    # İki adım önceye göre yaklaşıyorsa ekstra ödül
    if pos0_4ago is not None:
        dist_4ago = np.linalg.norm(np.array(pos0_4ago) - np.array(pos1_4ago))
        if dist + 0.1< dist_4ago :
            reward += dist_4ago - dist
            print("Approaching target! Approach bonus awarded.")
    # Çarpışma cezası
    if collision:
        reward += collision_penalty
        print("Collision detected! Penalty applied.")
    # 4 saniye hareketsizlik cezası
    if pos0_4ago is not None:
        pos0_4ago = np.array(pos0_4ago)
        if np.allclose(gps0, pos0_4ago, atol=0.1):
            reward += still_penalty
            print("No movement for 4 seconds! Stillness penalty applied.")
    return reward