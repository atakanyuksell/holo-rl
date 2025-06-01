import holoocean
import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers.td3_vision_agent import TD3VisionAgent
from helpers.key_control import start_listener, pressed_keys, parse_keys
from helpers.observation_utils import process_observation, pose_to_gps, pose_to_rotation
from helpers.collision import detect_collision_by_diff
from helpers.reward_utils import follow_reward
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter  # <-- TensorBoard importu
start_listener()

def main():
    # SAC agent başlat (girdi boyutlarını kendi ortamınıza göre ayarlayın)
    writer = SummaryWriter(log_dir="runs/holopd")  # <-- TensorBoard writer başlat
    agent = TD3VisionAgent(
        img_shape=(3, 256, 256),
        collision_dim=1,
        distance_dim=1,
        rotation_dim=9,
        action_dim=8,
        device="cuda",
        writer=writer
    )
    model_path = "td3_agent.pth"
    # Eğer model dosyası varsa yükle
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=agent.device)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.actor_target.load_state_dict(checkpoint['actor_target'])
        agent.critic1.load_state_dict(checkpoint['critic1'])
        agent.critic2.load_state_dict(checkpoint['critic2'])
        agent.critic1_target.load_state_dict(checkpoint['critic1_target'])
        agent.critic2_target.load_state_dict(checkpoint['critic2_target'])
        print("Model checkpoint yüklendi!")
    else:
        print("Model checkpoint bulunamadı, sıfırdan başlanıyor.")
    
    with holoocean.make("OpenWater-HoveringCamera",show_viewport=True) as env:
        
        feature_fig, feature_axes = plt.subplots(1, 8, figsize=(16, 2))
        episode_reward = 0
        episode = 0
        max_steps = 1000
        step = 0
        obs_prev = None
        gps_prev2 = None  # iki adım önceki gps
        pos0_history = deque(maxlen=5)  # 4 saniye öncesi için 5 elemanlık kuyruk
        pos1_history = deque(maxlen=5)
        action = np.zeros(agent.action_dim)  # <-- action'ı başta sıfırla
        while True:
            command = parse_keys(pressed_keys, 15)
            env.act("auv1", command)
            state = env.tick()
            auv0 = state['auv0']
            auv1 = state['auv1']
            if "Camera" in auv0 and "PoseSensor" in auv0 and "IMUSensor" in auv0:
                camera_img0 , gps0 , imu0 = process_observation(auv0)
                
                rotation = pose_to_rotation(auv0)
                rotation_flat = np.array(rotation).reshape(-1)  # (3,3) -> (9,)
                collision = detect_collision_by_diff(imu0, threshold=10)
                gps1 = pose_to_gps(auv1)
                
                # --- DISTANCE HESAPLAMA ---
                distance_auv0_auv1 = np.linalg.norm(np.array(gps0) - np.array(gps1))
                # Görüntüyü (C, H, W) formatına çevir
                if camera_img0.ndim == 3:
                    if camera_img0.shape[0] == 3:
                        pass
                
                if isinstance(camera_img0, np.ndarray):
                    if camera_img0.shape[0] == 4:
                        camera_img0 = camera_img0[:3, :, :]
                    camera_tensor = torch.from_numpy(camera_img0).float()
                else:
                    if camera_img0.shape[0] == 4:
                        camera_img0 = camera_img0[:3, :, :]
                    camera_tensor = camera_img0.float()
                if camera_tensor.ndim == 3:
                    camera_tensor = camera_tensor.unsqueeze(0)
                elif camera_tensor.ndim == 4:
                    pass
                elif camera_tensor.ndim == 5:
                    if camera_tensor.shape[1] == 1 and camera_tensor.shape[2] == 3:
                        camera_tensor = camera_tensor.squeeze(1)
                    else:
                        raise ValueError(f"Kamera tensörü beklenmeyen boyutta: {camera_tensor.shape}")
                else:
                    raise ValueError(f"Kamera tensörü beklenmeyen boyutta: {camera_tensor.shape}")
                gps0 = torch.tensor(gps0, dtype=torch.float32).unsqueeze(0)
                velocity = torch.tensor(imu0, dtype=torch.float32).unsqueeze(0)
                target = torch.tensor(gps1, dtype=torch.float32).unsqueeze(0)
                collision_tensor = torch.tensor([[float(collision)]], dtype=torch.float32, device=camera_tensor.device)
                distance_tensor = torch.tensor([[distance_auv0_auv1]], dtype=torch.float32, device=camera_tensor.device)
                # --- OBSERVATION'A DISTANCE VE GPS0, GPS1 EKLE ---
                obs = (camera_tensor, collision_tensor, distance_tensor, rotation_flat, gps0, target)
                # gps0 = gps0.squeeze().cpu().numpy()
                pos0_history.append(gps0)
                pos1_history.append(gps1)
                gps0_4ago = pos0_history[0] if len(pos0_history) == 5 else None  # 4 saniye önceki konum
                gps1_4ago = pos1_history[0] if len(pos1_history) == 5 else None  # 4 saniye önceki hedef konum
                # --- REWARD HESAPLAMA ---
                reward = follow_reward(gps0.squeeze().cpu().numpy(), gps1, collision, pos0_4ago=gps0_4ago,pos1_4ago=gps1_4ago)
                episode_reward += reward
                print(f"Episode: {episode}, Step: {step}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}, Distance: {distance_auv0_auv1:.2f}")
                # --- TensorBoard logları ---
                writer.add_scalar("Reward/step", reward, step + episode * max_steps)
                writer.add_scalar("Reward/episode_total", episode_reward, step + episode * max_steps)
                writer.add_scalar("Collision", float(collision), step + episode * max_steps)
                writer.add_scalar("Step", step, step + episode * max_steps)
                writer.add_scalar("Distance/auv0_auv1", distance_auv0_auv1, step + episode * max_steps)
                writer.add_scalar("Action/mean", np.mean(action), step + episode * max_steps)
                writer.add_scalar("Action/std", np.std(action), step + episode * max_steps)
                writer.add_histogram("Action/hist", action, step + episode * max_steps)
                # Eğer agent.train() fonksiyonu loss döndürüyorsa, loss'u da logla:
                # örn: loss = agent.train(); writer.add_scalar("Loss/critic", loss, step + episode * max_steps)
                # --- AGENT EYLEM ---
                # --- NEXT STATE ---
                next_obs = obs
                done = False
                step += 1
                if step >= 200 :
                    done = True
                # --- REPLAY BUFFER ---
                if obs_prev is not None:
                    agent.remember(obs_prev, prev_action, prev_reward, obs, done)
                prev_action = action if 'action' in locals() else np.zeros(agent.action_dim)
                prev_reward = reward
                obs_prev = obs
                gps_prev2 = gps0.squeeze().cpu().numpy()  # iki adım önceki gps güncelleniyor
                # --- AGENT ACTION ---
                with torch.no_grad():
                    action = agent.act(obs)
                env.act("auv0", action * 5)
                # --- AGENT TRAIN ---
                agent.train()
                # --- FEATURE MAP VIZ ---
                device = agent.actor.cnn.fc.weight.device
                # with torch.no_grad():
                #     feature_maps = agent.actor.cnn.conv(camera_tensor.to(device))
                # for i in range(min(8, feature_maps.shape[1])):
                #     feature_axes[i].cla()
                #     feature_axes[i].imshow(feature_maps[0, i].cpu().numpy(), cmap='viridis')
                #     feature_axes[i].set_title(f'F{i+1}')
                #     feature_axes[i].axis('off')
                # feature_fig.canvas.draw()
                # feature_fig.canvas.flush_events()
                if done:
                    print(f"Episode {episode} | Reward: {episode_reward}")
                    # --- TensorBoard episode summary ---
                    writer.add_scalar("Reward/episode", episode_reward, episode)
                    # --- MODELİ KAYDET ---
                    checkpoint = {
                        'actor': agent.actor.state_dict(),
                        'actor_target': agent.actor_target.state_dict(),
                        'critic1': agent.critic1.state_dict(),
                        'critic2': agent.critic2.state_dict(),
                        'critic1_target': agent.critic1_target.state_dict(),
                        'critic2_target': agent.critic2_target.state_dict()
                    }
                    torch.save(checkpoint, model_path)
                    print("Model checkpoint kaydedildi!")
                    episode += 1
                    episode_reward = 0
                    step = 0
                    obs_prev = None
                    env.reset()
            
    writer.close()  # <-- Writer'ı kapat

if __name__ == "__main__":
    main()