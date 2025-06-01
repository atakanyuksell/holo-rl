import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, img_shape, gps_size, action_size=8, target_size=3, rotation_size=9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *img_shape)
            cnn_out = self.cnn(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + gps_size + 1 + target_size + rotation_size, 128), nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh()
        )
        self.rotation_size = rotation_size

    def forward(self, camera_img, gps, collision, target, rotation):
        c = self.cnn(camera_img)
        if rotation.ndim == 1:
            rotation = rotation.unsqueeze(0)
        x = torch.cat([c, gps, collision, target, rotation], dim=1)
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, img_shape, gps_size, action_size=8, target_size=3, rotation_size=9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *img_shape)
            cnn_out = self.cnn(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + gps_size + 1 + action_size + target_size + rotation_size, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.rotation_size = rotation_size

    def forward(self, camera_img, gps, collision, action, target, rotation):
        c = self.cnn(camera_img)
        if rotation.ndim == 1:
            rotation = rotation.unsqueeze(0)
        x = torch.cat([c, gps, collision, action, target, rotation], dim=1)
        return self.fc(x)

class DDPGAgent:
    def __init__(self, img_shape, gps_size, action_size=8, device="cpu", target_size=3, rotation_size=9):
        self.action_size = action_size
        self.memory = deque(maxlen=250000)  # Replay buffer büyütüldü
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        self.device = device
        self.rotation_size = rotation_size

        self.actor = Actor(img_shape, gps_size, action_size, target_size=target_size, rotation_size=rotation_size).to(device)
        self.actor_target = Actor(img_shape, gps_size, action_size, target_size=target_size, rotation_size=rotation_size).to(device)
        self.critic = Critic(img_shape, gps_size, action_size, target_size=target_size, rotation_size=rotation_size).to(device)
        self.critic_target = Critic(img_shape, gps_size, action_size, target_size=target_size, rotation_size=rotation_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.noise_scale = 0.5

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, add_noise=True):
        camera_img, gps, collision, target, rotation = state
        camera_img_tensor = torch.FloatTensor(camera_img).unsqueeze(0).to(self.device)
        gps = torch.FloatTensor(gps).unsqueeze(0).to(self.device)
        collision = torch.FloatTensor([collision]).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        rotation = torch.FloatTensor(rotation).view(1, -1).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(camera_img_tensor, gps, collision, target, rotation).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            action += self.noise_scale * np.random.randn(self.action_size)
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        camera_imgs = torch.stack([torch.FloatTensor(s[0]) for s in states]).to(self.device)
        gpss = torch.FloatTensor(np.array([s[1] for s in states])).to(self.device)
        collisions = torch.FloatTensor([[s[2]] for s in states]).to(self.device)
        targets = torch.FloatTensor(np.array([s[3] for s in states])).to(self.device)
        rotations = torch.stack([torch.FloatTensor(s[4]).view(-1) for s in states]).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_camera_imgs = torch.stack([torch.FloatTensor(s[0]) for s in next_states]).to(self.device)
        next_gpss = torch.FloatTensor(np.array([s[1] for s in next_states])).to(self.device)
        next_collisions = torch.FloatTensor([[s[2]] for s in next_states]).to(self.device)
        next_targets = torch.FloatTensor(np.array([s[3] for s in next_states])).to(self.device)
        next_rotations = torch.stack([torch.FloatTensor(s[4]).view(-1) for s in next_states]).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_camera_imgs, next_gpss, next_collisions, next_targets, next_rotations)
            q_next = self.critic_target(next_camera_imgs, next_gpss, next_collisions, next_actions, next_targets, next_rotations)
            q_target = rewards + (1 - dones) * self.gamma * q_next
        q_val = self.critic(camera_imgs, gpss, collisions, actions, targets, rotations)
        critic_loss = self.loss_fn(q_val, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_actions = self.actor(camera_imgs, gpss, collisions, targets, rotations)
        actor_loss = -self.critic(camera_imgs, gpss, collisions, actor_actions, targets, rotations).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Target network update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
