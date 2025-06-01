import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from torch.utils.tensorboard import SummaryWriter

# --- CNN Encoder for camera image ---
class CameraCNN(nn.Module):
    def __init__(self, img_shape, out_dim):
        super().__init__()
        c, h, w = img_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            conv_out_size = int(np.prod(conv_out.shape[1:]))
        self.fc = nn.Linear(conv_out_size, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=1, rotation_dim=9):
        super().__init__()
        self.cnn = CameraCNN(img_shape, 128)
        input_dim = 128 + collision_dim + distance_dim + rotation_dim + gps_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        self.tanh = nn.Tanh()
        self.gps_dim = gps_dim
        self.distance_dim = distance_dim
        self.rotation_dim = rotation_dim
        self.collision_dim = collision_dim

    def forward(self, camera, collision, distance, rotation, gps0, gps1):
        cam_emb = self.cnn(camera)
        # Ensure correct shapes
        if collision.ndim == 1:
            collision = collision.unsqueeze(1)
        elif collision.ndim == 0:
            collision = collision.view(1, 1)
        if distance.ndim == 1:
            distance = distance.unsqueeze(1)
        elif distance.ndim == 0:
            distance = distance.view(1, 1)
        if rotation.ndim == 1:
            rotation = rotation.unsqueeze(0)
        if gps0.ndim == 1:
            gps0 = gps0.unsqueeze(0)
        if gps1.ndim == 1:
            gps1 = gps1.unsqueeze(0)
        x = torch.cat([cam_emb, collision, distance, rotation, gps0, gps1], dim=1)
        x = self.fc(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, camera, collision, distance, rotation, gps0, gps1):
        mean, std = self.forward(camera, collision, distance, rotation, gps0, gps1)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t
        # log_prob: see SAC paper appendix C
        log_prob = normal.log_prob(x_t)
        log_prob = log_prob - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=1, rotation_dim=9):
        super().__init__()
        self.cnn = CameraCNN(img_shape, 128)
        input_dim = 128 + collision_dim + distance_dim + rotation_dim + gps_dim * 2 + action_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.gps_dim = gps_dim
        self.distance_dim = distance_dim
        self.rotation_dim = rotation_dim
        self.collision_dim = collision_dim

    def forward(self, camera, collision, distance, rotation, gps0, gps1, action):
        cam_emb = self.cnn(camera)
        if collision.ndim == 1:
            collision = collision.unsqueeze(1)
        elif collision.ndim == 0:
            collision = collision.view(1, 1)
        if distance.ndim == 1:
            distance = distance.unsqueeze(1)
        elif distance.ndim == 0:
            distance = distance.view(1, 1)
        if rotation.ndim == 1:
            rotation = rotation.unsqueeze(0)
        if gps0.ndim == 1:
            gps0 = gps0.unsqueeze(0)
        if gps1.ndim == 1:
            gps1 = gps1.unsqueeze(0)
        x = torch.cat([cam_emb, collision, distance, rotation, gps0, gps1, action], dim=1)
        return self.fc(x)

# --- Replay Buffer ---
Transition = namedtuple('Transition', (
    'camera', 'collision', 'distance', 'rotation', 'gps0', 'gps1', 'action', 'reward',
    'next_camera', 'next_collision', 'next_distance', 'next_rotation', 'next_gps0', 'next_gps1', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

# --- SAC Agent ---
class SACVisionAgent:
    def __init__(self, img_shape, collision_dim, distance_dim, rotation_dim, action_dim, device, writer=None):
        self.device = device
        self.actor = Actor(img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=distance_dim, rotation_dim=rotation_dim).to(device)
        self.critic1 = Critic(img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=distance_dim, rotation_dim=rotation_dim).to(device)
        self.critic2 = Critic(img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=distance_dim, rotation_dim=rotation_dim).to(device)
        self.target_critic1 = Critic(img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=distance_dim, rotation_dim=rotation_dim).to(device)
        self.target_critic2 = Critic(img_shape, collision_dim, action_dim, gps_dim=3, distance_dim=distance_dim, rotation_dim=rotation_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.replay_buffer = ReplayBuffer()
        self.action_dim = action_dim
        self.rotation_dim = rotation_dim
        self.distance_dim = distance_dim
        self.gps_dim = 3
        self.writer = writer
        self._train_step = 0

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

    def act(self, obs, deterministic=False):
        camera, collision, distance, rotation, gps0, gps1 = obs
        camera = camera.to(self.device) if isinstance(camera, torch.Tensor) else torch.FloatTensor(camera).unsqueeze(0).to(self.device)
        collision = collision.to(self.device) if isinstance(collision, torch.Tensor) else torch.FloatTensor(collision).to(self.device)
        distance = distance.to(self.device) if isinstance(distance, torch.Tensor) else torch.FloatTensor(distance).to(self.device)
        rotation = torch.FloatTensor(rotation).view(1, -1).to(self.device)
        gps0 = gps0.to(self.device) if isinstance(gps0, torch.Tensor) else torch.FloatTensor(gps0).to(self.device)
        gps1 = gps1.to(self.device) if isinstance(gps1, torch.Tensor) else torch.FloatTensor(gps1).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(camera, collision, distance, rotation, gps0, gps1)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.actor.sample(camera, collision, distance, rotation, gps0, gps1)
        return action.cpu().numpy().flatten()

    def remember(self, state, action, reward, next_state, done):
        camera, collision, distance, rotation, gps0, gps1 = state
        next_camera, next_collision, next_distance, next_rotation, next_gps0, next_gps1 = next_state
        # Tüm numpy array ve tensorları cpu numpy array'e çevir
        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.array(x)
        camera = to_np(camera)
        collision = to_np(collision)
        distance = to_np(distance)
        rotation = to_np(rotation)
        gps0 = to_np(gps0)
        gps1 = to_np(gps1)
        next_camera = to_np(next_camera)
        next_collision = to_np(next_collision)
        next_distance = to_np(next_distance)
        next_rotation = to_np(next_rotation)
        next_gps0 = to_np(next_gps0)
        next_gps1 = to_np(next_gps1)
        action = to_np(action)
        self.replay_buffer.push(camera, collision, distance, rotation, gps0, gps1, action, reward,
                                next_camera, next_collision, next_distance, next_rotation, next_gps0, next_gps1, done)

    def train(self, batch_size=64, gamma=0.99, tau=0.005, alpha=0.2):
        if len(self.replay_buffer) < batch_size:
            return None
        batch = self.replay_buffer.sample(batch_size)
        device = self.device

        def to_tensor(x, shape=None):
            if isinstance(x, torch.Tensor):
                t = x
            elif isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
            else:
                t = torch.tensor(x, dtype=torch.float32)
            if shape is not None:
                t = t.view(shape)
            return t.float()

        camera = torch.stack([to_tensor(x).squeeze(0) for x in batch.camera]).to(device)
        collision = torch.stack([to_tensor(x).squeeze(0) for x in batch.collision]).to(device)
        distance = torch.stack([to_tensor(x).squeeze(0) for x in batch.distance]).to(device)
        rotation = torch.stack([to_tensor(x).view(-1) for x in batch.rotation]).to(device)
        gps0 = torch.stack([to_tensor(x).squeeze(0) for x in batch.gps0]).to(device)
        gps1 = torch.stack([to_tensor(x).squeeze(0) for x in batch.gps1]).to(device)
        action = torch.FloatTensor(np.array(batch.action)).to(device)
        reward = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        next_camera = torch.stack([to_tensor(x).squeeze(0) for x in batch.next_camera]).to(device)
        next_collision = torch.stack([to_tensor(x).squeeze(0) for x in batch.next_collision]).to(device)
        next_distance = torch.stack([to_tensor(x).squeeze(0) for x in batch.next_distance]).to(device)
        next_rotation = torch.stack([to_tensor(x).view(-1) for x in batch.next_rotation]).to(device)
        next_gps0 = torch.stack([to_tensor(x).squeeze(0) for x in batch.next_gps0]).to(device)
        next_gps1 = torch.stack([to_tensor(x).squeeze(0) for x in batch.next_gps1]).to(device)
        done = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_camera, next_collision, next_distance, next_rotation, next_gps0, next_gps1)
            q1_next = self.target_critic1(next_camera, next_collision, next_distance, next_rotation, next_gps0, next_gps1, next_action)
            q2_next = self.target_critic2(next_camera, next_collision, next_distance, next_rotation, next_gps0, next_gps1, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * gamma * (q_next - alpha * next_log_pi)

        q1 = self.critic1(camera, collision, distance, rotation, gps0, gps1, action)
        q2 = self.critic2(camera, collision, distance, rotation, gps0, gps1, action)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # --- Actor update ---
        new_action, log_pi, _ = self.actor.sample(camera, collision, distance, rotation, gps0, gps1)
        q1_new = self.critic1(camera, collision, distance, rotation, gps0, gps1, new_action)
        q2_new = self.critic2(camera, collision, distance, rotation, gps0, gps1, new_action)
        actor_loss = (alpha * log_pi - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Target update ---
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # --- TensorBoard loglama ---
        if self.writer is not None:
            self.writer.add_scalar("Loss/critic1", critic1_loss.item(), self._train_step)
            self.writer.add_scalar("Loss/critic2", critic2_loss.item(), self._train_step)
            self.writer.add_scalar("Loss/actor", actor_loss.item(), self._train_step)
        self._train_step += 1

        return {'critic1_loss': critic1_loss.item(), 'critic2_loss': critic2_loss.item(), 'actor_loss': actor_loss.item()}


