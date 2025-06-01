import torch
import torch.nn as nn
import torch.nn.functional as F

class BCActor(nn.Module):
    def __init__(self, img_shape, gps_size, action_size, target_size=3):
        super().__init__()
        c, h, w = img_shape
        # Basit bir CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out = self.cnn(dummy).view(1, -1).shape[1]
        # CNN + GPS + Target + Collision
        self.fc = nn.Sequential(
            nn.Linear(cnn_out + gps_size + target_size + 1, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, camera_img, gps, collision, target):
        x1 = self.cnn(camera_img)
        x = torch.cat([x1, gps, collision, target], dim=1)
        return self.fc(x)

class BCVisionAgent:
    def __init__(self, img_shape, gps_size, action_size, target_size=3, device='cpu'):
        self.device = device
        self.actor = BCActor(img_shape, gps_size, action_size, target_size).to(device)
        self.action_size = action_size

    def act(self, obs):
        camera_img, gps, collision, target = obs
        camera_img = torch.FloatTensor(camera_img).unsqueeze(0).to(self.device)
        gps = torch.FloatTensor(gps).unsqueeze(0).to(self.device)
        collision = torch.FloatTensor([collision]).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(target).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(camera_img, gps, collision, target)
        return action.cpu().numpy().flatten()
