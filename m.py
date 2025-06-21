import holoocean
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time
import os
from datetime import datetime
from pynput import keyboard
from helpers.key_control import parse_keys, start_listener, pressed_keys
from helpers.observation_utils import process_observation
from helpers.resnet18_obstacle_cnn import ResNet18ObstacleCNN
from helpers.collision import detect_collision_by_diff

force = 15

# ResNet18 CNN Model için setup - 512x512x3 için optimize edilmiş
device = "cuda" if torch.cuda.is_available() else "cpu"
obstacle_cnn = ResNet18ObstacleCNN().to(device)
print(f"🤖 ResNet18 Obstacle CNN loaded on {device}")
print(f"📐 Input format: 512x512x3 RGB kamera görüntüleri")

# Progressive Learning için global değişkenler
training_data = []
training_mode = False
last_label_time = 0
frame_buffer = []  # Son 10 frame için buffer

# Gelişmiş Eğitim Konfigürasyonu
class TrainingConfig:
    def __init__(self, data_count):
        if data_count < 200:
            self.batch_size = 8
            self.learning_rate = 1e-4
            self.save_interval = 25
            self.phase = "Bootstrap"
            self.weight_decay = 1e-5
        elif data_count < 500:
            self.batch_size = 16
            self.learning_rate = 5e-5
            self.save_interval = 50
            self.phase = "Acceleration"
            self.weight_decay = 1e-4
        else:
            self.batch_size = 32
            self.learning_rate = 1e-5
            self.save_interval = 100
            self.phase = "Fine-tuning"
            self.weight_decay = 1e-3

# Eğitim durumu tracking
training_stats = {
    'total_samples': 0,
    'obstacle_samples': 0,
    'no_obstacle_samples': 0,
    'recent_losses': [],
    'best_loss': float('inf'),
    'epochs_without_improvement': 0
}

# Dinamik optimizer ve criterion (data miktarına göre güncellenecek)
obstacle_optimizer = torch.optim.Adam(obstacle_cnn.parameters(), lr=1e-4, weight_decay=1e-5)
obstacle_criterion = nn.BCELoss()
obstacle_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    obstacle_optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# Görüntü gösterimi için global değişkenler
current_frame = None
current_prediction = None
display_active = True

def get_dataset_balance():
    """Mevcut dataset dengesini hesapla"""
    # Memory'deki denge
    memory_obstacle = sum(1 for s in training_data if s['label'] == 1.0)
    memory_no_obstacle = sum(1 for s in training_data if s['label'] == 0.0)
    
    # Disk'teki denge
    disk_obstacle = len([f for f in os.listdir('obstacle_data/with_obstacle') if f.endswith('.npy')]) if os.path.exists('obstacle_data/with_obstacle') else 0
    disk_no_obstacle = len([f for f in os.listdir('obstacle_data/no_obstacle') if f.endswith('.npy')]) if os.path.exists('obstacle_data/no_obstacle') else 0
    
    return {
        'memory': {'obstacle': memory_obstacle, 'no_obstacle': memory_no_obstacle},
        'disk': {'obstacle': disk_obstacle, 'no_obstacle': disk_no_obstacle}
    }

def check_training_readiness():
    """Eğitim için hazır olup olmadığını kontrol et"""
    balance = get_dataset_balance()
    
    # Her iki sınıftan en az 4 sample olmalı
    min_samples = 4
    memory_ready = (balance['memory']['obstacle'] >= min_samples and 
                   balance['memory']['no_obstacle'] >= min_samples)
    
    if not memory_ready:
        print(f"⚠️ Eğitim için yetersiz balanced veri!")
        print(f"   Memory: Engel={balance['memory']['obstacle']}, Engelsiz={balance['memory']['no_obstacle']}")
        print(f"   Minimum: Her sınıftan {min_samples} sample gerekli")
        return False
    
    return True

def save_training_sample(image, label, collision_info=None):
    """Eğitim verisi kaydet - Balanced klasör yapısında"""
    global training_data
    timestamp = time.time()
    sample = {
        'image': image.copy(),
        'label': label,
        'timestamp': timestamp,
        'collision': collision_info
    }
    training_data.append(sample)
    
    # Balanced klasör yapısı oluştur
    if label == 1.0:  # Engel var
        folder_path = 'obstacle_data/with_obstacle'
        label_name = "obstacle"
    else:  # Engel yok
        folder_path = 'obstacle_data/no_obstacle'
        label_name = "no_obstacle"
    
    os.makedirs(folder_path, exist_ok=True)
    
    # Dosya adı: klasör içinde sıralı numara
    existing_files = len([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    filename = f'{folder_path}/sample_{existing_files+1:06d}_{label_name}_{timestamp:.0f}.npy'
    
    np.save(filename, sample)
    print(f"💾 {label_name.upper()} verisi kaydedildi: {filename}")
    
    # Denge durumunu kontrol et ve raporla
    obstacle_count = len([f for f in os.listdir('obstacle_data/with_obstacle') if f.endswith('.npy')]) if os.path.exists('obstacle_data/with_obstacle') else 0
    no_obstacle_count = len([f for f in os.listdir('obstacle_data/no_obstacle') if f.endswith('.npy')]) if os.path.exists('obstacle_data/no_obstacle') else 0
    
    print(f"📊 Dataset Dengesi: Engel={obstacle_count}, Engelsiz={no_obstacle_count}, Toplam={obstacle_count + no_obstacle_count}")

def augment_image(image):
    """Basit data augmentation - noise ve brightness"""
    if isinstance(image, np.ndarray):
        augmented = image.copy().astype(np.float32)
        
        # Random noise (sadece %20 şansla)
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255)
        
        # Random brightness (sadece %30 şansla)
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness_factor, 0, 255)
        
        return augmented.astype(np.uint8)
    return image

def train_obstacle_step(image, label, use_augmentation=False):
    """ResNet18 ile tek adım eğitim - 512x512x3 optimized"""
    global obstacle_cnn, obstacle_optimizer, obstacle_criterion
    
    # Data augmentation (isteğe bağlı)
    if use_augmentation and np.random.random() < 0.3:
        image = augment_image(image)
    
    # Image preprocessing
    if isinstance(image, np.ndarray):
        if image.dtype == np.uint8:
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        else:
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    else:
        image_tensor = image.unsqueeze(0).to(device)
    
    label_tensor = torch.FloatTensor([label]).to(device)
    
    # Forward pass - ResNet18 direkt scalar döndürür
    obstacle_prob = obstacle_cnn(image_tensor)
    
    # Tensor boyutlarını uyumlu hale getir
    if obstacle_prob.dim() == 0:
        obstacle_prob = obstacle_prob.unsqueeze(0)
    elif obstacle_prob.dim() > 1:
        obstacle_prob = obstacle_prob.squeeze()
    
    # Loss hesaplama
    loss = obstacle_criterion(obstacle_prob, label_tensor)
    
    # Backward pass
    obstacle_optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (stability için)
    torch.nn.utils.clip_grad_norm_(obstacle_cnn.parameters(), max_norm=1.0)
    
    obstacle_optimizer.step()
    
    return loss.item()

def auto_label_from_collision(collision):
    """Çarpışma verilerinden otomatik etiketleme"""
    global frame_buffer
    
    if collision and len(frame_buffer) > 0:
        # Çarpışmadan önceki 3-5 frame'i engelli olarak etiketle
        frames_to_label = min(5, len(frame_buffer))
        for i in range(max(0, len(frame_buffer)-frames_to_label), len(frame_buffer)):
            if i < len(frame_buffer):
                loss = train_obstacle_step(frame_buffer[i], 1.0)
                save_training_sample(frame_buffer[i], 1.0, collision_info=True)
                print(f"🤖 OTO-ETİKET: Engel - Loss: {loss:.4f}")

def update_training_config():
    """Veri miktarına göre eğitim konfigürasyonunu güncelle"""
    global obstacle_optimizer, obstacle_scheduler, training_stats
    
    config = TrainingConfig(len(training_data))
    
    # Optimizer'ı güncelle
    for param_group in obstacle_optimizer.param_groups:
        param_group['lr'] = config.learning_rate
        param_group['weight_decay'] = config.weight_decay
    
    # İstatistikleri güncelle
    training_stats['total_samples'] = len(training_data)
    training_stats['obstacle_samples'] = sum(1 for s in training_data if s['label'] == 1.0)
    training_stats['no_obstacle_samples'] = training_stats['total_samples'] - training_stats['obstacle_samples']
    
    print(f"🎯 Eğitim Fazı: {config.phase}")
    print(f"📊 Batch Size: {config.batch_size}, LR: {config.learning_rate:.2e}")
    print(f"📈 Toplam Sample: {training_stats['total_samples']}")
    print(f"   ⚠️  Engel: {training_stats['obstacle_samples']}")
    print(f"   ✅ Engelsiz: {training_stats['no_obstacle_samples']}")
    
    return config

def advanced_batch_train_obstacles():
    """Geliştirilmiş balanced batch eğitim sistemi"""
    global training_data, obstacle_cnn, obstacle_optimizer, obstacle_criterion, obstacle_scheduler
    
    # Önce eğitim hazırlığını kontrol et
    if not check_training_readiness():
        return None
    
    config = TrainingConfig(len(training_data))
    
    if len(training_data) < config.batch_size:
        print(f"❌ Yetersiz veri: {len(training_data)}, minimum {config.batch_size} gerekli")
        return None
    
    # Balanced sampling için verileri ayır
    obstacle_samples = [s for s in training_data if s['label'] == 1.0]
    no_obstacle_samples = [s for s in training_data if s['label'] == 0.0]
    
    if len(obstacle_samples) == 0 or len(no_obstacle_samples) == 0:
        print("⚠️ Dengesiz dataset! Her iki sınıftan da veri gerekli.")
        return None
    
    # Balanced batch oluştur - tam eşitlik sağla
    half_batch = config.batch_size // 2
    
    # Sample selection with replacement if needed
    obstacle_indices = np.random.choice(len(obstacle_samples), 
                                      half_batch, 
                                      replace=len(obstacle_samples) < half_batch)
    no_obstacle_indices = np.random.choice(len(no_obstacle_samples), 
                                         half_batch, 
                                         replace=len(no_obstacle_samples) < half_batch)
    
    # Batch verisini topla
    batch_samples = []
    batch_samples.extend([obstacle_samples[i] for i in obstacle_indices])
    batch_samples.extend([no_obstacle_samples[i] for i in no_obstacle_indices])
    
    # Shuffle the batch for better training
    np.random.shuffle(batch_samples)
    
    # Batch training
    total_loss = 0
    correct_predictions = 0
    
    for sample in batch_samples:
        # Data augmentation %30 şansla
        use_aug = np.random.random() < 0.3
        loss = train_obstacle_step(sample['image'], sample['label'], use_augmentation=use_aug)
        total_loss += loss
        
        # Accuracy hesapla
        with torch.no_grad():
            if isinstance(sample['image'], np.ndarray):
                if sample['image'].dtype == np.uint8:
                    img_tensor = torch.FloatTensor(sample['image']).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                else:
                    img_tensor = torch.FloatTensor(sample['image']).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                img_tensor = sample['image'].unsqueeze(0).to(device)
            
            # ResNet18 direkt scalar döndürür
            obstacle_prob = obstacle_cnn(img_tensor)
            obstacle_prob_val = obstacle_prob.item() if obstacle_prob.dim() == 0 else obstacle_prob[0].item()
            
            predicted_class = 1.0 if obstacle_prob_val > 0.5 else 0.0
            if predicted_class == sample['label']:
                correct_predictions += 1
    
    avg_loss = total_loss / len(batch_samples)
    accuracy = correct_predictions / len(batch_samples)
    
    # Scheduler step
    obstacle_scheduler.step(avg_loss)
    
    # İstatistikleri güncelle
    training_stats['recent_losses'].append(avg_loss)
    if len(training_stats['recent_losses']) > 50:
        training_stats['recent_losses'].pop(0)
    
    if avg_loss < training_stats['best_loss']:
        training_stats['best_loss'] = avg_loss
        training_stats['epochs_without_improvement'] = 0
    else:
        training_stats['epochs_without_improvement'] += 1
    
    print(f"📊 ADVANCED BALANCED BATCH: {len(batch_samples)} samples")
    print(f"   📉 Loss: {avg_loss:.4f} (Best: {training_stats['best_loss']:.4f})")
    print(f"   🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"   ⚖️  Perfect Balance: {half_batch} engel, {half_batch} engelsiz")
    
    # Dataset balance durumunu göster
    balance = get_dataset_balance()
    print(f"   💾 Disk Balance: {balance['disk']['obstacle']} engel, {balance['disk']['no_obstacle']} engelsiz")
    
    # Early stopping warning
    if training_stats['epochs_without_improvement'] > 20:
        print("⚠️ 20 epoch'tan fazla gelişim yok! Model kaydetmeyi düşünün.")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'batch_size': len(batch_samples),
        'config': config,
        'balance': balance
    }

def smart_auto_save():
    """Akıllı otomatik model kaydetme"""
    config = TrainingConfig(len(training_data))
    
    # Belirli aralıklarla otomatik kaydet
    if len(training_data) % config.save_interval == 0 and len(training_data) > 0:
        save_obstacle_model()
        print(f"🤖 Otomatik kayıt: {len(training_data)} sample'da model kaydedildi")
        return True
    return False

def save_obstacle_model():
    """Model ağırlıklarını kaydet"""
    os.makedirs('obstacle_models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"obstacle_models/obstacle_cnn_{timestamp}_samples{len(training_data)}.pth"
    torch.save(obstacle_cnn.state_dict(), save_path)
    print(f"💾 Model kaydedildi: {save_path}")

def load_latest_obstacle_model():
    """En son kaydedilen modeli yükle"""
    model_dir = 'obstacle_models'
    if not os.path.exists(model_dir):
        return False
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not model_files:
        return False
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    try:
        obstacle_cnn.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Model yüklendi: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return False

def update_display():
    """Kamera görüntüsünü ve CNN tahminini gösteren thread"""
    global current_frame, current_prediction, display_active, training_mode
    
    # OpenCV window oluştur
    cv2.namedWindow('AUV Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AUV Camera Feed', 512, 512)
    
    while display_active:
        if current_frame is not None:
            # Frame'i kopyala (thread safety)
            frame = current_frame.copy()
            
            # CNN tahminini frame üzerine yaz
            if current_prediction is not None:
                obstacle_prob = current_prediction.get('obstacle_probability', 0)
                
                # Ana engel bilgisi
                if obstacle_prob > 0.5:
                    text = f"ENGEL VAR: {obstacle_prob:.3f}"
                    color = (0, 0, 255)  # Kırmızı
                    cv2.putText(frame, "ENGEL ALGILANDI!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    text = f"ENGEL YOK: {obstacle_prob:.3f}"
                    color = (0, 255, 0)  # Yeşil
                
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Eğitim modu göstergesi
            if training_mode:
                cv2.putText(frame, "EGITIM MODU AKTIF", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Samples: {len(training_data)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Dataset balance bilgisini göster
                obstacle_count = sum(1 for s in training_data if s['label'] == 1.0)
                no_obstacle_count = sum(1 for s in training_data if s['label'] == 0.0)
                cv2.putText(frame, f"Balance: {obstacle_count}O/{no_obstacle_count}N", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Training stats göster
                if training_stats['recent_losses']:
                    avg_recent_loss = np.mean(training_stats['recent_losses'][-10:])
                    cv2.putText(frame, f"Recent Loss: {avg_recent_loss:.3f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Phase göster
                config = TrainingConfig(len(training_data))
                cv2.putText(frame, f"Phase: {config.phase}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Kontrol tuşları bilgisini göster
            cv2.putText(frame, "W/S:Ileri/Geri A/D:Sol/Sag I/K:Yukari/Asagi J/L:Don", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "T:Egitim O:Engel N:Engelsiz B:AdvBatch M:Kaydet", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "C:Config R:Reset V:Balance Q:Cikis", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('AUV Camera Feed', frame)
        
        # ESC ile çıkış
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
            
        time.sleep(0.03)  # ~30 FPS
    
    cv2.destroyAllWindows()

# key_control helpers'ı kullanarak listener başlat
start_listener()

# Mevcut modeli yüklemeye ve veriyi restore etmeye çalış
print("🔄 Sistem başlatılıyor...")
try:
    # Balanced klasör yapısından verileri yükle
    obstacle_files = []
    no_obstacle_files = []
    
    if os.path.exists('obstacle_data/with_obstacle'):
        obstacle_files = [f for f in os.listdir('obstacle_data/with_obstacle') if f.endswith('.npy')]
    if os.path.exists('obstacle_data/no_obstacle'):
        no_obstacle_files = [f for f in os.listdir('obstacle_data/no_obstacle') if f.endswith('.npy')]
    
    print(f"📁 Mevcut veri: {len(obstacle_files)} engel, {len(no_obstacle_files)} engelsiz")
    
    # Balanced loading - her iki sınıftan eşit sayıda yükle
    max_load_per_class = 50  # Her sınıftan maksimum 50 sample yükle (hızlı başlatma)
    
    # Engel verilerini yükle
    for i, file in enumerate(sorted(obstacle_files)[:max_load_per_class]):
        try:
            file_path = os.path.join('obstacle_data/with_obstacle', file)
            sample = np.load(file_path, allow_pickle=True).item()
            training_data.append(sample)
        except:
            pass
    
    # Engelsiz verilerini yükle
    for i, file in enumerate(sorted(no_obstacle_files)[:max_load_per_class]):
        try:
            file_path = os.path.join('obstacle_data/no_obstacle', file)
            sample = np.load(file_path, allow_pickle=True).item()
            training_data.append(sample)
        except:
            pass
    
    print(f"✅ {len(training_data)} balanced veri memory'e yüklendi")
    
    # Dataset denge durumunu göster
    loaded_obstacle = sum(1 for s in training_data if s['label'] == 1.0)
    loaded_no_obstacle = sum(1 for s in training_data if s['label'] == 0.0)
    print(f"📊 Yüklenen veri dengesi: Engel={loaded_obstacle}, Engelsiz={loaded_no_obstacle}")
    
    # En son modeli yükle
    if load_latest_obstacle_model():
        print("✅ En son model başarıyla yüklendi")
    
    # İlk config güncellemesi
    if len(training_data) > 0:
        update_training_config()
        
except Exception as e:
    print(f"⚠️ Başlatma sırasında uyarı: {e}")
    print("💡 Yeni başlangıç yapılıyor...")

print("=== HoloOcean Advanced Progressive Obstacle Learning ===")
print("HAREKET:")
print("  W/S: İleri/Geri")
print("  A/D: Sol/Sağ")
print("  I/K: Yukarı/Aşağı") 
print("  J/L: Sola/Sağa Dön")
print("\nGELİŞMİŞ EĞİTİM:")
print("  T: Eğitim modunu aç/kapat")
print("  O: Bu görüntüde ENGEL var (eğitim modunda)")
print("  N: Bu görüntüde ENGEL yok (eğitim modunda)")
print("  B: Advanced balanced batch eğitimi başlat")
print("  M: Modeli kaydet")
print("  C: Eğitim konfigürasyonunu güncelle")
print("  R: Training stats resetle")
print("  V: Dataset balance durumunu göster")
print("\nDİĞER:")
print("  Q: Çıkış")
print("  ESC: Kamera penceresini kapat")
print("\n🚀 ÖZELLİKLER:")
print("  • Dinamik batch size (8→16→32)")
print("  • Adaptive learning rate")
print("  • Data augmentation")
print("  • Balanced sampling")
print("  • Learning rate scheduler")
print("  • Gradient clipping")
print("  • Smart auto-save")
print("  • Real-time training stats")
print("================================================================\n")

# Display thread'i başlat
display_thread = threading.Thread(target=update_display)
display_thread.daemon = True
display_thread.start()

with holoocean.make("OpenWater-HoveringCamera", show_viewport=True) as env:
      
    while True:
        
        if 'q' in pressed_keys:
            print("Çıkış yapılıyor...")
            display_active = False  # Display thread'i durdur
            break
        
        # Eğitim modu kontrolü
        if 't' in pressed_keys:
            training_mode = not training_mode
            print(f"🎓 Eğitim modu: {'AÇIK' if training_mode else 'KAPALI'}")
            time.sleep(0.3)  # Çift basımı önle
        
        # Manuel batch eğitimi
        if 'b' in pressed_keys:
            print("🔄 Advanced batch eğitimi başlıyor...")
            result = advanced_batch_train_obstacles()
            if result:
                update_training_config()
            time.sleep(0.5)  # Çift basımı önle
        
        # Model kaydetme
        if 'm' in pressed_keys:
            save_obstacle_model()
            time.sleep(0.5)  # Çift basımı önle
        
        # Konfigürasyon güncelle
        if 'c' in pressed_keys:
            update_training_config()
            time.sleep(0.5)  # Çift basımı önle
        
        # Stats reset
        if 'r' in pressed_keys:
            training_stats['recent_losses'] = []
            training_stats['best_loss'] = float('inf')
            training_stats['epochs_without_improvement'] = 0
            print("📊 Training stats resetlendi!")
            time.sleep(0.5)  # Çift basımı önle
        
        # Dataset balance durumunu göster
        if 'v' in pressed_keys:
            balance = get_dataset_balance()
            print(f"\n📊 DATASET BALANCE RAPORU:")
            print(f"   Memory: Engel={balance['memory']['obstacle']}, Engelsiz={balance['memory']['no_obstacle']}")
            print(f"   Disk:   Engel={balance['disk']['obstacle']}, Engelsiz={balance['disk']['no_obstacle']}")
            print(f"   Toplam: {balance['disk']['obstacle'] + balance['disk']['no_obstacle']} sample")
            
            # Balance skoru hesapla (ne kadar dengeli?)
            total = balance['disk']['obstacle'] + balance['disk']['no_obstacle']
            if total > 0:
                obstacle_ratio = balance['disk']['obstacle'] / total
                balance_score = 1 - abs(0.5 - obstacle_ratio) * 2  # 0-1 arası, 1 = perfect balance
                print(f"   Balance Score: {balance_score:.3f} (1.0 = perfect balance)")
            time.sleep(0.5)  # Çift basımı önle
        
        # Pressed keys'den command oluştur
        command = parse_keys(pressed_keys, force)
        
        # HoloOcean'a gönder
        env.act("auv0", command)
        state = env.tick()
       
        

        # Kamera görüntüsünü al ve işle
        if 'Camera' in state and 'IMUSensor' in state and 'PoseSensor' in state:
            imu = state['IMUSensor']
            collision = detect_collision_by_diff(imu, threshold=10)
            if collision:
                print("💥 Çarpışma algılandı!")
                
            try:
                # Kamera görüntüsünü işle
                camera_img, _, _ = process_observation(state)
                
                # BGR formatına çevir (OpenCV için)
                camera_bgr = camera_img.permute(1, 2, 0).numpy()  # [H, W, C]
                camera_bgr = (camera_bgr * 255).astype(np.uint8)
                camera_bgr = cv2.cvtColor(camera_bgr, cv2.COLOR_RGB2BGR)
                
                # Frame buffer'ı güncelle (collision auto-labeling için)
                frame_buffer.append(camera_bgr.copy())
                if len(frame_buffer) > 10:
                    frame_buffer.pop(0)
                
                # Global frame'i güncelle
                current_frame = camera_bgr
                
                # CNN ile engel tahmini - ResNet18 predict fonksiyonu kullan
                camera_tensor = camera_img.unsqueeze(0).to(device)  # [1, 3, H, W]
                with torch.no_grad():
                    prediction = obstacle_cnn.predict(camera_tensor)
                    current_prediction = {
                        'obstacle_probability': prediction['obstacle_probability']
                    }
                
                # Engel auto-labeling (çarpışma durumunda)
                auto_label_from_collision(collision)
                
                # Manuel labeling (eğitim modunda)
                if training_mode and current_frame is not None:
                    current_time = time.time()
                    
                    # Manuel labeling (minimum 1 saniye aralık)
                    if current_time - last_label_time > 1.0:
                        if 'o' in pressed_keys:  # Obstacle var
                            loss = train_obstacle_step(current_frame, 1.0, use_augmentation=True)
                            save_training_sample(current_frame, 1.0, collision)
                            smart_auto_save()  # Akıllı otomatik kayıt
                            print(f"🔴 ENGEL ETİKETLENDİ - Loss: {loss:.4f}")
                            last_label_time = current_time
                            
                        elif 'n' in pressed_keys:  # No obstacle  
                            loss = train_obstacle_step(current_frame, 0.0, use_augmentation=True)
                            save_training_sample(current_frame, 0.0, collision)
                            smart_auto_save()  # Akıllı otomatik kayıt
                            print(f"🟢 ENGELSİZ ETİKETLENDİ - Loss: {loss:.4f}")
                            last_label_time = current_time
                
                # Konsola engel durumu yazdır
                if current_prediction['obstacle_probability'] > 0.5:
                    print(f"⚠️  ENGEL ALGILANDI! Olasılık: {current_prediction['obstacle_probability']:.3f}")
                
            except Exception as e:
                print(f"Kamera işleme hatası: {e}")

print("Display thread durduruluyor...")
display_active = False
time.sleep(0.5)  # Thread'in temiz kapanması için bekle
print("Program sonlandırıldı.")