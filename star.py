import holoocean
import numpy as np
import torch
import os
import time
import cv2
import threading
from helpers.key_control import start_listener, parse_keys, pressed_keys
from helpers.observation_utils import pose_to_gps, get_pose_from_auv, get_relative_pose, velocity_to_speed_and_direction, pose_to_rotation, pose_to_euler, process_observation, extract_state_info
from helpers.resnet18_obstacle_cnn import ResNet18ObstacleCNN
from helpers.collision import detect_collision_by_diff
import tkinter as tk

# Global değişkenler
current_camera_frame = None
display_active = True
avoidance_target_gps = None
movement_tolerance = 2.0

#--------------------- ResNet18 CNN Obstacle Detection Setup ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
obstacle_cnn = ResNet18ObstacleCNN().to(device)
print(f"🤖 ResNet18 Obstacle CNN loaded on {device}")

def list_available_models():
    """List trainnoplot.py models in obstacle_models directory"""
    model_dir = 'obstacle_models'
    if not os.path.exists(model_dir):
        print("📁 obstacle_models directory not found")
        return []
    
    # Only look for trainnoplot.py models (resnet18_*)
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('resnet18_')]
    if not model_files:
        print("📁 No trainnoplot.py models found in obstacle_models/")
        print("   (Looking for files starting with 'resnet18_')")
        return []
    
    model_files = sorted(model_files)
    print(f"📁 Found {len(model_files)} trainnoplot.py model files:")
    for model in model_files:
        print(f"    - {model}")
    
    return model_files

def load_latest_obstacle_model():
    """Load latest ResNet18 obstacle detection model from trainnoplot.py"""
    model_dir = 'obstacle_models'
    if not os.path.exists(model_dir):
        print("⚠️ obstacle_models klasörü bulunamadı!")
        return False
    
    # Only look for trainnoplot.py models (resnet18_*)
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('resnet18_')]
    if not model_files:
        print("⚠️ trainnoplot.py model dosyası bulunamadı!")
        print("   (resnet18_ ile başlayan .pth dosyaları aranıyor)")
        return False
    
    # Get the latest trainnoplot.py model
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    print(f"🎯 trainnoplot.py modelini yüklüyor: {latest_model}")
    
    try:
        # Load trainnoplot.py format: dictionary with 'model_state_dict' key
        model_data = torch.load(model_path, map_location=device)
        
        if isinstance(model_data, dict) and 'model_state_dict' in model_data:
            # trainnoplot.py format
            obstacle_cnn.load_state_dict(model_data['model_state_dict'])
            print(f"✅ ResNet18 model yüklendi: {latest_model}")
            
            # Show training info if available
            if 'epoch' in model_data:
                print(f"📊 Eğitim Epoch: {model_data['epoch']}")
            if 'config' in model_data and isinstance(model_data['config'], dict):
                config = model_data['config']
                if 'batch_size' in config:
                    print(f"⚙️ Batch Size: {config['batch_size']}")
                if 'learning_rate' in config:
                    print(f"⚙️ Learning Rate: {config['learning_rate']}")
        else:
            print("❌ Model format geçersiz! trainnoplot.py formatı bekleniyor.")
            return False
        
        obstacle_cnn.eval()
        return True
        
    except Exception as e:
        print(f"❌ Model yüklenemedi: {e}")
        print(f"🔍 Model file: {latest_model}")
        return False

def predict_obstacle(camera_img):
    """ResNet18 ikili engel tahmini - sadece VAR/YOK"""
    try:
        # m.py ile tamamen aynı preprocessing: camera_img zaten [C,H,W] tensor ve 0-1 aralığında
        # process_observation() fonksiyonu tarafından önceden işlenmiş
        if isinstance(camera_img, torch.Tensor):
            # Already a tensor in [C,H,W] format and 0-1 range from process_observation
            camera_tensor = camera_img.unsqueeze(0).to(device)  # [1, C, H, W]
        else:
            # Fallback for numpy arrays (shouldn't happen with process_observation)
            if isinstance(camera_img, np.ndarray):
                if camera_img.dtype == np.uint8:
                    camera_tensor = torch.FloatTensor(camera_img).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
                else:
                    camera_tensor = torch.FloatTensor(camera_img).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                raise ValueError(f"Unsupported camera_img type: {type(camera_img)}")
        
        with torch.no_grad():
            # ResNet18'in predict methodunu kullan (m.py ile aynı)
            prediction = obstacle_cnn.predict(camera_tensor)
            
            # m.py formatında basit ikili çıktı
            obstacle_prob = prediction['obstacle_probability']
            obstacle_detected = prediction['obstacle_detected']  # True/False
            
            # Debug bilgisi
            print(f"🔍 star.py CNN: Prob={obstacle_prob:.3f}, Detected={obstacle_detected}")
            
        return {
            'obstacle_probability': obstacle_prob,
            'obstacle_detected': obstacle_detected
        }
    except Exception as e:
        print(f"⚠️ ResNet18 CNN prediction error: {e}")
        return {
            'obstacle_probability': 0.0,
            'obstacle_detected': False
        }

def create_simple_avoidance_action(obstacle_detected, current_gps, current_yaw=None):
    """Basit ikili kaçınma - sadece ENGEL VAR/YOK"""
    if not obstacle_detected:
        return None
    
    print(f"🚨 ENGEL ALGILANDI! Kaçınma hareketi başlatılıyor...")
    
    # Gelişmiş kaçınma stratejisi - yukarı çık ve ileri git
    forward_distance = 6.0  # İleri gitme mesafesi
    upward_distance = 4.0   # Yukarı çıkma mesafesi
    
    # Eğer yaw açısı verilmişse, baktığı yöne doğru ileri git
    if current_yaw is not None:
        # Yaw açısından ileri yönü hesapla (radyan cinsinden)
        yaw_rad = np.radians(current_yaw)
        forward_x = forward_distance * np.cos(yaw_rad)
        forward_y = forward_distance * np.sin(yaw_rad)
        
        avoidance_gps = [
            current_gps[0] + forward_x,  # Baktığı yöne X ekseni
            current_gps[1] + forward_y,  # Baktığı yöne Y ekseni
            current_gps[2] + upward_distance   # Yukarı çık
        ]
        print(f"🎯 Kaçınma hedefi (yaw {current_yaw:.1f}°): X={avoidance_gps[0]:.1f}, Y={avoidance_gps[1]:.1f}, Z={avoidance_gps[2]:.1f}")
    else:
        # Fallback: sadece yukarı çık
        avoidance_gps = [
            current_gps[0],              # X aynı
            current_gps[1],              # Y aynı
            current_gps[2] + upward_distance   # Yukarı çık
        ]
        print(f"🎯 Kaçınma hedefi (yukarı): X={avoidance_gps[0]:.1f}, Y={avoidance_gps[1]:.1f}, Z={avoidance_gps[2]:.1f}")
    
    return avoidance_gps

def create_avoidance_action(obstacle_prediction, current_gps, current_yaw=None):
    """Legacy function - redirects to simple avoidance with yaw support"""
    return create_simple_avoidance_action(obstacle_prediction['obstacle_detected'], current_gps, current_yaw)

#--------------------- Modüler Hale Getir ---------------------
def calculate_target_direction(current_gps, target_pos):
    """Hedefe olan yön vektörünü hesapla - pose_to_euler ile uyumlu"""
    direction = np.array(target_pos) - np.array(current_gps)
    distance = np.linalg.norm(direction)
    
    if distance < 0.1:
        return [0, 0, 0], 0
    
    # Normalize et
    normalized_direction = direction / distance
    
    # Yaw hesapla (pose_to_euler ile uyumlu olması için)
    # pose_to_euler ZYX euler kullanıyor, bu yüzden aynı koordinat sistemini kullanmalıyız
    target_yaw = np.degrees(np.arctan2(direction[1], direction[0]))
    
    return normalized_direction, target_yaw

def is_looking_at_target(current_yaw, target_yaw, tolerance=15.0):
    """AUV'nin hedefe bakıp bakmadığını kontrol et"""
    angle_diff = abs(target_yaw - current_yaw)
    # 360 derece wrap-around
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    
    return angle_diff < tolerance

# Kamera display değişkenleri
display_active = True

def update_camera_display():
    """Real-time camera feed with obstacle predictions"""
    
    cv2.namedWindow('AUV Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AUV Camera Feed', 512, 512)  # Updated for 512x512 resolution
    
    while display_active:
        if current_camera_frame is not None:
            frame = current_camera_frame.copy()
            cv2.imshow('AUV Camera Feed', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
        time.sleep(0.03)  # ~30 FPS
    
    cv2.destroyAllWindows()

def draw_obstacle_info_on_frame(frame, obstacle_prediction, navigation_mode, step_count):
    """Kamera frame'ine basit engel bilgisi çiz"""
    if frame is None:
        return frame
    
    # Frame'i kopyala
    annotated_frame = frame.copy()
    
    # Üst kısım - navigation bilgisi
    cv2.putText(annotated_frame, f"Step: {step_count} | Mode: {navigation_mode}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Basit engel bilgisi
    if obstacle_prediction:
        prob = obstacle_prediction['obstacle_probability']
        detected = obstacle_prediction['obstacle_detected']
        
        # Engel durumuna göre renk
        if detected:
            color = (0, 0, 255)  # Kırmızı - engel var
            status = "ENGEL VAR!"
        else:
            color = (0, 255, 0)  # Yeşil - engel yok
            status = "ENGEL YOK"
        
        # Engel bilgilerini çiz
        cv2.putText(annotated_frame, status, 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(annotated_frame, f"Olasılık: {prob:.3f}", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Engel varsa çerçeve çiz
        if detected:
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 5)
    
    return annotated_frame

#--------------------- Modüler Hale Getir ---------------------


# Keyboard dinleyicisini başlat
start_listener()
# target_location = [-547, 107, -257] 
target_location = [-565, -25, -245]
force = 25

# CNN modelini yükle
print("🔄 Obstacle detection modeli yükleniyor...")
list_available_models()  # Show available models for debugging
model_loaded = load_latest_obstacle_model()
if not model_loaded:
    print("⚠️ Model yüklenemedi, engel algılama devre dışı!")

# Navigation state variables
navigation_mode = "DIRECT"  # DIRECT, AVOIDING, ROTATING, ASCENDING
step_count = 0
last_avoidance_time = 0
obstacle_detections = []
obstacle_clearance_height = 0  # Engelden kaçınmak için çıkılan yükseklik
original_target_z = target_location[2]  # Orijinal hedef Z koordinatı

# Movement completion tracking
avoidance_target_gps = None  # Hedeflenen kaçınma pozisyonu
movement_tolerance = 0.3  # Hedefe ne kadar yakın olunca "ulaşıldı" sayılacak (metre) - daha hassas

with holoocean.make("OpenWater-HoveringCamera", ticks_per_sec=60) as env:
    # Kamera frame rate'ini artırmaya çalış
    try:
        # Kamera sensor'ını bul ve frame rate'ini ayarla
        env.set_control_scheme("auv0", 1)
        # Camera frame rate ayarını dene (HoloOcean versiyonuna bağlı)
        print("📹 Kamera yüksek frame rate moduna geçiriliyor...")
    except:
        print("⚠️ Kamera frame rate ayarlanamadı, default kullanılıyor")
    # Local değişkenleri initialize et
    current_camera_frame = None
    
    print("🚀 HoloOcean Intelligent Navigation System Başlatıldı")
    print(f"🎯 Hedef Konum: {target_location}")
    print(f"🤖 ResNet18 CNN Engel Algılama: {'✅ Aktif (m.py modeli)' if model_loaded else '❌ Devre Dışı'}")
    print("📝 Q tuşu ile çıkış, ESC ile kamera penceresi kapanır")
    print("=" * 70)
    
    # Kamera display thread'ini başlat
    camera_thread = threading.Thread(target=update_camera_display)
    camera_thread.daemon = True
    camera_thread.start()
    
    while True:
        if 'q' in pressed_keys:
            display_active = False  # Kamera thread'ini durdur
            break
        command = parse_keys(pressed_keys, force)

        # Environment state al
        state = env.tick()
        
        # Hedef noktayı çiz
        env.draw_point(target_location, color=[255.0, 0.0, 0.0], lifetime=0)
        
        # Kaçınma hedefini çiz (eğer varsa)
        if avoidance_target_gps is not None:
            env.draw_point(avoidance_target_gps, color=[255.0, 255.0, 0.0], lifetime=0)  # Sarı renk

        if 'PoseSensor' in state and 'IMUSensor' in state and 'Camera' in state:
            # State bilgilerini çıkar
            camera_img, _, _ = process_observation(state)
            pose = state['PoseSensor']
            imu = state['IMUSensor']

            gps = pose_to_gps(state)
            yaw, pitch, roll = pose_to_euler(pose)
            
            # Kamera görüntüsünü global değişkene kaydet (display için)
            if isinstance(camera_img, torch.Tensor):
                # PyTorch tensor'ı OpenCV formatına çevir (CHW -> HWC, [0,1] -> [0,255])
                img_np = camera_img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                # RGB'den BGR'ye çevir (OpenCV için)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                current_camera_frame = img_np
            
            # Hedefe olan yön ve mesafeyi hesapla
            direction, target_yaw = calculate_target_direction(gps, target_location)
            target_yaw = int(target_yaw)
            distance_to_target = np.linalg.norm(np.array(target_location) - np.array(gps))
            
            # Hedefe bakıp bakmadığını kontrol et
            looking_at_target = is_looking_at_target(yaw, target_yaw)
            
            # Çarpışma kontrolü
            collision = detect_collision_by_diff(imu, threshold=10)
            if collision:
                print("💥 Çarpışma algılandı!")
                # Çarpışma durumunda kaçınma moduna geç ve mevcut yaw'ı kullan
                navigation_mode = "AVOIDING"
                last_avoidance_time = current_time if 'current_time' in locals() else time.time()
                
                # Çarpışma sonrası kaçınma hareketi - mevcut yaw ile
                collision_avoidance_gps = create_simple_avoidance_action(True, gps, yaw)
                if collision_avoidance_gps:
                    avoidance_target_gps = collision_avoidance_gps
                    print(f"🚨 Çarpışma sonrası kaçınma hareketi: {collision_avoidance_gps}")
                else:
                    print("⚠️ Çarpışma sonrası kaçınma hareketi oluşturulamadı!")
            
            # Basit CNN ile engel algılama - sadece VAR/YOK
            obstacle_prediction = None
            obstacle_detected = False
            if model_loaded:
                # Basit ikili prediction
                obstacle_prediction = predict_obstacle(camera_img)
                obstacle_detected = obstacle_prediction['obstacle_detected']
                
                if obstacle_detected:
                    current_time = time.time()
                    
                    # Basit engel kaydı
                    obstacle_detections.append({
                        'step': step_count,
                        'timestamp': current_time,
                        'probability': obstacle_prediction['obstacle_probability'],
                        'gps': gps
                    })
                    
                    # Basit console output
                    print(f"🚨 ENGEL ALGILANDI! Olasılık: {obstacle_prediction['obstacle_probability']:.3f}")
            
            # Kamera frame'ine obstacle bilgisi çiz
            if current_camera_frame is not None and obstacle_prediction:
                current_camera_frame = draw_obstacle_info_on_frame(
                    current_camera_frame, obstacle_prediction, navigation_mode, step_count
                )
            
            # ===== NAVIGASYON LOJİĞİ =====
            current_time = time.time()
            
            # 1. Hedefe ulaştı mı?
            if distance_to_target < 3.0:
                print("🎉 Hedefe ulaşıldı!")
                print(f"📊 Toplam step: {step_count}")
                print(f"⚠️ Toplam engel: {len(obstacle_detections)}")
                break
            
            # 2. Kaçınma modunda mı? Hareket tamamlandı mı kontrol et
            if navigation_mode == "AVOIDING" and avoidance_target_gps is not None:
                # Kaçınma hedefine ulaştı mı kontrol et
                distance_to_avoidance_target = np.linalg.norm(np.array(avoidance_target_gps) - np.array(gps))
                
                if distance_to_avoidance_target < movement_tolerance:
                    print(f"✅ Kaçınma hareketi tamamlandı! (Mesafe: {distance_to_avoidance_target:.1f}m)")
                    navigation_mode = "DIRECT"  # Normal navigasyona geri dön
                    avoidance_target_gps = None  # Hedefi sıfırla
                    
                else:
                    # Hala kaçınma hedefine gitmeye devam et
                    pd_command = [
                        avoidance_target_gps[0],  # Kaçınma x
                        avoidance_target_gps[1],  # Kaçınma y
                        avoidance_target_gps[2],  # Kaçınma z
                        0,                        # roll = 0
                        0,                        # pitch = 0
                        yaw                       # mevcut yaw'ı koru
                    ]
                    print(f"🚨 Kaçınma devam ediyor... Kalan mesafe: {distance_to_avoidance_target:.1f}m")
                    
                    # Kaçınma komutunu gönder ve bu döngüyü atla
                    state = env.step(pd_command)
                    step_count += 1
                    continue
                    
            # 3. Yeni engel algılandı mı? (ve şu anda kaçınma modunda değil mi?)
            elif obstacle_detected and navigation_mode != "AVOIDING" and (current_time - last_avoidance_time > 5.0):
                navigation_mode = "AVOIDING"
                last_avoidance_time = current_time
                
                # Yeni kaçınma hareketi oluştur (mevcut yaw ile)
                avoidance_gps = create_avoidance_action(obstacle_prediction, gps, yaw)
                if avoidance_gps:
                    avoidance_target_gps = avoidance_gps  # Hedefi kaydet
                    pd_command = [
                        avoidance_gps[0],  # Kaçınma x
                        avoidance_gps[1],  # Kaçınma y
                        avoidance_gps[2],  # Kaçınma z
                        0,                 # roll = 0
                        0,                 # pitch = 0
                        yaw                # mevcut yaw'ı koru
                    ]
                    print(f"🚨 YENİ kaçınma hareketi başlatıldı: {avoidance_gps}")
                else:
                    # Kaçınma hareketi üretilemedi, dur
                    pd_command = [gps[0], gps[1], gps[2], 0, 0, yaw]
                    avoidance_target_gps = None
                    
            # 4. Hedefe bakmıyor mu? Önce dön (sadece kaçınma modunda değilken)
            elif not looking_at_target and navigation_mode != "AVOIDING":
                navigation_mode = "ROTATING"
                pd_command = [
                    gps[0],    # x pozisyon aynı
                    gps[1],    # y pozisyon aynı
                    gps[2],    # z pozisyon aynı
                    0,         # roll = 0
                    0,         # pitch = 0
                    target_yaw # hedefe dön
                ]
                print(f"🔄 Hedefe dönülüyor... (Fark: {target_yaw - yaw:.1f}°)")
                
            # 5. Normal ilerleme - hedefe git (sadece kaçınma modunda değilken)
            elif navigation_mode != "AVOIDING":
                navigation_mode = "DIRECT"
                
                # Mesafeye göre hız ayarla (sadece bilgi amaçlı)
                if distance_to_target > 30:
                    step_size = 8.0
                elif distance_to_target > 15:
                    step_size = 5.0
                elif distance_to_target > 8:
                    step_size = 3.0
                else:
                    step_size = 1.5
                
                # Doğrudan hedefe git
                pd_command = [
                    target_location[0],  # hedefe doğru x
                    target_location[1],  # hedefe doğru y
                    target_location[2],  # hedefe doğru z
                    0,                   # roll = 0
                    0,                   # pitch = 0
                    target_yaw           # hedefe bak
                ]
                print(f"🏃 Hedefe gidiliyor (Hız: {step_size:.1f}, Mesafe: {distance_to_target:.1f})")
            
            # Komutu gönder
            state = env.step(pd_command)
            step_count += 1
            
            # Açı farkını hesapla (display için)
            yaw_diff = target_yaw - yaw
            if yaw_diff > 180:
                yaw_diff -= 360
            elif yaw_diff < -180:
                yaw_diff += 360
            yaw_diff = int(yaw_diff)
        

            # Tkinter penceresini başlat (ilk seferde bir kez oluştur)
            if "root" not in globals():
                root = tk.Tk()
                root.title("AUV Intelligent Navigation System")
                text_widget = tk.Text(root, height=15, width=80, font=("Consolas", 10))
                text_widget.pack()
                root.update()

            # Ekrana yazdırmak için metni hazırla
            obstacle_info = ""
            if obstacle_prediction:
                status = "VAR" if obstacle_prediction['obstacle_detected'] else "YOK"
                obstacle_info = (
                    f"🚨 Engel: {status} ({obstacle_prediction['obstacle_probability']:.3f})\n"
                )
            
            # Kaçınma hedefi bilgisi
            avoidance_info = ""
            if avoidance_target_gps is not None:
                distance_to_avoidance = np.linalg.norm(np.array(avoidance_target_gps) - np.array(gps))
                avoidance_info = (
                    f"🎯 Kaçınma Hedefi: [{avoidance_target_gps[0]:.1f}, {avoidance_target_gps[1]:.1f}, {avoidance_target_gps[2]:.1f}]\n"
                    f"📏 Kaçınma Mesafesi: {distance_to_avoidance:.1f}m\n"
                )
            
            # CNN Performance Stats - ResNet18 Model aktif
            cnn_stats = ""
            if model_loaded:
                cnn_stats = "🚀 ResNet18 Model: Aktif | m.py'den yüklendi\n"
            
            output = (
                f"=== AUV INTELLIGENT NAVIGATION ===\n"
                f"📊 Step: {step_count} | Mode: {navigation_mode}\n"
                f"📍 GPS: [{gps[0]:.1f}, {gps[1]:.1f}, {gps[2]:.1f}]\n"
                f"🎯 Target: {target_location}\n"
                f"📏 Distance: {distance_to_target:.2f}m\n"
                f"🧭 Current Yaw: {yaw:.1f}°\n"
                f"🎯 Target Yaw: {target_yaw:.1f}°\n"
                f"🔄 Yaw Diff: {yaw_diff:.1f}°\n"
                f"👀 Looking at target: {'✅' if looking_at_target else '❌'}\n"
                f"🎮 PD Command: [x={pd_command[0]:.1f}, y={pd_command[1]:.1f}, z={pd_command[2]:.1f}, "
                f"roll={pd_command[3]:.1f}, pitch={pd_command[4]:.1f}, yaw={pd_command[5]:.1f}]\n"
                f"{obstacle_info}"
                f"{avoidance_info}"
                f"{cnn_stats}"
                f"⚠️ Total Obstacles: {len(obstacle_detections)}\n"
                f"🤖 ResNet18 CNN Status: {'✅ Active (m.py model)' if model_loaded else '❌ Disabled'}\n"
                f"{'='*50}\n"
            )

            # Text widget'ı güncelle
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, output)
            root.update()

# Program sonlandırıldığında özet rapor
print("\n" + "="*60)
print("🏁 INTELLIGENT NAVIGATION TAMAMLANDI - ÖZET RAPOR")
print("="*60)
print(f"📊 Toplam step sayısı: {step_count}")
print(f"⚠️ Toplam engel algılama: {len(obstacle_detections)}")
if obstacle_detections:
    avg_prob = np.mean([obs['probability'] for obs in obstacle_detections])
    print(f"📈 Ortalama engel olasılığı: {avg_prob:.3f}")
    print(f"🔢 En yüksek engel olasılığı: {max([obs['probability'] for obs in obstacle_detections]):.3f}")
print(f"🤖 CNN Model Durumu: {'Aktif' if model_loaded else 'Devre Dışı'}")
print("👋 Program sonlandırıldı.")
print("="*60)


