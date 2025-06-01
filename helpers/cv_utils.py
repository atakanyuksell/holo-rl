import cv2
import numpy as np

def show_camera_image(camera_img, window_name="Camera"):
    """
    Kamera görüntüsünü OpenCV ile ekranda gösterir.
    Args:
        camera_img (np.ndarray): Kamera görüntüsü (H, W, 3) veya (3, H, W)
        window_name (str): Pencere adı
    """
    if camera_img is None:
        return
    img = camera_img
    # Eğer (3, H, W) ise (H, W, 3)'e çevir
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # Eğer görüntü BGR değilse, RGB'den BGR'ye çevir
    if img.shape[-1] == 3:
        img = img[..., ::-1]
    cv2.imshow(window_name, img)
    cv2.waitKey(1)
