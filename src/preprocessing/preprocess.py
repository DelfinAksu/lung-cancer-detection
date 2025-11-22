import cv2
import numpy as np

def preprocess_image(img):
    """
    Preprocessing used inside Keras ImageDataGenerator.

    Keras -> img: float32, shape (H, W, 3), RGB
    1) Convert to uint8
    2) Convert to grayscale
    3) Apply CLAHE
    4) Normalize to [0,1]
    5) Stack to 3 channels again
    """

    # 1) Keras'tan gelen float32'yi 8-bit'e çevir
    if img.dtype != np.uint8:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_uint8 = img

    # 2) RGB -> Gray (Keras RGB kullanıyor)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # 3) CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12))
    enhanced = clahe.apply(gray)   # burada artık hata yok

    # 4) Normalize [0,1]
    norm = enhanced.astype(np.float32) / 255.0

    # 5) 3 kanala geri döndür (CNN input için)
    img_3ch = np.stack([norm, norm, norm], axis=-1)

    return img_3ch
