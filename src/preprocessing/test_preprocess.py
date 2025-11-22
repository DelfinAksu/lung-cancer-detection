import cv2
import glob
import matplotlib.pyplot as plt
from src.preprocessing.preprocess import preprocess_image

# Klasördeki ilk test görüntüsünü otomatik al
image_list = glob.glob("data/raw/chest_ct/test/normal/*.png")

if not image_list:
    raise Exception("No images found in test/normal folder!")

TEST_IMG_PATH = image_list[0]
print("Using image:", TEST_IMG_PATH)

# 1) Load original
orig = cv2.imread(TEST_IMG_PATH)

# 2) Apply preprocessing
processed = preprocess_image(orig)

# 3) Show side-by-side
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,2,2)
plt.title("CLAHE Applied")
plt.imshow(processed, cmap='gray')
plt.axis('off')

plt.show()
