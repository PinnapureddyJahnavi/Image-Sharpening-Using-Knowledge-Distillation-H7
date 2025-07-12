
import os, cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_ssim(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(img1, img2)

scores = []
for img in os.listdir("data/raw"):
    ref = f"data/raw/{img}"
    sr = f"data/lr/{img}"  # or "data/sr/{img}" if output saved
    if os.path.exists(sr):
        score = calculate_ssim(ref, sr)
        scores.append(score)

print("Average SSIM Score:", np.mean(scores) * 100)
