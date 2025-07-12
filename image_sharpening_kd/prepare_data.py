
import cv2, os
from glob import glob

def downscale_images(input_dir, output_dir, scale=0.5):
    os.makedirs(output_dir, exist_ok=True)
    for img_path in glob(f"{input_dir}/*.jpg"):
        img = cv2.imread(img_path)
        img_lr = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        img_lr_up = cv2.resize(img_lr, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), img_lr_up)

downscale_images("data/raw", "data/lr")
