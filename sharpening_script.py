import torch
import torch.nn as nn
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

# The following line is for installing libraries, typically done once in the terminal
# pip install torch torchvision scikit-image opencv-python matplotlib

# Create a directory to store images if it doesn't exist
os.makedirs("images", exist_ok=True)

# If running as a standalone script, manually move your image files
# to the 'images' directory. For example, you can place your images
# in the 'images' folder created in the same directory as this script.


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SharpnessDataset(Dataset):
    # Added optional img_path parameter
    def __init__(self, path="images", img_path=None):
        self.path = path
        self.img_path = img_path
        self.transform = transforms.ToTensor()
        self.files = []

        if self.img_path:
            # If a specific image path is provided, only load that file if it exists
            if os.path.exists(self.img_path):
                self.files = [os.path.basename(self.img_path)]
                self.path = os.path.dirname(self.img_path) # Update path to directory of the image
                print(f"Loading single image: {self.img_path}")
            else:
                print(f"Image file not found at: {self.img_path}. Falling back to loading from directory: {self.path}")
                if os.path.isdir(self.path):
                     self.files = os.listdir(self.path)
                else:
                    print(f"Image directory not found at: {self.path}")
        else:
            # Otherwise, load all files from the directory
            if os.path.isdir(self.path):
                 self.files = os.listdir(self.path)
                 print(f"Loading all images from directory: {self.path}")
            else:
                print(f"Image directory not found at: {self.path}")

        # Filter out non-image files if necessary (optional)
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]


    def __getitem__(self, idx):
        # Modified to handle both single file and directory listing
        img_name = self.files[idx]
        img_path = os.path.join(self.path, img_name)

        hr = cv2.imread(img_path)
        if hr is None:
             # Skip this image and return None, will be handled in DataLoader
             print(f"Warning: Failed to load image at: {img_path}. Skipping.")
             return None, None

        # Convert BGR to RGB for consistency with matplotlib and PIL
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = cv2.resize(hr, (384, 384))
        lr = cv2.resize(hr, (128, 128), interpolation=cv2.INTER_LINEAR)
        upscaled = cv2.resize(lr, (384, 384), interpolation=cv2.INTER_LINEAR)

        # Ensure consistent data type and range (0-255) before converting to Tensor
        upscaled = upscaled.astype(np.float32) / 255.0
        hr = hr.astype(np.float32) / 255.0

        return self.transform(Image.fromarray((upscaled * 255).astype(np.uint8))), self.transform(Image.fromarray((hr * 255).astype(np.uint8)))


    def __len__(self):
        return len(self.files)

def compute_ssim(img1, img2):
    # Convert tensors to numpy arrays for SSIM calculation
    # Permute dimensions from (C, H, W) to (H, W, C)
    img1 = img1.permute(1, 2, 0).cpu().numpy()
    img2 = img2.permute(1, 2, 0).cpu().numpy()
    # Ensure data range is between 0 and 1 if image data is normalized
    # SSIM requires images with the same data range. Our tensors are 0-1.
    return ssim(img1, img2, channel_axis=2, data_range=1)

# Main execution block
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    student = StudentModel().to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Instantiate dataset with a specific image path (example)
    # Replace 'images/butterfly.jpg' with the actual path to your image file
    # To process all images in the 'images' directory, use: dataset = SharpnessDataset()

    # Check if a specific image file exists, otherwise use the default directory
    # You can modify the image_to_process variable to the path of your image
    image_to_process = 'images/butterfly.jpg'
    if os.path.exists(image_to_process):
        dataset = SharpnessDataset(img_path=image_to_process)
    else:
        print(f"Specific image '{image_to_process}' not found. Loading all images from 'images' directory.")
        dataset = SharpnessDataset()


    # Check if the dataset is not empty before creating the DataLoader
    if len(dataset) == 0:
        print("No images found in the specified path or directory. Please ensure the path is correct and contains images.")
    else:
        # If a single image path is provided, batch size should be 1 and shuffle should be False for deterministic output
        # If processing a directory, use batch_size=1 for visualization purposes later
        batch_size = 1
        shuffle = False if dataset.img_path and os.path.exists(dataset.img_path) else True # Shuffle only if processing a directory


        # Custom collate function to handle None values from dataset (due to loading errors)
        def collate_fn(batch):
            batch = list(filter(lambda x: x[0] is not None, batch))
            if len(batch) == 0:
                return None, None
            return torch.utils.data.dataloader.default_collate(batch)

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


        # Training loop (only runs if using a directory with multiple images)
        # We typically don't train on a single image for sharpening
        if not (dataset.img_path and os.path.exists(dataset.img_path)) and len(dataset) > 1:
            print("Starting training...")
            for epoch in range(5):
                student.train()
                running_loss = 0
                num_batches = 0
                for input_img, target_img in loader:
                     # Skip if image loading failed or batch is empty after filtering
                    if input_img is None or target_img is None:
                        continue

                    input_img, target_img = input_img.to(device), target_img.to(device)

                    output = student(input_img)
                    loss = criterion(output, target_img)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1

                if num_batches > 0:
                     print(f"Epoch {epoch+1} Loss: {running_loss/num_batches:.4f}")
                else:
                     print(f"Epoch {epoch+1}: No valid images processed for training.")
        elif len(dataset) == 1:
             print("Processing a single image. Skipping training loop.")
        else:
             print("Not enough images in the directory for training. Skipping training loop.")


        # Evaluation and SSIM calculation
        student.eval()
        total_ssim = 0
        num_images = 0

        print("Starting evaluation...")
        with torch.no_grad():
            for input_img, target_img in loader:
                # Skip if image loading failed or batch is empty after filtering
                if input_img is None or target_img is None:
                    continue

                input_img, target_img = input_img.to(device), target_img.to(device)
                output = student(input_img)

                # SSIM calculation is done per image, assuming batch size 1 for evaluation/visualization
                for i in range(input_img.size(0)):
                     total_ssim += compute_ssim(output[i], target_img[i])
                     num_images += 1


        if num_images > 0:
            print("Average SSIM:", total_ssim / num_images)
        else:
            print("No valid images processed for SSIM calculation.")


        # Visualization
        student.eval()

        print("Starting visualization...")
        # Iterate through all images in the loader for visualization
        with torch.no_grad():
            for i, (input_img, target_img) in enumerate(loader):
                 # Skip if image loading failed or batch is empty after filtering
                if input_img is None or target_img is None:
                    continue

                input_img, target_img = input_img.to(device), target_img.to(device)
                output = student(input_img)

                # Process each image in the batch (batch size is 1 for single image or during visualization)
                for j in range(input_img.size(0)):
                    # Convert tensor to numpy array for visualization
                    input_np = input_img[j].permute(1, 2, 0).cpu().numpy()
                    target_np = target_img[j].permute(1, 2, 0).cpu().numpy()
                    output_np = output[j].permute(1, 2, 0).cpu().numpy()

                    # Ensure image data is in the correct range for display (0-1)
                    input_np = np.clip(input_np, 0, 1)
                    target_np = np.clip(target_np, 0, 1)
                    output_np = np.clip(output_np, 0, 1)


                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                    axes[0].imshow(input_np)
                    axes[0].set_title("Original Image (Low Resolution)")
                    axes[0].axis('off')

                    axes[1].imshow(target_np)
                    axes[1].set_title("Target Image (High Resolution)")
                    axes[1].axis('off')

                    axes[2].imshow(output_np)
                    axes[2].set_title("Sharpened Output (Student Model)")
                    axes[2].axis('off')

                    plt.show()
