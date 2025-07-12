
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.files = sorted(os.listdir(lr_dir))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lr_img = Image.open(os.path.join(self.lr_dir, self.files[idx])).convert('RGB')
        hr_img = Image.open(os.path.join(self.hr_dir, self.files[idx])).convert('RGB')
        return self.transform(lr_img), self.transform(hr_img)
