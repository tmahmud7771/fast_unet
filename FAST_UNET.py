import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import os
from PIL import Image
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from functools import lru_cache

# Fast Dataset with caching


class FastDataset(Dataset):
    def __init__(self, json_path, image_dir):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.image_paths = [os.path.join(
            image_dir, item['image_name']) for item in data]
        self.labels = [item['label'] for item in data]

        # Fast transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),  # Very small size for speed
            transforms.ToTensor(),
        ])

    @lru_cache(maxsize=None)
    def _load_image(self, path):
        # Cache image loading
        return Image.open(path).convert('L')  # Convert to grayscale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self._load_image(self.image_paths[idx])
        img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)

# Minimal U-Net


class MinimalUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        # Minimal encoder (single channel input - grayscale)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Minimal decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(16, num_classes, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


def train_fast():
    # Configuration
    json_path = 'path/to/your/annotations.json'
    image_dir = 'path/to/your/images'

    # Hyperparameters optimized for speed
    batch_size = 128  # Large batch size
    num_epochs = 10   # Minimal epochs
    learning_rate = 1e-2  # Aggressive learning rate

    # Set number of CPU threads
    torch.set_num_threads(4)  # Adjust based on your CPU

    # Create dataset and loader
    dataset = FastDataset(json_path, image_dir)

    # Fast data loading
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single thread loading
        pin_memory=False,
        drop_last=True  # Speed up by dropping incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    # Initialize model
    model = MinimalUNet()

    # Fast optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Learning rate scheduler for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader)
    )

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))

            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{
                      len(train_loader)}, Loss: {loss.item():.4f}')

        # Quick validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss += criterion(outputs, labels.view(-1, 1)).item()
        val_loss /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

        # Save model every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')


if __name__ == '__main__':
    train_fast()
