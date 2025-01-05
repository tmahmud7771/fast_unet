import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import os


class FastInference:
    def __init__(self, model_path):
        # Load model
        self.model = MinimalUNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Setup transforms - must match training
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def preprocess_image(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path)
        image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension

    @torch.no_grad()  # Disable gradient computation for inference
    def predict(self, image_path):
        # Preprocess
        image = self.preprocess_image(image_path)

        # Inference
        output = self.model(image)

        # Get prediction
        prediction = output.squeeze().numpy()
        return prediction

# Example usage


def run_inference():
    # Configuration
    model_path = 'model_epoch_10.pth'  # Use your saved model path
    test_image_dir = 'path/to/test/images'

    # Initialize inference
    inferencer = FastInference(model_path)

    # Process all images in directory
    results = {}
    for image_file in os.listdir(test_image_dir):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(test_image_dir, image_file)

            # Get prediction
            prediction = inferencer.predict(image_path)
            results[image_file] = prediction

            # Print result
            print(f"Image: {image_file}, Prediction: {prediction}")

    return results

# Batch inference for multiple images


def batch_inference(image_paths, model_path, batch_size=32):
    # Initialize model
    model = MinimalUNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    results = {}

    # Process in batches
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            # Prepare batch
            for img_path in batch_paths:
                image = Image.open(img_path)
                image = transform(image)
                batch_images.append(image)

            # Stack images into a batch
            batch_tensor = torch.stack(batch_images)

            # Get predictions
            outputs = model(batch_tensor)
            predictions = outputs.squeeze().numpy()

            # Store results
            for path, pred in zip(batch_paths, predictions):
                results[path] = pred

    return results


if __name__ == '__main__':
    # Example usage
    print("Running single image inference...")
    results = run_inference()

    print("\nRunning batch inference...")
    image_paths = [
        'path/to/image1.jpg',
        'path/to/image2.jpg',
        # Add more paths as needed
    ]
    batch_results = batch_inference(image_paths, 'model_epoch_10.pth')

    # Print batch results
    for path, prediction in batch_results.items():
        print(f"Image: {path}, Prediction: {prediction}")
