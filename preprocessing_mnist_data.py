import os
import struct
import numpy as np
from PIL import Image
import pandas as pd

def read_mnist_images(filename):
    """Read MNIST image file in ubyte format"""
    with open(filename, 'rb') as f:
        # Read header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def read_mnist_labels(filename):
    """Read MNIST label file in ubyte format"""
    with open(filename, 'rb') as f:
        # Read header
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def convert_mnist_to_images(data_dir, output_dir):
    """Convert MNIST ubyte files to individual image files"""
    
    # Define file paths
    train_images_file = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_file = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_file = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Lists to store image paths and labels for CSV files
    train_data = []
    test_data = []
    
    # Process training data
    print("Processing training data...")
    train_images = read_mnist_images(train_images_file)
    train_labels = read_mnist_labels(train_labels_file)
    
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        # Create filename
        filename = f"train_{i:05d}_label_{label}.png"
        filepath = os.path.join(train_dir, filename)
        
        # Convert numpy array to PIL Image and save
        img = Image.fromarray(image, mode='L')
        img.save(filepath)
        
        # Add to data list
        train_data.append({
            'image_path': f"train/{filename}",
            'label': int(label),
            'absolute_path': filepath
        })
        
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} training images")
    
    # Process test data
    print("Processing test data...")
    test_images = read_mnist_images(test_images_file)
    test_labels = read_mnist_labels(test_labels_file)
    
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        # Create filename
        filename = f"test_{i:05d}_label_{label}.png"
        filepath = os.path.join(test_dir, filename)
        
        # Convert numpy array to PIL Image and save
        img = Image.fromarray(image, mode='L')
        img.save(filepath)
        
        # Add to data list
        test_data.append({
            'image_path': f"test/{filename}",
            'label': int(label),
            'absolute_path': filepath
        })
        
        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1} test images")
    
    # Create CSV files with image paths and labels
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Save CSV files
    train_csv_path = os.path.join(output_dir, 'train_labels.csv')
    test_csv_path = os.path.join(output_dir, 'test_labels.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    print(f"\nConversion completed!")
    print(f"Training images: {len(train_data)} saved in {train_dir}")
    print(f"Test images: {len(test_data)} saved in {test_dir}")
    print(f"Training labels CSV: {train_csv_path}")
    print(f"Test labels CSV: {test_csv_path}")
    
    # Print some statistics
    print(f"\nDataset statistics:")
    print(f"Training set - Total: {len(train_data)}")
    print("Training set - Label distribution:")
    print(train_df['label'].value_counts().sort_index())
    
    print(f"\nTest set - Total: {len(test_data)}")
    print("Test set - Label distribution:")
    print(test_df['label'].value_counts().sort_index())
    
    return train_df, test_df

# Usage
if __name__ == "__main__":
    # Define paths
    data_dir = "/Users/qureshsu/Learning/TSAI/ERAV4/session4/train_deploy_mnist_cnn/train_deploy_mnist_cnn/data/MNIST/raw"
    output_dir = "/Users/qureshsu/Learning/TSAI/ERAV4/session4/train_deploy_mnist_cnn/train_deploy_mnist_cnn/mnist_images"
    
    # Convert MNIST data
    train_df, test_df = convert_mnist_to_images(data_dir, output_dir)
    
    # Display first few entries
    print("\nFirst 5 training entries:")
    print(train_df.head())
    
    print("\nFirst 5 test entries:")
    print(test_df.head())