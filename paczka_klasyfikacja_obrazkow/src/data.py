import os
import torch
from torchvision import datasets, transforms
from PIL import Image

data_dir = './data/train/'
output_dir = './data/processed/'

os.makedirs(output_dir, exist_ok=True)

transform_train2 = transforms.Compose([
    transforms.RandomRotation(24),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transforms_list = [transform_train2]

def preprocess_and_save_by_class(source_dir, output_dir, augmentation_transforms):
    dataset = datasets.ImageFolder(root=source_dir)
    
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx
    
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Class to index mapping: {class_to_idx}")
    
    images_by_class = {cls: [] for cls in classes}
    
    for img_path, label in dataset.samples:
        class_name = classes[label]
        images_by_class[class_name].append((img_path, label))
    
    class_info = {
        'classes': classes,
        'class_to_idx': class_to_idx
    }
    torch.save(class_info, os.path.join(output_dir, 'metadata_class_info.pt'))
    
    for class_name, images in images_by_class.items():
        print(f"Processing class: {class_name} with {len(images)} images")
        
        processed_images = []
        labels = []
        
        for img_path, label in images:
            print(f"  Processing image: {os.path.basename(img_path)}")
            
            try:
                img = Image.open(img_path).convert('RGB')
                
                base_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                
                processed_img = base_transform(img)
                processed_images.append(processed_img)
                labels.append(label)
                
                for transform in augmentation_transforms:
                    augmented_img = transform(img)
                    processed_images.append(augmented_img)
                    labels.append(label)
                    
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        X = torch.stack(processed_images)
        Y = torch.tensor(labels)
        
        print(f"  Class {class_name} dataset created with shape X: {X.shape}, Y: {Y.shape}")
        
        class_file = os.path.join(output_dir, f'class_{class_name}.pt')
        torch.save({'X': X, 'Y': Y, 'class_name': class_name}, class_file)
        print(f"  Class {class_name} dataset saved to {class_file}")
        
        del X, Y, processed_images, labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == "__main__":
    preprocess_and_save_by_class(data_dir, output_dir, transforms_list)
