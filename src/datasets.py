import os
import torch
from torch.utils.data import Dataset

class ProcessedImageDataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        
        class_info = torch.load(os.path.join(processed_dir, 'metadata_class_info.pt'))
        self.classes = class_info['classes']
        self.class_to_idx = class_info['class_to_idx']
        
        self.class_files = [f for f in os.listdir(processed_dir) if f.startswith('class_') and f.endswith('.pt')]
        
        self.sample_map = []
        for class_file in self.class_files:
            data = torch.load(os.path.join(processed_dir, class_file))
            num_samples = data['X'].shape[0]
            self.sample_map.extend([(class_file, i) for i in range(num_samples)])
    
    def __len__(self):
        return len(self.sample_map)
    
    def __getitem__(self, idx):
        class_file, sample_idx = self.sample_map[idx]
        data = torch.load(os.path.join(self.processed_dir, class_file))
        return data['X'][sample_idx], data['Y'][sample_idx]
