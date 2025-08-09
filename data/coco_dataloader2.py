import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re
# Gender-specific keywords
male_keywords = ['male', 'boy', 'man', 'gentleman', 'boys', 'men', 'males', 'gentlemen', 'father', 'boyfriend']
female_keywords = ['female', 'girl', 'woman', 'lady', 'girls', 'women', 'females', 'ladies', 'mother', 'girlfriend']
# Select two object categories for bias
bias_category_1 = ['sports ball', 'baseball bat', 'skateboard','suitcase', 'frisbee', 'skis', 'surfboard', 'tennis racket']
bias_category_2 = ['oven', 'refrigerator', 'sink','cup', 'fork', 'knife', 'spoon', 'bowl']

# Load COCO categories
def load_coco_categories(instance_path):
    with open(instance_path, 'r') as f:
        data = json.load(f)
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    return categories

# Extract gender labels using captions
def extract_gender_labels(captions_path):
    with open(captions_path, 'r') as f:
        data = json.load(f)
    gender_labels = {}

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption'].lower()
        male_match = any(re.search(r'\b' + word + r'\b', caption) for word in male_keywords)
        female_match = any(re.search(r'\b' + word + r'\b', caption) for word in female_keywords)

        if male_match and not female_match:
            label = 1
        elif female_match and not male_match:
            label = 0
        else:
            label = -1  # Ambiguous

        if image_id not in gender_labels:
            gender_labels[image_id] = []
        gender_labels[image_id].append(label)

    # Majority vote for final label
    return {k: max(set(v), key=v.count) for k, v in gender_labels.items() if -1 not in v}

# Extract bias labels from instance data
def extract_bias_labels(instances_path, categories, gender_labels):
    with open(instances_path, 'r') as f:
        data = json.load(f)
    bias_labels = {}
    
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in gender_labels:
            continue
        
        category_name = categories[ann['category_id']]
        if image_id not in bias_labels:
            bias_labels[image_id] = [0, 0]
        if category_name in bias_category_1:
            bias_labels[image_id][0] = 1
        elif category_name in bias_category_2:
            bias_labels[image_id][1] = 1

    return bias_labels

# Custom COCO Dataset
class CocoGenderBiasDataset(Dataset):
    def __init__(self, image_dir, captions_path, instances_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.categories = load_coco_categories(instances_path)
        self.gender_labels = extract_gender_labels(captions_path)
        self.bias_labels = extract_bias_labels(instances_path, self.categories, self.gender_labels)
        self.image_ids = list(self.gender_labels.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id:012d}.jpg")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        gender_label = self.gender_labels[image_id]
        bias_label_1, bias_label_2 = self.bias_labels.get(image_id, (0, 0))
        return image, gender_label, torch.tensor([bias_label_1, bias_label_2])

# Print Dataset Statistics
def print_dataset_stats(dataset):
    gender_count = {0: 0, 1: 0}
    bias_group_count = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    for _, gender, biases in dataset:
        gender_count[gender] += 1
        bias_tuple = tuple(biases.numpy())
        bias_group_count[bias_tuple] += 1

    print("Gender Distribution:")
    print(f"  Female: {gender_count[0]} | Male: {gender_count[1]}")
    print("\nBias Group Distribution (Bias1, Bias2):")
    for key, value in bias_group_count.items():
        print(f"  {key}: {value} images")

# Data Loader and Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def create_dataloader(image_dir, captions_path, instances_path, batch_size=64, shuffle=True):
    dataset = CocoGenderBiasDataset(
        image_dir=image_dir,
        captions_path=captions_path,
        instances_path=instances_path,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print_dataset_stats(dataset)
    return dataloader

# Create Train and Test Dataloaders
# print("Creating Train Dataloader...")
# train_dataloader = create_dataloader(
#     image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/train2017',
#     captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_train2017.json',
#     instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_train2017.json'
# )

# print("\nCreating Test Dataloader...")
# test_dataloader = create_dataloader(
#     image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/val2017',
#     captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_val2017.json',
#     instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_val2017.json',
#     shuffle=False
# )