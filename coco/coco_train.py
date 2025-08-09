import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from data.coco_dataloader import create_dataloader
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataloaders
print("Loading data...")
train_dataloader = create_dataloader(
    image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/train2017',
    captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_train2017.json',
    instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_train2017.json'
)

test_dataloader = create_dataloader(
    image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/val2017',
    captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_val2017.json',
    instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_val2017.json',
    shuffle=False
)

# Initialize Model
model = ResNet18(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        torch.save(model.state_dict(), f"real_resnet18_epoch_{epoch+1}.pth")
        print(f"Model saved as resnet18_epoch_{epoch+1}.pth")
    return model
# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(test_dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
def evaluate_groupwise(model):
    correct = 0
    total = 0

    # Initialize group metrics for each bias (4 groups per bias)
    group_metrics_1 = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
    group_metrics_2 = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}

    with torch.no_grad():
        for images, labels, biases in tqdm(test_dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Overall accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Group-wise bias accuracy
            for i in range(labels.size(0)):
                group_1 = (labels[i].item(), biases[i][0].item())
                group_2 = (labels[i].item(), biases[i][1].item())
                
                group_metrics_1[group_1][0] += (predicted[i] == labels[i]).item()
                group_metrics_1[group_1][1] += 1
                
                group_metrics_2[group_2][0] += (predicted[i] == labels[i]).item()
                group_metrics_2[group_2][1] += 1

    print(f"Overall Test Accuracy: {100 * correct / total:.2f}%")

    # Calculate Unbiased Accuracy (Average of all group accuracies)
    unbiased_acc_1 = sum(100 * group_metrics_1[key][0] / group_metrics_1[key][1] if group_metrics_1[key][1] > 0 else 0 for key in group_metrics_1) / 4
    unbiased_acc_2 = sum(100 * group_metrics_2[key][0] / group_metrics_2[key][1] if group_metrics_2[key][1] > 0 else 0 for key in group_metrics_2) / 4
    print(f"Unbiased Accuracy (Bias 1): {unbiased_acc_1:.2f}%")
    print(f"Unbiased Accuracy (Bias 2): {unbiased_acc_2:.2f}%")

    # Calculate Bias Conflicting Accuracy for Bias 1 using (1,0) and (0,1)
    conflict_acc_1 = sum(100 * group_metrics_1[key][0] / group_metrics_1[key][1] if group_metrics_1[key][1] > 0 else 0 for key in [(1, 0), (0, 1)]) / 2
    
    # Calculate Bias Conflicting Accuracy for Bias 2 using (1,1) and (0,0)
    conflict_acc_2 = sum(100 * group_metrics_2[key][0] / group_metrics_2[key][1] if group_metrics_2[key][1] > 0 else 0 for key in [(1, 1), (0, 0)]) / 2
    
    print(f"Bias Conflicting Accuracy (Bias 1): {conflict_acc_1:.2f}%")
    print(f"Bias Conflicting Accuracy (Bias 2): {conflict_acc_2:.2f}%")
# Run Training and Evaluation
if __name__ == "__main__":
    print("Starting Training...")
    model = train_model(num_epochs=10)
    print("\nEvaluating on Test Set...")
    evaluate_groupwise(model)
