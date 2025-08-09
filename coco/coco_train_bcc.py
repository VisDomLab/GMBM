import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from data.coco_dataloader2 import create_dataloader
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

# Initialize Models for Bias 1 and Bias 2
model_bias1 = ResNet18(num_classes=2).to(device)
model_bias2 = ResNet18(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model_bias1.parameters(), lr=0.0001)
optimizer2 = optim.Adam(model_bias2.parameters(), lr=0.0001)

# Training function
def train_model(model, optimizer, bias_index, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Bias {bias_index+1} - Epoch {epoch+1}/{num_epochs}")
        for images, _, biases in progress_bar:
            images = images.to(device)
            labels = biases[:, bias_index].to(device)

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

        print(f"Bias {bias_index+1} - Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        torch.save(model.state_dict(), f"real_resnet18_bias{bias_index+1}_epoch_{epoch+1}.pth")
        print(f"Model saved as resnet18_bias{bias_index+1}_epoch_{epoch+1}.pth")

# Run Training
if __name__ == "__main__":
    print("Starting Training for Bias 1...")
    train_model(model_bias1, optimizer1, bias_index=0, num_epochs=10)
    
    print("\nStarting Training for Bias 2...")
    train_model(model_bias2, optimizer2, bias_index=1, num_epochs=10)

    print("Training Complete!")
