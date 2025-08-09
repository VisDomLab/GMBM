import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from data.coco_dataloader2 import create_dataloader
from tqdm import tqdm
#from evaluate_coco_maba2 import compute_maba, compute_maba_weighted, compute_maba_with_threshold
from sba_metric_coco import compute_sba
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import itertools



# Load BCC models
def load_bcc_model(path):
    model = ResNet18(num_classes=2)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model

bcc_model_1 = load_bcc_model('/home/ankur/Desktop/badd_celeba/code/real_resnet18_bias1_epoch_10.pth')
bcc_model_2 = load_bcc_model('/home/ankur/Desktop/badd_celeba/code/real_resnet18_bias2_epoch_10.pth')

# Extract features from BCC models
def extract_bcc_features(model, images):
    with torch.no_grad():
        _, features = model(images)
    return features

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

# Initialize Main Model
model = ResNet18(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

best_val_acc = 0.0

# Training function with BCC features
def train_model(num_epochs=20):
    global best_val_acc
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            # Extract BCC features
            f1 = extract_bcc_features(bcc_model_1, images)
            f2 = extract_bcc_features(bcc_model_2, images)
            
            optimizer.zero_grad()
            outputs, _ = model.concat_forward3(images, f1, f2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Perform Validation
        val_acc = evaluate_model()
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet18_with_bcc_for_tuning.pth')
            print(f"Best model saved with accuracy: {best_val_acc:.2f}%")

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(test_dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            f1 = extract_bcc_features(bcc_model_1, images)
            f2 = extract_bcc_features(bcc_model_2, images)
            outputs, _ = model.concat_forward3(images, f1, f2)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy
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



# Define alpha and beta values to test
alpha_values = [0.01]
beta_values = [0.01]

# Logging results
results = []

# Fine-tuning function with multiple alpha-beta values
def fine_tune_model(model,num_epochs=10):
    global results
    correct = 0
    total = 0
    for alpha, beta in itertools.product(alpha_values, beta_values):
        print(f"\nRunning fine-tuning with alpha={alpha}, beta={beta}")
        #model.load_state_dict(torch.load('/home/ankur/Desktop/badd_celeba/code/best_resnet18_with_bcc_for_tuning.pth'))

        # Freeze all layers except the last two
        for name, param in model.named_parameters():
            if 'fc' not in name and 'layer4' not in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)

        def modified_loss(outputs, labels, features, f1, f2):
            base_loss = criterion(outputs, labels)
            grad_outputs = torch.ones_like(outputs)
            #grads = torch.autograd.grad(outputs, features, grad_outputs=grad_outputs, create_graph=True)[0]
            grads = torch.autograd.grad(base_loss, features, create_graph=True)[0]
            proj_f1 = torch.sum(features * f1, dim=1, keepdim=True) * f1 / torch.sum(f1 ** 2, dim=1, keepdim=True)
            proj_f2 = torch.sum(features * f2, dim=1, keepdim=True) * f2 / torch.sum(f2 ** 2, dim=1, keepdim=True)
            diff_f1 = f1 - proj_f1
            diff_f2 = f2 - proj_f2
            bias_loss = alpha * (torch.sum(diff_f1 * grads))**2 + beta * (torch.sum(diff_f2 * grads))**2
            return base_loss + bias_loss
        epoch = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)    
            optimizer.zero_grad()
            f1 = extract_bcc_features(bcc_model_1, images)
            f2 = extract_bcc_features(bcc_model_2, images)
            outputs, features = model(images)
            loss = modified_loss(outputs, labels, features, f1, f2)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)

        #val_acc = evaluate_model()
        #evaluate_groupwise(model)
        #maba_base_avg, maba_base_var = compute_sba(model, test_dataloader, device)
        #print(maba_base_avg)
        
    # Save results
    with open("tuning_results.txt", "w") as f:
        for alpha, beta, acc, maba in results:
            f.write(f"alpha={alpha}, beta={beta}, Accuracy={acc:.2f}%, MABA={maba:.4f}\n")

    print("Tuning completed. Results saved in tuning_results.txt")
    torch.save(model.state_dict(), 'final_model.pth')
# Run Training and Evaluation
if __name__ == "__main__":
    print("Starting Training...")
    train_model(num_epochs=20)
    #print("\nEvaluating Best Model on Test Set...")
    #model.load_state_dict(torch.load('/home/ankur/Desktop/badd_celeba/code/best_resnet18_with_bcc_for_tuning.pth'))
    #maba_base_avg, maba_base_var = compute_sba(model, test_dataloader, device)
    #print(maba_base_avg)
    print("\nStarting Fine-tuning...")
    fine_tune_model(model,num_epochs=10)
    
