import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.resnet import ResNet18
from data.coco_dataloader import create_dataloader
from tqdm import tqdm
from sba_metric_coco import compute_sba
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Test Dataloader
print("Loading validation data...")
test_dataloader = create_dataloader(
    image_dir='/home/ankur/Desktop/badd_celeba/code/data/coco/val2017',
    captions_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/captions_val2017.json',
    instances_path='/home/ankur/Desktop/badd_celeba/code/data/coco/annotations/instances_val2017.json',
    shuffle=False
)

# Load Model
model = ResNet18(num_classes=2).to(device)
model.load_state_dict(torch.load('/home/ankur/Desktop/badd_celeba/code/best_fine_tuned_resnet18_with_bcc.pth'))
model.eval()

# Evaluation function with group-wise bias accuracy
def evaluate_groupwise():
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

    print("-------------------------------------")
    print("-------------------------------------")
    sba_base_avg, maba_base_var = compute_sba(model, test_dataloader, device)
    print(sba_base_avg)
    

# Evaluate
evaluate_groupwise()
