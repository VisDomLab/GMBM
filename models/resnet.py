import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model=None):
        super().__init__()
        if model == None:
            model = resnet18(pretrained=pretrained)
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 512
            self.num_classes = num_classes
            self.fc = nn.Linear(self.embed_size, num_classes)
        else:
            modules = list(model.children())[:-1]
            self.extractor = nn.Sequential(*modules)
            self.embed_size = 512
            self.num_classes = num_classes
            self.fc = model.fc
        print(f"ResNet18 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        out.requires_grad_(True)
        logits = self.fc(out)

        return logits, out

    def concat_forward(self, x, f):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        out = out + f  
        logits = self.fc(out)

        return logits, out
    

    def concat_forward2(self, x, f, f2):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        out = out + ((f + f2) / 2)
        logits = self.fc(out)

        return logits, out




    def concat_forward3(self, x, f1, f2):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)

        # Compute magnitudes (L2 norms)
        norm_out = torch.norm(out, dim=1, keepdim=True)  
        norm_f1 = torch.norm(f1, dim=1, keepdim=True)    
        norm_f2 = torch.norm(f2, dim=1, keepdim=True)    

        # Compute dot products
        dot_f1 = torch.sum(out * f1, dim=1, keepdim=True)  
        dot_f2 = torch.sum(out * f2, dim=1, keepdim=True)  

        # Compute normalized dot product similarities
        normalized_dot_f1 = dot_f1 / (norm_out * norm_f1 + 1e-8)  
        normalized_dot_f2 = dot_f2 / (norm_out * norm_f2 + 1e-8)  

        # Compute softmax-based weights
        weights = F.softmax(torch.cat([normalized_dot_f1, normalized_dot_f2], dim=1), dim=1)

        # Compute weighted feature fusion
        weighted_f1 = weights[:, 0:1] * f1  
        weighted_f2 = weights[:, 1:2] * f2  

        # Update output features
        out = out + weighted_f1 + weighted_f2  

        # Final classification layer
        logits = self.fc(out)

        return logits, out
