import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets.celeba_experimental import get_dataloaders
from tqdm import tqdm
from scipy.stats import chi2_contingency
from scipy.special import kl_div

def extract_biases(loader):
    all_bias1 = []
    all_bias2 = []
    for idx, (images, labels, biases, biases2) in enumerate(tqdm(loader)):
        all_bias1.append(biases.cpu().numpy())
        all_bias2.append(biases2.cpu().numpy())
    all_bias1 = np.concatenate(all_bias1)
    all_bias2 = np.concatenate(all_bias2)
    return all_bias1, all_bias2

def compute_joint_distribution(bias1, bias2):
    joint_dist = np.zeros((2, 2))
    for b1, b2 in zip(bias1, bias2):
        joint_dist[int(b1), int(b2)] += 1
    return joint_dist

def plot_joint_distribution(train_prob, test_prob):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Training heatmap
    im0 = ax[0].imshow(train_prob, cmap="Blues", interpolation="nearest")
    ax[0].set_title("Train Joint Distribution\n(Wearing_Lipstick vs Heavy_Makeup)")
    ax[0].set_xlabel("Heavy_Makeup")
    ax[0].set_ylabel("Wearing_Lipstick")
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(['0', '1'])
    ax[0].set_yticklabels(['0', '1'])
    # Add annotations
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f'{train_prob[i,j]:.3f}', ha='center', va='center', color='black')
    fig.colorbar(im0, ax=ax[0])
    
    # Test heatmap
    im1 = ax[1].imshow(test_prob, cmap="Reds", interpolation="nearest")
    ax[1].set_title("Test Joint Distribution\n(Wearing_Lipstick vs Heavy_Makeup)")
    ax[1].set_xlabel("Heavy_Makeup")
    ax[1].set_ylabel("Wearing_Lipstick")
    ax[1].set_xticks([0, 1])
    ax[1].set_yticks([0, 1])
    ax[1].set_xticklabels(['0', '1'])
    ax[1].set_yticklabels(['0', '1'])
    # Add annotations
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f'{test_prob[i,j]:.3f}', ha='center', va='center', color='black')
    fig.colorbar(im1, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig('joint_distributions.png', dpi=300)

def compute_kl_divergence(p, q):
    p = p.flatten()
    q = q.flatten()
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(kl_div(p, q))

# Main execution
if __name__ == "__main__":
    # Paths to CelebA data
    data_dir = "/home/ankur/Desktop/BAdd_Bias_Mitigation/code/data/celeba/img_align_celeba"
    csv_dir = "/home/ankur/Desktop/BAdd_Bias_Mitigation/code/data/celeba/list_attr_celeba.txt"

    # Get data loaders with balanced test set
    train_loader, test_loader = get_dataloaders(
        data_dir,
        csv_dir,
        precrop=256,
        crop=224,
        bs=64,
        nw=4,
        split=0.7,
    )

    # Extract biases
    train_bias1, train_bias2 = extract_biases(train_loader)
    test_bias1, test_bias2 = extract_biases(test_loader)

    # Compute joint distributions
    train_joint_dist = compute_joint_distribution(train_bias1, train_bias2)
    test_joint_dist = compute_joint_distribution(test_bias1, test_bias2)

    # Normalize to probabilities
    train_joint_prob = train_joint_dist / np.sum(train_joint_dist)
    test_joint_prob = test_joint_dist / np.sum(test_joint_dist)

    # Print distributions
    print("Train Joint Distribution (Wearing_Lipstick vs Heavy_Makeup):")
    print(train_joint_prob)
    print("Test Joint Distribution (Wearing_Lipstick vs Heavy_Makeup):")
    print(test_joint_prob)

    # Visualize distributions
    plot_joint_distribution(train_joint_prob, test_joint_prob)

    # Chi-Square Test
    chi2_stat, p_val, dof, expected = chi2_contingency([train_joint_dist.flatten(), test_joint_dist.flatten()])
    print(f"Chi-Square Test Statistic: {chi2_stat}")
    print(f"P-Value: {p_val}")

    # KL Divergence
    kl_divergence_train_to_test = compute_kl_divergence(train_joint_prob, test_joint_prob)
    kl_divergence_test_to_train = compute_kl_divergence(test_joint_prob, train_joint_prob)
    print(f"KL Divergence (Train -> Test): {kl_divergence_train_to_test}")
    print(f"KL Divergence (Test -> Train): {kl_divergence_test_to_train}")
