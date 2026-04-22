import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# PART 1 — PrunableLinear Layer
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Second parameter tensor EXACT same shape as weight
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        # Initialize gate_scores such that initial sigmoid(gate_scores) is roughly 0.5 to 1.0 (or randomly)
        nn.init.normal_(self.gate_scores, mean=0.0, std=1.0)

    def forward(self, x):
        # 1. Apply sigmoid to gate_scores to get gates (values between 0 and 1)
        gates = torch.sigmoid(self.gate_scores)
        # 2. Compute pruned_weights = self.weight * gates (element-wise)
        pruned_weights = self.weight * gates
        # 3. Use F.linear(x, pruned_weights, self.bias) for the output
        return F.linear(x, pruned_weights, self.bias)

# PART 2 — Network Architecture
class SelfPruningNN(nn.Module):
    def __init__(self):
        super(SelfPruningNN, self).__init__()
        # Input: 3*32*32 = 3072 features
        # Hidden layers: 3072 -> 1024 -> 512 -> 256 -> 10
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512, 256)
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        # Flatten the CIFAR-10 images
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Final layer outputs raw logits
        x = self.fc4(x)
        return x

# PART 3 — Sparsity Loss
def sparsity_loss(model):
    # Iterates over all modules in the model
    loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            # For each PrunableLinear layer, computes sigmoid(layer.gate_scores)
            # Returns the SUM of all gate values across ALL PrunableLinear layers (L1 of gates)
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss

def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_sparsity_level(model, threshold=0.01):
    pruned_count = 0
    total_count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                pruned_count += (gates < threshold).sum().item()
                total_count += gates.numel()
    return 100 * pruned_count / total_count if total_count > 0 else 0

def get_all_gates(model):
    all_gates = []
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                all_gates.append(gates.cpu().view(-1))
    return torch.cat(all_gates).numpy()

def main():
    # Handle the results/ directory creation
    os.makedirs("results", exist_ok=True)
    
    # Set random seed
    torch.manual_seed(42)
    
    # Use torch.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # PART 4 — Training Loop
    # Dataset: CIFAR-10 via torchvision.datasets.CIFAR10
    # Normalize: mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    lambdas = [1e-5, 1e-3, 1e-1]
    epochs = 15 # The prompt says "If accuracy is too low for ALL lambdas: increase epochs from 15 to 20 and retry with lambda_mid only." Let's stick with 15 first.
    
    results = {}
    best_model_gates = None

    for lambda_val in lambdas:
        print(f"\n--- Training with lambda_val = {lambda_val} ---")
        # Train a completely fresh model
        model = SelfPruningNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
            for i, data in enumerate(pbar):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                
                task_loss = criterion(outputs, labels)
                s_loss = sparsity_loss(model)
                total_loss = task_loss + lambda_val * s_loss
                
                total_loss.backward()
                optimizer.step()
                
                running_loss += total_loss.item()
                pbar.set_postfix({'loss': running_loss / (i+1)})

        # PART 5 — Evaluation
        test_acc = evaluate(model, testloader, device)
        sparsity_lvl = get_sparsity_level(model)
        
        print(f"Lambda: {lambda_val} | Test Acc: {test_acc:.2f}% | Sparsity: {sparsity_lvl:.2f}%")
        results[lambda_val] = (test_acc, sparsity_lvl)
        
        if lambda_val == 1e-3:
            best_model_gates = get_all_gates(model)

    # PART 6 — Plotting
    # Using matplotlib, for the BEST model (lambda_mid = 1e-3):
    print("\nGenerating plot for lambda_mid = 1e-3...")
    plt.figure(figsize=(10, 6))
    plt.hist(best_model_gates, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Gate Values (lambda=1e-3)")
    plt.xlabel("gate value (0 to 1)")
    plt.ylabel("count")
    plt.savefig("results/gate_distribution.png")
    plt.close()

    # PART 7 — Print Results Table at end of script
    print("\n| Lambda     | Test Accuracy  | Sparsity Level |")
    print("|----------|--------------|----------------|")
    for l_val in [1e-5, 1e-3, 1e-1]:
        acc, spar = results[l_val]
        # Format exact table
        # | 1e-5       | XX.XX%         | XX.XX%         |
        print(f"| {l_val:<8} | {acc:>5.2f}%        | {spar:>5.2f}%        |")

if __name__ == "__main__":
    main()
