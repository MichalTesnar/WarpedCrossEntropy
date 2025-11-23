import torch
from model import SimpleMLP
from warped_cross_entropy import WarpedCrossEntropy, CustomCrossEntropyLoss
from constants import REGIME, Regime
from dataloading import test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NORMAL
# MODEL_CHECKPOINT_FILE = f'checkpoints/model_{REGIME}_epoch_5.pth'
# LOSS_CHECKPOINT_FILE = f'checkpoints/loss_{REGIME}_epoch_5.pth'

# WARPED JOINT & WARPED
MODEL_CHECKPOINT_FILE = f'checkpoints/model_{REGIME}_epoch_50.pth'
LOSS_CHECKPOINT_FILE = f'checkpoints/loss_{REGIME}_epoch_50.pth'

model = SimpleMLP().to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT_FILE))

if REGIME == Regime.NORMAL:
    criterion = CustomCrossEntropyLoss()
elif REGIME == Regime.WARPED:
    criterion = WarpedCrossEntropy(hidden_dimension=3, number_of_classes=10)
    criterion.load_state_dict(torch.load(LOSS_CHECKPOINT_FILE))

# Evaluation
model.eval()
test_loss = 0.0
correct = 0
total = 0

print(f"Starting testing on regime {REGIME}")
with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Get predictions
        predicted = criterion.predict(outputs)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

avg_test_loss = test_loss / len(test_loader)
accuracy = 100 * correct / total

print(f'Test Results:')
print(f'  Average Test Loss: {avg_test_loss:.4f}')
print(f'  Test Accuracy:     {accuracy:.2f} %')
