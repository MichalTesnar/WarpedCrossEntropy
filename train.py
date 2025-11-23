import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import SimpleMLP
from warped_cross_entropy import WarpedCrossEntropy, CustomCrossEntropyLoss
from constants import LEARNING_RATE, NUM_EPOCHS, REGIME, Regime
from dataloading import train_loader, val_loader

### SETUP ###
device = "cpu" if torch.cuda.is_available() else "cpu"
model = SimpleMLP().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if REGIME == Regime.NORMAL:
    criterion = CustomCrossEntropyLoss()
elif REGIME == Regime.WARPED:
    criterion = WarpedCrossEntropy(hidden_dimension=3, number_of_classes=10)

# Visualization setup
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
train_losses = []
val_losses = []
epochs = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = train_loss / len(train_loader)

    # --- Evaluation ---
    model.eval()
    validation_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            validation_loss += loss.item()
            
            predicted = criterion.predict(outputs)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_test_loss = validation_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Summary:')
    print(f'  Average Train Loss: {avg_train_loss:.4f}')
    print(f'  Average Validation Loss:  {avg_test_loss:.4f}')
    print(f'  Validation Accuracy:      {accuracy:.2f} %')

    # Update plot
    epochs.append(epoch + 1)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_test_loss)
    
    ax.clear()
    ax.plot(epochs, train_losses, label='Train Loss')
    ax.plot(epochs, val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Progress - {REGIME}')
    ax.legend()
    plt.pause(0.1)
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'checkpoints/model_{REGIME}_epoch_{epoch+1}.pth')
        torch.save(criterion.state_dict(), f'checkpoints/loss_{REGIME}_epoch_{epoch+1}.pth')

plt.ioff() # Turn off interactive mode
plt.savefig(f"training_{REGIME}.png")
plt.show() # Keep the window open after training finishes