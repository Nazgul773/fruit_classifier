import os

import torch
import torch.nn as nn
from utils.preprocessors import preprocess
from utils.models import PretrainedFruitVeggieClassifier

# Hyperparameter
NUM_CLASSES = 36
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001


# Option to build the model from scratch or load a saved model
def build_or_load_model(model):
    path = model.model_path
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    start_epoch = 0

    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w'):
            pass
        print("Model built from scratch.")

    return model, optimizer, scheduler, start_epoch


def train_and_validate(train_loader, val_loader, model, optimizer,
                       scheduler, start_epoch, num_epochs, patience):
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    save_path = model.model_path

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{start_epoch + num_epochs}], Train Loss: {avg_train_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        scheduler.step(avg_val_loss)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': running_loss,
            }, save_path)
            print(f'Model weights updated at {save_path}')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f'Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}.')
                break


def is_cuda_available(device):
    if device == torch.device("cuda"):
        device_name = torch.cuda.get_device_name(0)
        print(f'CUDA is available - Using {device_name}.')
    else:
        print(f'CUDA is not available - Using {device}.')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda_available(device)

init_model = PretrainedFruitVeggieClassifier(num_classes=NUM_CLASSES).to(device)

model_type = init_model.model_type
train_loader, val_loader, benchmark_loader = preprocess(model_type=model_type)

net, optimizer, scheduler, start_epoch = build_or_load_model(model=init_model)

train_and_validate(train_loader=train_loader,
                   val_loader=val_loader,
                   model=net,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   start_epoch=start_epoch,
                   num_epochs=start_epoch + NUM_EPOCHS,
                   patience=5)
