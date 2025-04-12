import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_utils import Object_detection
from train import FER2013Dataset

import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import numpy as np

def visualize_landmarks(image, landmarks, predicted_landmarks=None):
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToPILImage

    to_pil = ToPILImage()
    image = to_pil(image.cpu())

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')

    # Ground truth landmarks (should be in [0, 224] range)
    landmarks = landmarks.view(-1, 2).cpu().numpy()
    plt.scatter(landmarks[:, 0], landmarks[:, 1], color='blue', label='Ground Truth', s=40, alpha=0.6)

    # Predicted landmarks
    if predicted_landmarks is not None:
        predicted_landmarks = predicted_landmarks.view(-1, 2).cpu().numpy()
        plt.scatter(predicted_landmarks[:, 0], predicted_landmarks[:, 1], color='red', label='Predicted', s=40, alpha=0.6, marker='x')

    plt.legend(loc='upper left')
    plt.show()





def train_model(model, train_loader, optimizer, device, epochs, alpha=1.0, beta=1.0, save_path="best_model.pth"):
    model.train()
    loss_history = []
    best_loss = float('inf')

    # Scheduler to reduce learning rate when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(epochs):
        total_loss = 0

        for images,  expression_labels, landmark_labels in tqdm(train_loader, desc="Training"):
            # Move data to the correct device (either CPU or GPU)
            images, expression_labels, landmark_labels = images.to(device), expression_labels.to(device), landmark_labels.to(device)

            optimizer.zero_grad()
            expression_pred, landmark_pred = model(images)

            # Calculate the loss
            loss = model.compute_loss(expression_pred, expression_labels,
                                      landmark_pred, landmark_labels,
                                      expression_weights=None, alpha=alpha, beta=beta)
            # If loss is NaN, stop the training
            if torch.isnan(loss):
                print("Loss is NaN, stopping training.")
                break
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        for param_group in optimizer.param_groups:
            print(f"🔧 Learning rate: {param_group['lr']}")

        # Update scheduler
        scheduler.step(avg_loss)

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"✅ New best model saved at epoch {epoch+1} with loss {avg_loss:.4f}")

    # Plot the loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', linestyle='-')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()



def test_model(model, test_loader, device):
    model.eval()
    correct_expression = 0
    total_expression = 0
    total_landmark_error = 0  # Add tracking for landmark error

    with torch.no_grad():
        for images, expression_labels, landmark_labels in tqdm(test_loader, desc="Testing"):
            images, expression_labels, landmark_labels = images.to(device), expression_labels.to(device), landmark_labels.to(device)
            expression_pred, landmark_pred = model(images)

            # Expression accuracy
            _, predicted = torch.max(expression_pred, 1)
            total_expression += expression_labels.size(0)
            correct_expression += (predicted == expression_labels).sum().item()

            # Landmark error (using Mean Squared Error for simplicity)
            landmark_error = torch.mean((landmark_pred - landmark_labels) ** 2)
            total_landmark_error += landmark_error.item()

            # Visualize the first image and landmarks in the batch
            if len(images) > 0:
                visualize_landmarks(images[0], landmark_labels[0], predicted_landmarks=landmark_pred[0])

    expression_accuracy = 100 * correct_expression / total_expression
    print(f"🎯 Test Expression Accuracy: {expression_accuracy:.2f}%")
    print(f"📏 Average Landmark Error: {total_landmark_error / len(test_loader):.4f}")



def main():
    # === Path setup ===
    img_root = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\Train"
    landMark_root = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\Train_csv\landmarks_with_labels.csv"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # === Dataset and loaders ===
    dataset = FER2013Dataset(img_root, landMark_root, transform=transform)  # Ensure this loads both tasks
    print(f"🔍 Loaded dataset with {len(dataset)} samples")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Object_detection().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # Checkpoint loading
    if os.path.exists("best_model.pth"):
        checkpoint = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Loaded best model checkpoint. Resuming from epoch {start_epoch} with new learning rate")
    else:
        print("🆕 No checkpoint found, starting from scratch.")
        start_epoch = 0

    # === Train ===
    train_model(model,train_loader,optimizer,device,epochs=150)
    
    # === Test ===
    test_expr_dir = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\Test"
    test_landMark_root = r"C:\Users\miner\OneDrive\Desktop\introtocs\Python_Project\training\archive_\Train_csv\landmarks_with_labels_Test.csv"

    test_dataset = FER2013Dataset(test_expr_dir, test_landMark_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()
