import os
import cv2
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_utils import Object_detection
from Dataset import FER2013Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# Visualization function for results
def visualize_landmarks_tensor(image_tensor, landmarks_tensor, predicted_landmarks=None, save_path=None, show=True):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = image_tensor.cpu() * std + mean
    image = image.permute(1, 2, 0).numpy()  # (H, W, C)

    gt_landmarks = landmarks_tensor.cpu().view(-1, 2).numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='cyan', s=5, label='Ground Truth')

    if predicted_landmarks is not None:
        pred_landmarks = predicted_landmarks.cpu().view(-1, 2).numpy()
        plt.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=3, alpha=0.7, label='Prediction')

    plt.title("Facial Landmarks")
    plt.axis('off')
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📸 Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

def plot_training_results(loss_history, val_loss_history, expression_accuracy_history):
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_history, label='Training Loss', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_loss_history, label='Validation Loss', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, expression_accuracy_history, label='Validation Accuracy', color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Expression Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Training function with early stopping
def train_model(model, train_loader, val_loader, optimizer, device, epochs=30,
                alpha=15.0, beta=10.0, gamma=5.0, patience=5, start_epoch=0):
    model.train()
    loss_history = []
    val_loss_history = []
    expression_accuracy_history = []

    best_val_loss = float('inf')
    early_stop_counter = 0  # For early stopping

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        total_expression_loss = 0
        total_landmark_loss = 0
        total_bbox_loss = 0

        for images, expression_labels, landmark_labels, bbox_labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
            images, expression_labels, landmark_labels, bbox_labels = images.to(device), expression_labels.to(device), landmark_labels.to(device), bbox_labels.to(device)

            optimizer.zero_grad()

            expression_pred, landmark_pred, bbox_pred = model(images)
            loss = model.compute_loss(expression_pred, expression_labels,
                                      landmark_pred, landmark_labels,
                                      bbox_pred, bbox_labels,
                                      alpha=alpha, beta=beta, gamma=gamma)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_expression_loss += model.compute_loss(expression_pred, expression_labels, landmark_pred, landmark_labels, bbox_pred, bbox_labels, alpha=alpha, beta=0, gamma=0).item()
            total_landmark_loss += model.compute_loss(expression_pred, expression_labels, landmark_pred, landmark_labels, bbox_pred, bbox_labels, alpha=0, beta=beta, gamma=0).item()
            total_bbox_loss += model.compute_loss(expression_pred, expression_labels, landmark_pred, landmark_labels, bbox_pred, bbox_labels, alpha=0, beta=0, gamma=gamma).item()

        avg_loss = total_loss / len(train_loader)
        avg_expression_loss = total_expression_loss / len(train_loader)
        avg_landmark_loss = total_landmark_loss / len(train_loader)
        avg_bbox_loss = total_bbox_loss / len(train_loader)

        loss_history.append(avg_loss)

        val_loss = 0
        model.eval()
        correct_expression = 0
        total_expression = 0
        with torch.no_grad():
            for val_images, val_expression_labels, val_landmark_labels, val_bbox_labels in val_loader:
                val_images, val_expression_labels, val_landmark_labels, val_bbox_labels = val_images.to(device), val_expression_labels.to(device), val_landmark_labels.to(device), val_bbox_labels.to(device)

                val_expression_pred, val_landmark_pred, val_bbox_pred = model(val_images)

                val_loss += model.compute_loss(val_expression_pred, val_expression_labels,
                                               val_landmark_pred, val_landmark_labels,
                                               val_bbox_pred, val_bbox_labels,
                                               alpha=alpha, beta=beta, gamma=gamma).item()

                _, predicted = torch.max(val_expression_pred, 1)
                correct_expression += (predicted == val_expression_labels).sum().item()
                total_expression += val_expression_labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        val_expression_accuracy = 100 * correct_expression / total_expression
        expression_accuracy_history.append(val_expression_accuracy)

        print(f"📉 Validation Loss: {avg_val_loss:.4f}")
        print(f"📉 box Loss: {avg_bbox_loss:.4f}")
        print(f"📉 expression Loss: {avg_expression_loss:.4f}")
        print(f"📉 landmark Loss: {avg_landmark_loss:.4f}")
        print(f"🎯 Validation Accuracy: {val_expression_accuracy:.2f}%")

        # Early stopping check and checkpoint save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'best_model.pth')
            print(f"✅ Saved best model at epoch {epoch + 1}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"⏸️ Early stopping at epoch {epoch + 1}")
                break

    return loss_history, val_loss_history, expression_accuracy_history



# Learning rate scheduler
def get_lr_scheduler(optimizer):
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    return scheduler

# Testing function
def test_model(model, test_loader, device, alpha=15.0, beta=10.0, gamma=5.0):
    model.eval()
    correct_expression = 0
    total_expression = 0
    total_landmark_error = 0
    total_bbox_error = 0
    total_test_loss = 0
    total_expression_loss = 0
    total_landmark_loss = 0
    total_bbox_loss = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, expression_labels, landmark_labels, bbox_labels in tqdm(test_loader, desc="Testing"):
            images, expression_labels, landmark_labels, bbox_labels = images.to(device), expression_labels.to(
                device), landmark_labels.to(device), bbox_labels.to(device)

            expression_pred, landmark_pred, bbox_pred = model(images)

            loss = model.compute_loss(expression_pred, expression_labels,
                                      landmark_pred, landmark_labels,
                                      bbox_pred, bbox_labels,
                                      alpha=alpha, beta=beta, gamma=gamma)
            total_test_loss += loss.item()

            total_expression_loss += model.compute_loss(expression_pred, expression_labels, landmark_pred, landmark_labels, bbox_pred, bbox_labels, alpha=alpha, beta=0, gamma=0).item()
            total_landmark_loss += model.compute_loss(expression_pred, expression_labels, landmark_pred, landmark_labels, bbox_pred, bbox_labels, alpha=0, beta=beta, gamma=0).item()
            total_bbox_loss += model.compute_loss(expression_pred, expression_labels, landmark_pred, landmark_labels, bbox_pred, bbox_labels, alpha=0, beta=0, gamma=gamma).item()

            _, predicted = torch.max(expression_pred, 1)
            correct_expression += (predicted == expression_labels).sum().item()
            total_expression += expression_labels.size(0)

            landmark_error = torch.mean((landmark_pred - landmark_labels) ** 2)
            total_landmark_error += landmark_error.item()

            bbox_error = torch.mean((bbox_pred - bbox_labels) ** 2)
            total_bbox_error += bbox_error.item()

            all_labels.extend(expression_labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    expression_accuracy = 100 * correct_expression / total_expression
    avg_test_loss = total_test_loss / len(test_loader)
    avg_landmark_error = total_landmark_error / len(test_loader)
    avg_bbox_error = total_bbox_error / len(test_loader)

    print(f"🧪 Test Total Loss: {avg_test_loss:.4f}")
    print(f"🧪 Expression Loss: {total_expression_loss / len(test_loader):.4f}")
    print(f"🧪 Landmark Loss: {total_landmark_loss / len(test_loader):.4f}")
    print(f"🧪 Bounding Box Loss: {total_bbox_loss / len(test_loader):.4f}")
    print(f"🎯 Test Expression Accuracy: {expression_accuracy:.2f}%")
    print(f"📏 Average Landmark Error: {avg_landmark_error:.4f}")
    print(f"📦 Average Bounding Box Error: {avg_bbox_error:.4f}")

    print("\n📊 Classification Report:\n", classification_report(all_labels, all_preds))
    print("\n🧮 Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))

# Main function
def main():
    img_root = r""
    landMark_root = r""
    test_root = r""
    landMark_test_root = r""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.4, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = FER2013Dataset(img_root, landMark_root, transform=transform)
    print(f"🔍 Loaded dataset with {len(full_dataset)} samples")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Object_detection()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Load checkpoint if exists
    start_epoch = 0
    checkpoint_path = "best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔁 Loaded checkpoint from epoch {start_epoch}")

    # Train
    loss_history, val_loss_history, expression_accuracy_history = train_model(
        model, train_loader, val_loader, optimizer, device,
        epochs=30, alpha=15.0, beta=10.0, gamma=5.0, patience=5,
        start_epoch=start_epoch
    )
    plot_training_results(loss_history, val_loss_history, expression_accuracy_history)
    # Test on validation set
    test_dataset = FER2013Dataset(test_root, landMark_test_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"🧪 Loaded test dataset with {len(test_dataset)} samples")

    # Evaluate on test set
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()