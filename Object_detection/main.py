import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model.model_utils import Object_detection

# Expression classes
expression_classes = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Object_detection()
checkpoint = torch.load("model/best_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device).eval()

# Deterministic transform for inference (no random stuff!)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Start webcam
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

true_labels = []
pred_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_orig, w_orig = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Crop and pad the face region
        padding = 0.2
        x_pad = int(x - padding * w)
        y_pad = int(y - padding * h)
        w_pad = int(w + 2 * padding * w)
        h_pad = int(h + 2 * padding * h)

        x_pad = max(0, x_pad)
        y_pad = max(0, y_pad)
        w_pad = min(w_orig - x_pad, w_pad)
        h_pad = min(h_orig - y_pad, h_pad)

        face_crop = frame[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                expression_pred, landmark_pred, bbox_pred = output
            else:
                expression_pred, landmark_pred = output

            # Get predicted expression class
            probabilities = torch.nn.functional.softmax(expression_pred, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            expression_label = expression_classes[predicted.item()]

            # Collect predictions and ground truths
            true_labels.append(expression_label)  # You should know the ground truth in your test set
            pred_labels.append(expression_label)

            print(f"Predicted: {expression_label} with {confidence.item() * 100:.2f}% confidence")

        # Draw bounding box
        cv2.rectangle(frame, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), (255, 0, 0), 2)

        # Rescale landmarks to face_crop and map to original frame
        landmarks = landmark_pred.view(-1, 2).cpu().numpy()
        for i in range(len(landmarks)):
            landmarks[i, 0] = int(landmarks[i, 0] * (w_pad / 224.0)) + x_pad
            landmarks[i, 1] = int(landmarks[i, 1] * (h_pad / 224.0)) + y_pad

        for (lx, ly) in landmarks.astype(int):
            if 0 <= lx < w_orig and 0 <= ly < h_orig:
                cv2.circle(frame, (lx, ly), 1, (0, 255, 0), -1)

        cv2.putText(frame, f"Expression: {expression_label} ({confidence.item() * 100:.2f}%)",
                    (x_pad, y_pad - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow('Real-Time Face Landmark & Expression Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop ends, calculate the confusion matrix
conf_matrix = confusion_matrix(true_labels, pred_labels, labels=expression_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(expression_classes))
plt.xticks(tick_marks, expression_classes, rotation=45)
plt.yticks(tick_marks, expression_classes)

for i in range(len(expression_classes)):
    for j in range(len(expression_classes)):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment="center", color="red")

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

cap.release()
cv2.destroyAllWindows()
