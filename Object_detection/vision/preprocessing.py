
import torchvision.transforms as transforms

# preproccess the image input image so every image has the same size so it will be easier for the model to understand
def preprocess_iamge(image):
    resized_image = transforms.Resize((416,416))
    resized_image = transforms.ToTensor()
    resized_image = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])


