from utils import load_model
from torchvision.models import densenet169
from torchvision import transforms
from PIL import Image
import torch

def predict(path_to_image: str, path_to_model: str):
    filename = path_to_model# pre-trained model path
    use_gpu = False  # load weights on the gpu
    model = densenet169(num_classes=1081) # 1081 classes in Pl@ntNet-300K
    model.eval()

    image = Image.open(path_to_image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Step 3: Make predictions
    with torch.no_grad():  # No need to track gradients
        output = model(input_batch)

    # Step 4: Interpret the results
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()


print(predict(path_to_image="./images/uknown_plant.jpeg", path_to_model="densenet201_weights_best_acc.tar"))