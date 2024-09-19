import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

Image_Width = 64
Image_Height = 64

mean = torch.tensor([0.4698, 0.4374, 0.4010])
std = torch.tensor([0.2640, 0.2552, 0.2551])

itol = {
    0: 'Gato',
    1: 'Perro'
}

class Miau(nn.Module):
    def __init__(self):
        super(Miau, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (Image_Width // 8) * (Image_Height // 8), 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop3(x)

        #print(x.shape)
        
        x = self.flatten(x)
        #print(x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop4(x)
        x = F.softmax(self.fc2(x), dim=1)
        
        return x

model = Miau()

model = torch.load('./miauNET.pth', map_location=torch.device('cpu'))

trans = transforms.ToTensor()
resize = transforms.Resize((64, 64))
norm = transforms.Normalize(mean = mean, std = std)




# Título de la aplicación
st.title("Clasificador de Gatos y Perros")

# Captura de imagen desde la webcam
img_file_buffer = st.camera_input("Clasifica")


if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    #img = './data/training_set/training_set/dogs/dog.13.jpg'
    #img = Image.open(img)


    #aplicar transforms
    drawing_tensor = trans(img)
    drawing_tensor = resize(drawing_tensor)
    norm = norm(drawing_tensor)

    image = transforms.functional.to_pil_image(norm, mode=None)

    #st.text(drawing_tensor.shape)
    norm = norm.unsqueeze(0)
    result = model(norm)
    #st.text(drawing_tensor)
    st.text(result)
    st.text(f"Creo que estoy viendo a un {itol[result.argmax().item()]}")

    # Muestra la imagen capturada
    #st.image(img, caption="Imagen Capturada", use_column_width=True)
    #st.image(image, caption="Imagen Capturada", use_column_width=True)

    # Opcional: Procesar la imagen (puedes añadir tu propio código aquí)
    # Por ejemplo, convertir a escala de grises
    #img_gray = img.convert("L")
    #st.image(img_gray, caption="Imagen en Escala de Grises", use_column_width=True)
