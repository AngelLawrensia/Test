import streamlit as st
import torch
from PIL import Image
from model_dummy import DummyUNet3D

model = DummyUNet3D()
model.eval()

st.title("Demo Integrasi Model (Dummy 3D UNet)")

file = st.file_uploader("Upload gambar MRI atau apa aja", type=["png","jpg","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Input")

    # Pretend preprocessing
    tensor = torch.randn(1, 1, 128, 128, 128)  # fake 3D volume
    output = model(tensor)

    st.success("Model terhubung!")
    st.json(output if isinstance(output, dict) else str(output))
