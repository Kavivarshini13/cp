
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter


st.set_page_config(page_title="Digit AI Lab", layout="wide")
st.title("Generative vs. Discriminative Model Lab")
st.write(
    "Explore handwritten digit classification and "
    "image generation using deep learning."
)

torch.manual_seed(42)
DEVICE = torch.device("cpu")

# ---------------- CNN CLASSIFIER ----------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- GENERATIVE MODEL: VAE ----------------
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, 16)
        self.logvar = nn.Linear(256, 16)
        self.decoder = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        reconstruction = self.decoder(z).view(-1, 1, 28, 28)
        return reconstruction, mu, logvar


# ---------------- MNIST DATA ----------------
@st.cache_resource
def get_data():
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    return train_data


# ---------------- TRAIN CNN ----------------
@st.cache_resource
def train_cnn():
    dataset = get_data()
    subset = torch.utils.data.Subset(dataset, range(20000))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=128, shuffle=True
    )

    model = CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(2):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# ---------------- TRAIN VAE ----------------
@st.cache_resource
def train_vae():
    dataset = get_data()
    subset = torch.utils.data.Subset(dataset, range(10000))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=128, shuffle=True
    )

    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for images, _ in loader:
            images = images.to(DEVICE)
            reconstruction, mu, logvar = model(images)

            reconstruction_loss = F.binary_cross_entropy(
                reconstruction, images, reduction="sum"
            )
            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            loss = reconstruction_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


# ---------------- LOAD MODELS ----------------
with st.spinner("Preparing MNIST and training the CNN..."):
    classifier = train_cnn()

st.success("CNN classifier is ready!")

# ---------------- IMAGE CLASSIFICATION ----------------
st.header("1. Upload a handwritten digit")
st.write(
    "Upload a single digit image, preferably a black background "
    "with a white digit, similar to MNIST."
)

uploaded = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    original = Image.open(uploaded).convert("L")
    st.image(original, caption="Uploaded image", width=180)

    # Prepare image for MNIST-style input.
    image = ImageOps.autocontrast(original)
    image = ImageOps.fit(image, (28, 28))

    # Convert dark-background/bright-digit images to MNIST style.
    if sum(image.getpixel((x, y)) for x in range(28)
           for y in range(28)) / 784 > 127:
        image = ImageOps.invert(image)

    tensor = transforms.ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(
            classifier(tensor), dim=1
        )[0]
        prediction = int(probabilities.argmax())
        confidence = float(probabilities[prediction]) * 100

    col1, col2 = st.columns(2)
    col1.metric("Predicted digit", prediction)
    col2.metric("Confidence score", f"{confidence:.2f}%")

    st.caption(
        "Confidence is the model's softmax score, not a guarantee "
        "that the prediction is correct."
    )

    st.subheader("Probability for each digit")
    st.bar_chart({
        str(i): float(probabilities[i])
        for i in range(10)
    })

# ---------------- GENERATE NEW DIGITS ----------------
st.header("2. Generative Model — VAE")
st.write(
    "The VAE learns patterns in MNIST and generates new "
    "digit-like images from random latent vectors."
)

if st.button("Train VAE and Generate 10 Digits"):
    with st.spinner("Training VAE. This may take several minutes..."):
        generator = train_vae()

    with torch.no_grad():
        z = torch.randn(10, 16)
        samples = generator.decoder(z).reshape(10, 28, 28)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Generated {i + 1}")
        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.success("Ten new samples generated by the VAE!")

# ---------------- MODEL COMPARISON ----------------
st.header("3. Model Comparison")

st.markdown("""
| Feature | CNN Classifier | VAE Generator |
|---|---|---|
| Model type | Discriminative | Generative |
| Input | Handwritten digit image | Random latent vector for generation |
| Output | Digit label and confidence | New digit-like images |
| Main purpose | Classification | Data generation |
""")


st.info(
    "Both models are trained locally on MNIST. The CNN predicts "
    "labels; the VAE learns a latent representation and generates "
    "new images. This demo trains small models for demonstration, "
    "so prediction accuracy and generated image quality may be limited."
)
