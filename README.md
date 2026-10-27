# Generative vs. Discriminative Model Lab

A Streamlit app for handwritten digit classification and generation using MNIST.

## Features

- Upload a handwritten digit image and classify it from 0 to 9.
- View the prediction confidence and class probabilities.
- Train a variational autoencoder (VAE) and generate digit-like images.
- Compare discriminative and generative models.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The app downloads MNIST the first time it starts if the dataset is not already
available locally. The first run also trains the CNN, which can take a little
while. Training the VAE is optional and starts when you click its button.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository. Do not commit credentials or
   `.streamlit/secrets.toml`; the MNIST dataset is excluded by `.gitignore`
   and is downloaded by the app when needed.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and connect
   your GitHub account if prompted.
3. Select **Create app**, choose the repository and branch, and set the main
   file path to `app.py`.
4. Select **Deploy**. Streamlit Cloud installs packages from
   `requirements.txt` and provides a public app URL when the build completes.

The first cloud startup can take longer because it downloads MNIST and trains
the CNN. The VAE also trains on demand and may take several minutes on a
CPU-based instance.