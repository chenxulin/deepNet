# DATA PREPARATION
# A call is made to GitHub via Python's requests module to download a .zip file and unzip it.
import os
import requests
import zipfile
import pathlib

# Setup path to data folder
data_path = pathlib.Path("./data")
image_path = data_path / "pizza_steak_sushi"

# if the image folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f"{image_path} derectory already exists.")
else:
    print(f"Did not find {image_path} directory, createing one...")
    image_path.mkdir(parents = True, exist_ok = True)

# Download pizza, steak, sushi data(.zip) from GitHub
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    print("Downloading pizza, steak, sushi data...")
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    request = requests.get(url)
    f.write(request.content)
    print("Download complete.")

# unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...")
    zip_ref.extractall(image_path)
    print("Unzipping complete.")

# remove the zip file after unzipping
os.remove(data_path / "pizza_steak_sushi.zip")
