import kagglehub
import shutil
import os


# Download latest version
path = kagglehub.dataset_download("cholling/game-of-hex")
data_path = os.path.join('data')
shutil.move(path, data_path)

