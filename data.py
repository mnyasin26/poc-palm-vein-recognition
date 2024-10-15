# data.py

import os
import numpy as np
import cv2

# Hyperparameters
img_width, img_height = 128, 128

# Fungsi untuk memuat dan resize gambar
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Membaca gambar grayscale
    img = cv2.resize(img, (img_width, img_height))  # Resize gambar
    img = img.astype('float32') / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=-1)  # Menambahkan dimensi channel
    return img

# Fungsi untuk memuat data dan label
def data_loader(dataset_path):
    image_paths = []
    labels = []
    
    for class_dir in os.listdir(dataset_path):
        class_dir_path = os.path.join(dataset_path, class_dir)
        if os.path.isdir(class_dir_path):
            for img_name in os.listdir(class_dir_path):
                img_path = os.path.join(class_dir_path, img_name)
                image_paths.append(img_path)
                labels.append(class_dir)
                    
    return np.array(image_paths), np.array(labels)

# Fungsi untuk memuat dan memproses gambar baru
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Membaca gambar dalam grayscale
    img = cv2.resize(img, (img_width, img_height))      # Resize gambar
    img = img.astype('float32') / 255.0                 # Normalisasi
    img = np.expand_dims(img, axis=-1)                  # Menambahkan dimensi channel
    return img

# Fungsi untuk menghasilkan ID baru
def generate_new_id(dataset_path):
    existing_ids = [int(folder) for folder in os.listdir(dataset_path) if folder.isdigit()]
    if existing_ids:
        new_id = max(existing_ids) + 1
    else:
        new_id = 1
    return f"{new_id:04d}"

# Fungsi untuk menyimpan gambar palm vein yang diunggah
def save_palm_vein_image(image_bytes, user_id, dataset_path):
    # Buat folder pengguna jika belum ada
    user_folder = os.path.join(dataset_path, user_id)
    os.makedirs(user_folder, exist_ok=True)

    # Tentukan nama file baru
    existing_files = os.listdir(user_folder)
    new_filename = f"{len(existing_files) + 1}_01_s.jpg"
    image_path = os.path.join(user_folder, new_filename)

    # Simpan gambar
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(image_path, img)

    return image_path
