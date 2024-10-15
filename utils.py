# utils.py

import os
import numpy as np
import pickle
from data import data_loader  # Mengimpor data_loader dari data.py

# Fungsi untuk menghitung jarak Euclidean antara embeddings
def compute_distances(embedding, db_embeddings):
    distances = np.linalg.norm(db_embeddings - embedding, axis=1)
    return distances

# Fungsi untuk menyimpan embeddings dan labels ke file
def save_embeddings(db_embeddings, db_labels, db_image_paths, filename='db_embeddings.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({
            'embeddings': db_embeddings,
            'labels': db_labels,
            'image_paths': db_image_paths
        }, f)
    print(f"Embeddings saved to {filename}")

# Fungsi untuk memuat embeddings dan labels dari file
def load_embeddings(filename='db_embeddings.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Embeddings loaded from {filename}")
    return data['embeddings'], data['labels'], data['image_paths']

# Fungsi untuk menambahkan embedding baru ke database
def add_embedding(new_image_path, label, db_embeddings, db_labels, db_image_paths, base_model, load_and_preprocess_image):
    # Memuat dan memproses gambar baru
    new_image = load_and_preprocess_image(new_image_path)
    new_image = np.expand_dims(new_image, axis=0)
    
    # Menghasilkan embedding baru
    new_embedding = base_model.predict(new_image)[0]
    
    # Menambahkan embedding dan label ke database
    db_embeddings = np.vstack([db_embeddings, new_embedding])
    db_labels = np.append(db_labels, label)
    db_image_paths.append(new_image_path)
    
    print(f"Added embedding for {label} from {new_image_path}")
    return db_embeddings, db_labels, db_image_paths

# Fungsi untuk menghapus embedding dari database berdasarkan path gambar
def delete_embedding_by_image_paths(image_paths_to_delete, db_embeddings, db_labels, db_image_paths):
    # Mengumpulkan indeks dari embeddings yang akan dihapus
    indices_to_delete = [db_image_paths.index(img_path) for img_path in image_paths_to_delete]
    # Menghapus embeddings dan labels
    db_embeddings = np.delete(db_embeddings, indices_to_delete, axis=0)
    db_labels = np.delete(db_labels, indices_to_delete)
    # Menghapus image paths
    db_image_paths = [path for idx, path in enumerate(db_image_paths) if idx not in indices_to_delete]
    return db_embeddings, db_labels, db_image_paths

# Fungsi untuk mensinkronisasi embeddings database dengan folder dataset
def synchronize_embeddings(dataset_path, embeddings_file, base_model, load_and_preprocess_image):
    """
    Mensinkronisasi embeddings database dengan gambar-gambar di folder dataset.
    """
    # Memuat gambar database dari folder dataset
    current_image_paths, current_labels = data_loader(dataset_path)
    
    # Membuat mapping dari image paths ke labels
    current_image_to_label = dict(zip(current_image_paths, current_labels))
    
    if os.path.exists(embeddings_file):
        # Memuat embeddings yang sudah ada
        db_embeddings, db_labels, db_image_paths = load_embeddings(embeddings_file)
        db_image_paths = list(db_image_paths)  # Pastikan db_image_paths adalah list
        
        # Membuat set dari image paths untuk perbandingan
        current_image_set = set(current_image_paths)
        db_image_set = set(db_image_paths)
        
        # Mencari image paths yang perlu ditambahkan
        image_paths_to_add = list(current_image_set - db_image_set)
        
        # Mencari image paths yang perlu dihapus
        image_paths_to_delete = list(db_image_set - current_image_set)
        
        # Menambahkan embeddings untuk gambar baru
        if image_paths_to_add:
            print(f"Menambahkan {len(image_paths_to_add)} embeddings baru ke database.")
            for img_path in image_paths_to_add:
                label = current_image_to_label[img_path]
                db_embeddings, db_labels, db_image_paths = add_embedding(
                    img_path, label, db_embeddings, db_labels, db_image_paths, base_model, load_and_preprocess_image
                )
        
        # Menghapus embeddings untuk gambar yang dihapus
        if image_paths_to_delete:
            print(f"Menghapus {len(image_paths_to_delete)} embeddings dari database.")
            db_embeddings, db_labels, db_image_paths = delete_embedding_by_image_paths(
                image_paths_to_delete, db_embeddings, db_labels, db_image_paths
            )
        
        # Menyimpan kembali embeddings setelah sinkronisasi
        save_embeddings(db_embeddings, db_labels, db_image_paths, embeddings_file)
    else:
        # Jika embeddings belum ada, buat embeddings dari dataset
        print("Embeddings database tidak ditemukan. Membuat embeddings baru.")
        db_image_paths, db_labels = current_image_paths, current_labels
        db_embeddings = []
        for img_path in db_image_paths:
            img = load_and_preprocess_image(img_path)
            img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch
            embedding = base_model.predict(img)
            db_embeddings.append(embedding[0])
        db_embeddings = np.array(db_embeddings)
        db_labels = np.array(db_labels)
        db_image_paths = list(db_image_paths)
        # Menyimpan embeddings ke file
        save_embeddings(db_embeddings, db_labels, db_image_paths, embeddings_file)
    
    return db_embeddings, db_labels, db_image_paths
