# main.py

import os
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from model import get_siamese_model
from data import (
    data_loader,
    load_and_preprocess_image,
    generate_new_id,
    save_palm_vein_image
)
from utils import (
    compute_distances,
    save_embeddings,
    load_embeddings,
    add_embedding,
    synchronize_embeddings
)
import uvicorn

app = FastAPI()

# Hyperparameters
learning_rate = 0.01

# Build and load the siamese model
siamese_model, base_model = get_siamese_model(learning_rate)
siamese_model.load_weights('palm_vein_siamese_model.h5')

dataset_path = 'data/NIR'
embeddings_file = 'db_embeddings.pkl'

# Sinkronisasi embeddings database saat startup
db_embeddings, db_labels, db_image_paths = synchronize_embeddings(
    dataset_path, embeddings_file, base_model, load_and_preprocess_image
)

# Endpoint 1: Registrasi Palm Vein
@app.post("/register")
async def register_palm_vein(file: UploadFile = File(...)):
    try:
        # Validasi ekstensi file
        if file.content_type not in ['image/jpeg', 'image/png']:
            raise HTTPException(status_code=400, detail="File harus berupa gambar JPEG atau PNG.")
        
        # Generate ID baru
        new_id = generate_new_id(dataset_path)
        
        # Baca file gambar
        image_bytes = await file.read()
        
        # Simpan gambar
        image_path = save_palm_vein_image(image_bytes, new_id, dataset_path)
        
        # Sinkronisasi embeddings database
        global db_embeddings, db_labels, db_image_paths
        db_embeddings, db_labels, db_image_paths = synchronize_embeddings(
            dataset_path, embeddings_file, base_model, load_and_preprocess_image
        )
        
        return {"status": "success", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 2: Autentikasi Palm Vein
@app.post("/authenticate")
async def authenticate_palm_vein(file: UploadFile = File(...)):
    try:
        # Validasi ekstensi file
        if file.content_type not in ['image/jpeg', 'image/png']:
            raise HTTPException(status_code=400, detail="File harus berupa gambar JPEG atau PNG.")
        
        # Baca file gambar
        image_bytes = await file.read()
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        
        # Preprocessing gambar
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        # Mendapatkan embedding untuk gambar baru
        new_embedding = base_model.predict(img)
        
        # Menghitung jarak antara embedding baru dan database
        if len(db_embeddings) > 0:
            distances = compute_distances(new_embedding[0], db_embeddings)
            
            # Menentukan threshold
            threshold = 0.05  # Sesuaikan nilai ini
            
            # Mencari jarak terdekat
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            matched_label = db_labels[min_distance_index]
            
            if min_distance < threshold:
                return {
                    "status": "success",
                    "message": "Palm vein terautentikasi.",
                    "id": matched_label,
                    "distance": float(min_distance)
                }
            else:
                return {
                    "status": "fail",
                    "message": "Palm vein tidak dikenali.",
                    "distance": float(min_distance)
                }
        else:
            return {
                "status": "fail",
                "message": "Database embeddings kosong."
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint 3: Menghapus Palm Vein
@app.delete("/delete/{user_id}")
async def delete_palm_vein(user_id: str):
    try:
        # Path folder user
        user_folder = os.path.join(dataset_path, user_id)
        
        # Cek apakah folder user ada
        if not os.path.exists(user_folder):
            raise HTTPException(status_code=404, detail="User ID tidak ditemukan.")
        
        # Hapus folder user
        for root, dirs, files in os.walk(user_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)
        
        # Sinkronisasi embeddings database
        global db_embeddings, db_labels, db_image_paths
        db_embeddings, db_labels, db_image_paths = synchronize_embeddings(
            dataset_path, embeddings_file, base_model, load_and_preprocess_image
        )
        
        return {"status": "success", "message": f"User ID {user_id} berhasil dihapus."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
