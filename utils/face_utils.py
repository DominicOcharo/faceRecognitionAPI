import os
import pickle
import numpy as np

DATA_PATH = "./data"

def save_face_data(name: str, embeddings: np.ndarray):
    """Save the face embeddings and associated name."""
    # Save names
    if 'names.pkl' not in os.listdir(DATA_PATH):
        names = [name] * len(embeddings)
        with open(f'{DATA_PATH}/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(f'{DATA_PATH}/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [name] * len(embeddings)
        with open(f'{DATA_PATH}/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save embeddings
    if 'face_embeddings.pkl' not in os.listdir(DATA_PATH):
        with open(f'{DATA_PATH}/face_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings, f)
    else:
        with open(f'{DATA_PATH}/face_embeddings.pkl', 'rb') as f:
            face_embeddings = pickle.load(f)
        face_embeddings = np.append(face_embeddings, embeddings, axis=0)
        with open(f'{DATA_PATH}/face_embeddings.pkl', 'wb') as f:
            pickle.dump(face_embeddings, f)

def load_face_data(data_path: str):
    """Load face embeddings and labels."""
    if 'names.pkl' not in os.listdir(data_path) or 'face_embeddings.pkl' not in os.listdir(data_path):
        raise FileNotFoundError("No registered users. Please register first.")

    with open(f'{data_path}/names.pkl', 'rb') as f:
        labels = pickle.load(f)

    with open(f'{data_path}/face_embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)

    return labels, embeddings
