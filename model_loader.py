from google.cloud import storage
import os
import logging
import json
from PIL import Image
import streamlit as st
import torch
import torchvision
from transformers import Mask2FormerForUniversalSegmentation
from models_fpn import FPN_Segmenter  # Import du modèle FPN

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Désactiver CUDA pour forcer le CPU si aucun GPU n'est disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Gestion de la clé GCP pour authentification
GCP_CREDENTIALS_PATH = "/tmp/gcp_key.json"

def load_gcp_credentials():
    """Charge la clé GCP depuis Streamlit Secrets ou les variables d'environnement."""
    try:
        credentials_json = None

        # Cas 0: Exécution sur Colab (si la variable d'environnement GOOGLE_COLAB est présente)
        if "COLAB_GPU" in os.environ:
            logging.info("🔹 Exécution détectée sur Google Colab.")
            
            # Vérifier si la clé GCP est présente dans un fichier local
            local_gcp_path = "/content/gcp_key.json"
            if os.path.exists(local_gcp_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_gcp_path
                logging.info(f"Clé GCP chargée depuis {local_gcp_path}.")
                return  # On s'arrête ici, pas besoin de chercher ailleurs.
            else:
                raise RuntimeError("🚨 Clé GCP introuvable sur Colab ! Ajoutez `/content/gcp_key.json`.")
                
        # Cas 1: Streamlit Secrets
        if "GCP_CREDENTIALS" in st.secrets:
            credentials_json = st.secrets["GCP_CREDENTIALS"]
            logging.info("Clé GCP détectée dans Streamlit Secrets.")

        # Cas 2: Variable d'environnement (Google Cloud Run)
        elif "GCP_CREDENTIALS" in os.environ:
            credentials_json = os.environ["GCP_CREDENTIALS"]
            logging.info("Clé GCP détectée dans les variables d'environnement.")

        if credentials_json:
            credentials_dict = json.loads(credentials_json) if isinstance(credentials_json, str) else credentials_json
            
            # Sauvegarder la clé dans un fichier temporaire
            with open(GCP_CREDENTIALS_PATH, "w") as f:
                json.dump(credentials_dict, f)

            # Définir la variable d’environnement
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
            logging.info("Clé GCP correctement écrite dans /tmp/gcp_key.json")
        else:
            raise RuntimeError("Aucune clé GCP trouvée dans Streamlit Secrets ni les variables d'environnement.")
    except json.JSONDecodeError:
        logging.error("Erreur de décodage JSON dans GCP_CREDENTIALS.")
        raise RuntimeError("Erreur de décodage JSON dans les secrets GCP.")
    except Exception as e:
        logging.error(f"Impossible de charger la clé GCP : {e}")
        raise RuntimeError("Erreur de configuration GCP.")

# Charger les credentials GCP
load_gcp_credentials()

# Configuration du bucket Google Cloud
BUCKET_NAME = "p8_segmentation_models"
MODEL_PATHS = {
    "fpn": "fpn_best.pth",
    "mask2former": "mask2former_best.pth"
}

MODEL_INPUT_SIZES = {
    "fpn": (512, 512),
    "mask2former": (512, 512)
}

# 🎨 Palette de couleurs pour affichage (Cityscapes)
GROUP_PALETTE = [
    (0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153),
    (107, 142, 35), (70, 130, 180), (220, 20, 60), (0, 0, 142)
]

def apply_cityscapes_palette(group_mask):
    pil_mask = Image.fromarray(group_mask.astype('uint8'))
    flat_palette = [value for color in GROUP_PALETTE for value in color]
    pil_mask.putpalette(flat_palette)
    return pil_mask.convert("RGB")

def list_images():
    """Liste toutes les images disponibles dans le bucket et affiche un log détaillé."""
    try:
        logging.info("Connexion à Google Cloud Storage...")
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix="images/RGB/")

        image_files = [blob.name.split("/")[-1] for blob in blobs if blob.name.endswith(".png")]
        
        if not image_files:
            logging.warning("Aucune image trouvée dans le dossier RGB du bucket.")
        else:
            logging.info(f"Images trouvées : {image_files}")

        return image_files
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des images : {e}")
        return []

def download_file(bucket_name, source_blob_name, destination_file_name):
    """Télécharge un fichier depuis GCP."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Fichier téléchargé : {destination_file_name}")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement de {source_blob_name} : {e}")
        raise RuntimeError(f"Impossible de télécharger le fichier {source_blob_name}")

def load_model(model_name="fpn"):
    """Charge un modèle de segmentation basé sur Torch."""
    model_path = MODEL_PATHS[model_name]
    local_model_path = os.path.join(os.getcwd(), model_path)

    # Vérifie si le modèle est local
    if not os.path.exists(local_model_path):
        logging.info(f"Le modèle {model_name} n'est pas trouvé localement. Tentative de téléchargement...")
        download_file(BUCKET_NAME, model_path, local_model_path)
        
        if not os.path.exists(local_model_path):
            raise RuntimeError(f"Le modèle {model_name} n'a pas été correctement téléchargé ou est introuvable.")

        logging.info(f"Modèle {model_name} téléchargé avec succès.")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "mask2former":
            model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco").to(device)
            model.load_state_dict(torch.load(local_model_path, map_location=device))
        else:
            model = FPN_Segmenter(num_classes=8).to(device)
            model.load_state_dict(torch.load(local_model_path, map_location=device))

        model.eval()  # Mode évaluation
        logging.info(f"Modèle {model_name} chargé avec succès.")
        return model

    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_name} : {e}")
        raise RuntimeError(f"Impossible de charger le modèle {model_name}")
