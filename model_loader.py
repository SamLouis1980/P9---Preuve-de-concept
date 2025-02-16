import os
import logging
import json
from google.cloud import storage
import torch
from transformers import Mask2FormerForUniversalSegmentation
from models_fpn import FPN_Segmenter  # Import du modèle FPN

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Forcer le chemin correct des credentials GCP
GCP_CREDENTIALS_PATH = "/app/cle_gcp.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH

# Vérification de l'existence et de l'accessibilité de la clé
if not os.path.exists(GCP_CREDENTIALS_PATH):
    logging.error(f"\u26a0\ufe0f La clé GCP n'existe pas à l'emplacement : {GCP_CREDENTIALS_PATH}")
    raise RuntimeError(f"Clé GCP introuvable : {GCP_CREDENTIALS_PATH}")

if not os.access(GCP_CREDENTIALS_PATH, os.R_OK):
    logging.error(f"\ud83d\udeab La clé GCP est présente mais illisible : {GCP_CREDENTIALS_PATH}")
    raise RuntimeError(f"Clé GCP illisible : {GCP_CREDENTIALS_PATH}")

logging.info(f"\u2705 Clé GCP trouvée et lisible à : {GCP_CREDENTIALS_PATH}")

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

def download_file(bucket_name, source_blob_name, destination_file_name):
    """Télécharge un fichier depuis Google Cloud Storage."""
    try:
        logging.debug(f"Tentative de téléchargement du fichier {source_blob_name} vers {destination_file_name}")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Fichier téléchargé avec succès : {destination_file_name}")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement de {source_blob_name} : {e}")
        raise RuntimeError(f"Impossible de télécharger {source_blob_name}")

def load_model(model_name="fpn"):
    """Charge un modèle de segmentation basé sur Torch."""
    model_path = MODEL_PATHS[model_name]
    local_model_path = os.path.join("/app", model_path)  # Force le chemin correct

    # Vérifie si le modèle est local
    if not os.path.exists(local_model_path):
        logging.info(f"Le modèle {model_name} n'est pas trouvé localement. Téléchargement en cours...")
        download_file(BUCKET_NAME, model_path, local_model_path)

        if not os.path.exists(local_model_path):
            raise RuntimeError(f"Le modèle {model_name} n'a pas été correctement téléchargé.")

        logging.info(f"Modèle {model_name} téléchargé avec succès.")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "mask2former":
            logging.info("Chargement de Mask2Former pré-entraîné...")
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic"
            ).to(device)
        else:
            logging.info("Chargement du modèle FPN...")
            model = FPN_Segmenter(num_classes=8).to(device)

        # Charger les poids fine-tunés
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        model.eval()
        logging.info(f"Modèle {model_name} chargé avec succès.")
        return model

    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_name} : {e}")
        raise RuntimeError(f"Impossible de charger le modèle {model_name}")
