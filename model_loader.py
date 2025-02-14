import os
import logging
import json
from google.cloud import storage
import torch
import torchvision
from transformers import Mask2FormerForUniversalSegmentation
from models_fpn import FPN_Segmenter  # Import du modèle FPN

# Configuration du logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Désactiver CUDA pour forcer le CPU si aucun GPU n'est disponible
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Chemin de la clé GCP
GCP_CREDENTIALS_PATH = "/tmp/gcp_key.json"

def load_gcp_credentials():
    """Charge la clé GCP uniquement si elle est nécessaire pour télécharger les modèles."""
    try:
        credentials_json = None

        # Vérifier si une clé existe déjà dans l'environnement
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            logging.info("Clé GCP déjà configurée dans GOOGLE_APPLICATION_CREDENTIALS.")
            return  # Rien à faire, la clé est déjà définie

        # Vérifier si une clé est disponible via une variable d'environnement
        if "GCP_CREDENTIALS" in os.environ:
            credentials_json = os.environ["GCP_CREDENTIALS"]
            logging.info("Clé GCP détectée dans les variables d'environnement.")

        if credentials_json:
            credentials_dict = json.loads(credentials_json) if isinstance(credentials_json, str) else credentials_json

            # Sauvegarder la clé dans un fichier temporaire
            with open(GCP_CREDENTIALS_PATH, "w") as f:
                json.dump(credentials_dict, f)

            # Définir la variable d’environnement pour Google Cloud
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
            logging.info("Clé GCP enregistrée et utilisée dans /tmp/gcp_key.json")
        else:
            logging.warning("Aucune clé GCP trouvée. L'API pourra fonctionner uniquement si la clé est déjà configurée.")
    
    except json.JSONDecodeError:
        logging.error("Erreur de décodage JSON dans GCP_CREDENTIALS.")
        raise RuntimeError("Erreur de décodage JSON dans GCP_CREDENTIALS.")
    except Exception as e:
        logging.error(f"Impossible de charger la clé GCP : {e}")
        raise RuntimeError("Erreur de configuration GCP.")

# Charger les credentials GCP si nécessaire
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

def download_file(bucket_name, source_blob_name, destination_file_name):
    """Télécharge un fichier depuis Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Fichier téléchargé : {destination_file_name}")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement de {source_blob_name} : {e}")
        raise RuntimeError(f"Impossible de télécharger {source_blob_name}")

def load_model(model_name="fpn"):
    """Charge un modèle de segmentation basé sur Torch."""
    model_path = MODEL_PATHS[model_name]
    local_model_path = os.path.join(os.getcwd(), model_path)

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
