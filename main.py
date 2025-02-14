from fastapi import FastAPI, File, UploadFile, Response, Query
import numpy as np
import cv2
import torch
from model_loader import load_model, MODEL_PATHS, MODEL_INPUT_SIZES
from io import BytesIO
from PIL import Image
import uvicorn
import os
import logging
from utils import preprocess_image, resize_and_colorize_mask

# Configuration du logging pour afficher les logs DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fastapi_logs.log"),  # Enregistrer dans un fichier
        logging.StreamHandler()  # Afficher dans la console
    ]
)

logging.debug("Logging DEBUG activé dans FastAPI !")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = FastAPI()

# Liste des modèles disponibles
AVAILABLE_MODELS = list(MODEL_PATHS.keys())

# Palette officielle Cityscapes
CLASS_COLORS = {
    0: (0, 0, 0),        # Void
    1: (128, 64, 128),   # Flat
    2: (70, 70, 70),     # Construction
    3: (153, 153, 153),  # Object
    4: (107, 142, 35),   # Nature
    5: (70, 130, 180),   # Sky
    6: (220, 20, 60),    # Human
    7: (0, 0, 142)       # Vehicle
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...), model_name: str = Query("fpn", enum=AVAILABLE_MODELS)):
    logging.debug(f"Content type reçu : {file.content_type}")
    """Endpoint qui prend une image en entrée, applique la segmentation et retourne le masque colorisé."""
    logging.debug(f"Requête reçue avec modèle : {model_name}")

    # Vérification du format et de la taille
    if file.content_type not in ["image/jpeg", "image/png"]:
        logging.error("Format non supporté reçu !")
        return {"error": "Format non supporté. Utilisez JPEG ou PNG."}
    
    if file.size > 10 * 1024 * 1024:
        logging.error("Image trop grande (>10MB) reçue !")
        return {"error": "Image trop grande. Taille max: 10MB."}

    # Lire l'image reçue
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            logging.error("Fichier image vide reçu !")
            return {"error": "Fichier image vide"}
        
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        logging.error(f"Impossible de lire l'image reçue : {e}")
        return {"error": "Format d'image non supporté"}

    # Prétraitement de l'image
    input_size = MODEL_INPUT_SIZES[model_name]
    image_array, original_size = preprocess_image(image, input_size)
    logging.debug(f"Taille après prétraitement : {image_array.shape}")

    # Charger le modèle
    logging.info(f"Chargement du modèle {model_name}...")
    model = load_model(model_name)
    model.eval()

    # Exécution de la prédiction
    logging.info("Exécution de la prédiction...")
    
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = torch.tensor(image_array).unsqueeze(0).to(device)  # Ajouter batch dimension
        outputs = model(inputs)
        mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()
    
    logging.info(f"Classes uniques prédites : {np.unique(mask)}")

    # Post-traitement du masque
    color_mask = resize_and_colorize_mask(mask, original_size, CLASS_COLORS)

    # Vérification du masque généré
    if color_mask is None or color_mask.size == 0:
        logging.error("Le masque généré est vide !")
        return {"error": "Le masque généré est vide"}

    # Convertir en image PNG
    success, buffer = cv2.imencode(".png", np.array(color_mask))

    if not success or buffer is None or len(buffer.tobytes()) == 0:
        logging.error("Échec de l'encodage du masque en PNG !")
        return {"error": "Erreur lors de l'encodage du masque prédictif"}

    logging.info(f"Masque généré avec succès ({len(buffer.tobytes())} bytes), envoi au client.")
    return Response(buffer.tobytes(), media_type="image/png")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))  # Récupérer le port défini dans Cloud Run
    uvicorn.run(app, host="0.0.0.0", port=port)
