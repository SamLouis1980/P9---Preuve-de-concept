#  Développement d’une preuve de concept – DataSpace

## Contexte

Dans le cadre d’un test technique pour un poste chez **DataSpace**, il m’a été demandé de développer une **preuve de concept (PoC)** autour d’un **nouvel algorithme de machine learning récent**, à partir d’un projet ou d’un dataset de mon choix.

J’ai choisi de réutiliser le **projet de segmentation d’images embarqué pour voiture autonome** (projet 8), en proposant une **amélioration significative du modèle initial** grâce à un algorithme d’architecture récente.

Ce projet vise à démontrer ma capacité à effectuer une veille technologique, intégrer un nouvel algorithme, comparer les performances avec une baseline, et valoriser les résultats dans un **dashboard interactif**.

## Objectif

- Réutiliser le pipeline du projet initial avec une **baseline FPN + ResNet50**
- Intégrer une amélioration basée sur le modèle **FPN + ConvNeXt V2**
- Comparer les deux modèles selon des métriques standard de segmentation (IoU, F1, etc.)
- Valoriser les résultats dans une **note technique**, un **dashboard interactif**, et une **présentation professionnelle**
- Tester la capacité d’industrialisation du modèle amélioré dans un contexte de production

## Technologies utilisées

- `PyTorch`, `torchvision`, `transformers`, `timm`, `albumentations`
- `ConVNext V2`, `ResNet50`
- `sklearn`, `torchmetrics`, `matplotlib`, `seaborn`, `cv2`
- `Streamlit` (pour le dashboard)
- `tqdm`, `PIL`, `json`, `os`, `datetime`, `time`

## Structure du projet

- notebook.ipynb - Comparaison des deux modèles (baseline vs ConVNext V2)
- dashboard.py - Dashboard interactif déployable (Streamlit)
- plan_previsionnel.pdf - Plan de travail validé
- note_methodologique.pdf - Note expliquant la démarche et les résultats
- presentation.pptx - Présentation pour la soutenance

## Répertoire Associé : P9---Dashboard
