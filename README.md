# Détection de Fake News Politiques

Projet Master Data Science — 2026

## Description

Ce projet explore les capacités du NLP et du Machine Learning pour détecter automatiquement les déclarations politiques trompeuses. Il utilise le **LIAR Dataset** (~12 800 déclarations politiques annotées) et évalue la généralisation sur le **BuzzFeed Political News Dataset**.

## Structure du projet

```
fake-news-detection/
├── README.md                    # Ce fichier
├── requirements.txt             # Dépendances Python
└── fake_news_detection.ipynb    # Notebook principal (complet)
```

## Installation

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

```bash
jupyter notebook fake_news_detection.ipynb
```

Le notebook est entièrement auto-contenu : il télécharge automatiquement les datasets nécessaires. Exécutez les cellules de haut en bas.

### Option BERT

Par défaut, le fine-tuning BERT est désactivé (`TRAIN_BERT = False`) car il nécessite un GPU. Pour l'activer, modifiez la variable dans la première cellule de code du notebook.

## Contenu du notebook

| Section | Description |
|---------|-------------|
| 1 | Introduction & Setup |
| 2 | Exploration des données (EDA) |
| 3 | Prétraitement NLP |
| 4 | Feature Engineering (TF-IDF) |
| 5 | Modélisation classique (LR, RF, XGBoost) |
| 6 | Analyse d'importance des features (SHAP) |
| 7 | Modèle avancé : BERT (optionnel) |
| 8 | Évaluation Out-of-Domain |
| 9 | Analyse de biais et fairness |
| 10 | Discussion & Conclusion |

## Datasets

- **LIAR Dataset** (Wang, 2017) — 6 labels de véracité : pants-fire, false, barely-true, half-true, mostly-true, true
- **BuzzFeed Political News Dataset** — articles politiques annotés (test de généralisation)

## Modèles implémentés

- Logistic Regression (TF-IDF)
- Random Forest (TF-IDF)
- XGBoost (TF-IDF)
- BERT fine-tuné (optionnel)

## Résultats attendus

| Modèle | Accuracy | F1 (macro) |
|--------|----------|------------|
| Logistic Regression | ~0.61 | ~0.61 |
| Random Forest | ~0.59 | ~0.58 |
| XGBoost | ~0.60 | ~0.60 |
| BERT | ~0.64 | ~0.63 |

> Note : Le LIAR dataset est notoirement difficile. Même les modèles SOTA atteignent ~67% d'accuracy en classification binaire.
