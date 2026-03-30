# 🔍 Détection de Fake News Politiques

**Projet 3 — Module Data Science & Business Intelligence**  
Master 2 Epitech — Mars 2026

---

## 👥 Groupe 2

| Membre |
|--------|
| Alexis FAYAN |
| Yassin CHAAIRATE |
| Jacqueline MARIANADIN |
| Ethan HARY |
| Aissatou-Blondin DIOP |

---

## 📋 Description

Ce projet explore la détection automatique de **fake news politiques** à l'aide de techniques NLP et Machine Learning.

- **Dataset principal :** [LIAR Dataset](https://github.com/tfs4/liar_dataset) (~12 800 déclarations politiques annotées par PolitiFact)
- **Dataset externe :** [BuzzFeed Political News Dataset](https://github.com/BuzzFeedNews/2016-10-facebook-fact-check) (test de généralisation)

**Objectif :** Classifier des déclarations politiques selon leur véracité, puis évaluer la capacité de généralisation du modèle sur des données jamais vues.

---

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/AlexisFayan/fake-news-detection.git
cd fake-news-detection

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
jupyter notebook fake_news_detection.ipynb
```

> Le notebook est **auto-contenu** : il télécharge automatiquement les datasets. Exécutez les cellules de haut en bas.

---

## 📓 Structure du Notebook

| # | Section | Contenu |
|---|---------|---------|
| 1 | Introduction & Setup | Imports, téléchargement automatique des données |
| 2 | Exploration (EDA) | Distribution des labels, speakers, partis, wordclouds |
| 3 | Prétraitement | Nettoyage NLP, mapping binaire (fake/real), analyse déséquilibre |
| 4 | Feature Engineering | TF-IDF (unigrams + bigrams, 10 000 features) |
| 5 | Modélisation Classique | Logistic Regression, Random Forest, XGBoost |
| 6 | Importance des Features | SHAP values, mots les plus discriminants |
| 7 | BERT (optionnel) | Fine-tuning bert-base-uncased |
| 8 | Out-of-Domain | Test sur BuzzFeed, analyse du domain shift |
| 9 | Biais & Fairness | Performance par affiliation politique |
| 10 | Conclusion | Réponses aux questions, limites, pistes d'amélioration |

### ⚠️ Note sur BERT
Le fine-tuning BERT est désactivé par défaut (`TRAIN_BERT = False`) car il nécessite un GPU. Pour l'activer, modifiez la variable dans la Section 1 du notebook.

---

## 🗂️ Structure du Projet

```
fake-news-detection/
├── README.md                     # Ce fichier
├── requirements.txt              # Dépendances Python
└── fake_news_detection.ipynb     # Notebook principal
```

---

## 📅 Calendrier

| Date | Événement |
|------|-----------|
| 3 avril 2026 | Follow-up (Groupe 2 : 11h00) |
| 17 avril 2026 | Soutenance (Groupe 2 : 14h30) |

---

## 📚 Références

- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017.
- Silverman, C. (2016). *BuzzFeed News — Facebook Fact-Check Dataset.*
