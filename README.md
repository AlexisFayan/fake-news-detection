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

## 📋 Description du projet

On cherche à **détecter automatiquement les fake news politiques** grâce au NLP (traitement du langage) et au Machine Learning.

**En gros :** on donne une déclaration politique au modèle → il prédit si c'est **fake** ou **real**.

### Données utilisées

| Dataset | Description | Taille | Source |
|---------|------------|--------|--------|
| **LIAR** | Déclarations politiques US annotées par PolitiFact | ~12 800 exemples | [GitHub](https://github.com/tfs4/liar_dataset) |
| **FakeNewsNet PolitiFact** | Titres d'articles politiques (fake/real) | ~1 000 articles | [GitHub](https://github.com/KaiDMML/FakeNewsNet) |

> **Le notebook télécharge tout automatiquement**, pas besoin de télécharger les datasets à la main.

### Ce qu'on fait concrètement

1. On **explore** les données (qui parle ? quels mots ? quelle distribution ?)
2. On **prétraite** le texte (nettoyage, mapping en fake/real)
3. On crée des **features** (TF-IDF, Word2Vec, métadonnées)
4. On entraîne **3 modèles classiques** : Logistic Regression, Random Forest, XGBoost
5. On entraîne un **modèle BERT** (deep learning, optionnel car long)
6. On teste la **généralisation** sur un dataset externe (out-of-domain)
7. On analyse les **biais** (est-ce que le modèle est fair envers tous les partis ?)

---

## 🚀 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/AlexisFayan/fake-news-detection.git
cd fake-news-detection

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate     # Mac / Linux
# venv\Scripts\activate      # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Prérequis

- **Python 3.9+**
- ~2 Go d'espace disque (datasets + modèles)
- GPU optionnel (pour BERT uniquement — sinon ça tourne sur CPU, c'est juste plus long)

---

## ▶️ Lancer le notebook

```bash
jupyter notebook fake_news_detection.ipynb
```

**Exécuter les cellules de haut en bas.** Tout est auto-contenu : les datasets se téléchargent au lancement.

---

## 📓 Structure du Notebook (10 sections)

| # | Section | Ce qu'on y fait |
|---|---------|-----------------|
| 1 | **Introduction & Setup** | Imports, téléchargement auto des données |
| 2 | **Exploration (EDA)** | Distribution des labels, top speakers, wordclouds, stats |
| 3 | **Prétraitement** | Nettoyage du texte, mapping 6 labels → binaire (fake/real) |
| 4 | **Feature Engineering** | TF-IDF (10 000 features) + Word2Vec + features manuelles (longueur texte, etc.) |
| 5 | **Modélisation Classique** | Logistic Regression, Random Forest, XGBoost + cross-validation 5-fold + GridSearch |
| 6 | **Importance des Features** | SHAP values — quels mots influencent le plus les prédictions |
| 7 | **BERT** | Fine-tuning bert-base-uncased (early stopping, warmup, weight decay) |
| 8 | **Out-of-Domain** | Test sur FakeNewsNet → est-ce que le modèle généralise ? |
| 9 | **Biais & Fairness** | Performance par parti politique → le modèle est-il équitable ? |
| 10 | **Conclusion** | Tableau récapitulatif, réponses aux 3 questions, limites |

---

## 🧠 Résultats clés

### Modèles classiques (sur LIAR test set)

| Modèle | Accuracy | F1 macro |
|--------|----------|----------|
| Logistic Regression | ~64% | ~64% |
| Random Forest | ~62% | ~61% |
| XGBoost | ~73% | ~72% |

> XGBoost est le meilleur grâce aux métadonnées (credit history du speaker).

### BERT

| Modèle | Accuracy | F1 macro |
|--------|----------|----------|
| BERT (text only) | ~64% | ~61% |

> BERT n'a accès qu'au texte (pas de métadonnées), d'où un score similaire aux modèles classiques en text-only.

### Points importants à retenir

- **Les métadonnées comptent énormément** : +10 pts quand on ajoute le credit history
- **Le texte seul ne suffit pas** pour détecter les fake news de manière fiable
- **La généralisation est faible** : les modèles performent moins bien sur des données jamais vues
- **Des biais existent** : les performances varient selon le parti politique du speaker

---

## ⚠️ Note sur BERT

Le fine-tuning BERT est **désactivé par défaut** (`TRAIN_BERT = False`) car il est long (~15-30 min sur GPU).

Pour l'activer : modifier `TRAIN_BERT = True` dans la cellule 7.1 du notebook.

Si BERT a déjà été entraîné, les résultats sont sauvegardés dans `bert_results.json` et chargés automatiquement.

### Améliorations BERT intégrées
- **Early stopping** (patience=2) pour éviter l'overfitting
- **Learning rate scheduler** (warmup linéaire 10%)
- **Weight decay** (0.01) pour la régularisation
- **5 epochs** (l'early stopping coupe avant si nécessaire)

---

## 🗂️ Structure du projet

```
fake-news-detection/
├── README.md                       # Ce fichier (guide du projet)
├── requirements.txt                # Dépendances Python
├── fake_news_detection.ipynb       # Notebook principal (tout est dedans)
├── bert_results.json               # Résultats BERT sauvegardés (généré auto)
└── best_bert_model.pt              # Poids du modèle BERT (généré auto)
```

---

## 📅 Calendrier

| Date | Événement |
|------|-----------|
| **3 avril 2026** | Follow-up (Groupe 2 : 11h00) |
| **17 avril 2026** | Soutenance (Groupe 2 : 14h30) |

---

## 💡 Pour la soutenance — Points à maîtriser

1. **Pourquoi LIAR ?** → Dataset de référence pour la détection de fake news politiques, annoté par des fact-checkers professionnels
2. **Pourquoi binaire ?** → Les 6 labels originaux (pants-fire, false, barely-true, half-true, mostly-true, true) sont trop granulaires, on simplifie en fake/real
3. **TF-IDF vs Word2Vec** → TF-IDF marche mieux ici car le vocabulaire spécifique (mots-clés politiques) est important
4. **Pourquoi XGBoost gagne ?** → Il exploite mieux les métadonnées (credit history) que les autres modèles
5. **Pourquoi BERT ne fait pas mieux ?** → Il n'a que le texte, pas les métadonnées ; et LIAR est un dataset difficile
6. **Généralisation faible** → Les patterns appris sur LIAR ne se transfèrent pas bien (domain shift)
7. **Biais** → Le modèle n'est pas neutre politiquement, il faut un humain dans la boucle

---

## 📚 Références

- Wang, W. Y. (2017). *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection.* ACL 2017.
- Shu, K. et al. (2020). *FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media.*
- Silverman, C. (2016). *BuzzFeed News — Facebook Fact-Check Dataset.*
