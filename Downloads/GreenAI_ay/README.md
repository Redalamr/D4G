# GreenAI - API de Résumé Éco-Conçue

API Flask de résumé de textes optimisée pour réduire l'empreinte énergétique, développée pour le challenge Design4Green 2025.

## Description

Cette application génère des résumés de 10-15 mots en français tout en mesurant et optimisant la consommation énergétique. Elle utilise le modèle EleutherAI/pythia-70m-deduped avec deux modes :
- **Baseline** : FP32 (mode par défaut, sans optimisation)
- **Optimisé** : INT8 quantization dynamique pour réduire la consommation

## Prérequis

- Python 3.8 ou supérieur
- pip

## Installation

1. Cloner le dépôt :
```bash
git clone <repo-url>
cd GreenAI
```

2. Créer un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer l'application web

```bash
python app.py
```

L'application sera accessible sur http://127.0.0.1:5000

Ouvrez votre navigateur et accédez à l'interface web pour :
- Saisir un texte à résumer (≤ 4000 caractères)
- Choisir le mode (Baseline ou Optimisé)
- Obtenir le résumé avec les métriques (énergie, latence)

### Utiliser l'API directement

L'API expose un endpoint POST `/summarize` :

```bash
curl -X POST http://127.0.0.1:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Votre texte ici", "optimized": false}'
```

**Paramètres :**
- `text` (string, requis) : Texte à résumer (≤ 4000 caractères)
- `optimized` (boolean, requis) : `false` pour baseline, `true` pour optimisé

**Réponse :**
```json
{
  "summary": "Résumé en 10-15 mots",
  "energy_wh": 0.000123,
  "latency_ms": 245.67,
  "mode": "baseline"
}
```

## Évaluation avec judge.py

L'API est compatible avec le script d'évaluation fourni :

```bash
python judge.py
```

Le script évalue automatiquement les deux modes et affiche :
- Score final (/100)
- Économie d'énergie (%) par rapport au baseline

## Structure du projet

```
GreenAI/
├── app.py              # Application Flask principale
├── requirements.txt    # Dépendances Python
├── README.md          # Documentation
└── templates/
    └── index.html     # Interface web
```

## Optimisations appliquées

### Mode Optimisé (optimized=true)
- **Quantization INT8 dynamique** : Réduit la précision des poids du modèle de FP32 à INT8
- Réduit la consommation mémoire et accélère l'inférence sur CPU

### Mode Baseline (optimized=false)
- Modèle FP32 standard sans optimisation
- Sert de référence pour mesurer les gains

## Mesures

Toutes les métriques sont mesurées par requête :
- **Énergie** : CodeCarbon (en Wh)
- **Latence** : Temps d'inférence (en ms)
- Mesure du début à la fin de la génération uniquement

## Notes techniques

- Port par défaut : 5000
- Langue : Français
- Longueur résumé : 10-15 mots
- Reproductibilité : PYTHONHASHSEED=0, seed=42
- Modèle : EleutherAI/pythia-70m-deduped

## Améliorations futures

- Pruning léger des couches
- torch.compile pour optimisation supplémentaire
- Ajustement des hyperparamètres de génération
- Optimisation low-rank sur les couches d'attention
