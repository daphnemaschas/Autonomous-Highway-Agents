# Autonomous-Highway-Agents
Projet RL sur l'environnement highway-v0 pour CentraleSupelec. Comparaison d'un agent DQN personnalisé avec un modèle Stable-Baselines via des métriques de performance et de sécurité. Inclut une extension algorithmique et une analyse des échecs.

## Le modèle custom DQN

Les paramètres du modèles sont dans le fichier dqn_params.yaml

### 1. Entraînement

```bash
python scripts/run_custom_dqn.py --mode train --config configs/dqn_params.yaml
```

Cette commande entraîne l'agent avec les paramètres du fichier YAML, sauvegarde les poids dans `models/` et les poids du modèle entrainé dans `results/`.

### 2. Évaluation

```bash
python scripts/run_custom_dqn.py --mode eval --config configs/dqn_params.yaml
```

Par défaut, l'évaluation charge le modèle final `models/<experiment_name>_final.pth`.

Pour évaluer un checkpoint précis:

```bash
python scripts/run_custom_dqn.py --mode eval --config configs/dqn_params.yaml --model_path models/<nom_du_modele>.pth
```

L'évaluation génère un GIF de rollout dans `results/<experiment_name>_rollout.gif`.

## Le modèle Stable-Baselines 3

### 1. Entraînement

```bash
python scripts/run_sb3_dqn.py --mode train_all
```

Cette commande entraîne l'agent sur 3 seeds, sauvegarde les modèles dans `models/` et les poids du modèle entrainé dans `results/`.

### 2. Évaluation

```bash
python scripts/run_sb3_dqn.py --mode eval --model_path results/sb3/sb3_seed_1_last.zip
```

L'évaluation génère un GIF de rollout dans `results/sb3/<experiment_name>_rollout.gif`.

