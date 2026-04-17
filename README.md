# Autonomous-Highway-Agents

Projet RL sur l'environnement highway-v0 pour CentraleSupelec. Comparaison d'un agent DQN personnalisé avec un modèle Stable-Baselines via des métriques de performance et de sécurité. Inclut une extension algorithmique et une analyse des échecs.

## Extension : Double DQN + Prioritized Experience Replay

Implémentation du Double DQN et du Double DQN + PER sur highway-v0.
Les paramètres sont dans `configs/double_dqn_params.yaml` et `configs/double_dqn_per_params.yaml`.

### Double DQN

#### 1. Entraînement
```bash
python scripts/run_double_dqn.py --mode train --config configs/double_dqn_params.yaml
```
Sauvegarde les poids dans `models/` et les rewards dans `results/`.

#### 2. Évaluation
```bash
python scripts/run_double_dqn.py --mode eval --config configs/double_dqn_params.yaml
```
Par défaut, charge `models/<experiment_name>_final.pth`.
Pour évaluer un checkpoint précis :
```bash
python scripts/run_double_dqn.py --mode eval --config configs/double_dqn_params.yaml --model_path models/<nom_du_modele>.pth
```
Génère un GIF de rollout dans `results/<experiment_name>_rollout.gif`.

### Double DQN + PER

#### 1. Entraînement
```bash
python scripts/run_double_dqn_per.py --mode train --config configs/double_dqn_per_params.yaml
```

#### 2. Évaluation
```bash
python scripts/run_double_dqn_per.py --mode eval --config configs/double_dqn_per_params.yaml --n_runs 50 --seed 0
```

#### 3. Enregistrer un rollout
```bash
python scripts/run_double_dqn_per.py --mode record --config configs/double_dqn_per_params.yaml
```
Génère un GIF dans `results/<experiment_name>_rollout.gif`.