# Résumé Du Benchmark

## Protocole
- Environnement: `highway-v0`
- Seeds d'évaluation partagées: `100..149`
- Nombre d'épisodes d'évaluation par modèle: `50`
- Hash de configuration (reproductibilité): `b1f97473cf49b8a53d9ce3c2bd3d96b6bf286bfa155888401ebe10d790dba8bb`

### Interprétation Du Protocole
- Les deux approches sont évaluées dans le même cadre, donc la comparaison est équitable.
- Les seeds identiques réduisent le biais lié au hasard.
- Le hash permet de prouver que la configuration n'a pas changé entre les évaluations.

## Visualisations

### Comparaison Finale (Reward)
![Comparaison benchmark](benchmark_comparison.png)

Interprétation:
- La reward moyenne agrégée de `sb3_dqn` est légèrement supérieure à `custom_dqn`.
- L'écart reste modéré, donc la supériorité en reward n'est pas écrasante.

### Courbes D'entraînement
![Courbes entraînement](training_curves.png)

Interprétation:
- Les courbes montrent la dynamique d'apprentissage de chaque méthode.
- On observe la stabilité/variabilité des runs SB3 entre seeds.
- La courbe custom DQN sert de référence de convergence face aux runs SB3.

## Résultats Agrégés
| algorithme | reward_moyenne | ecart_type_reward | crash_rate_moyen_pct | nb_modeles |
| --- | --- | --- | --- | --- |
| custom_dqn | 17.721299998870197 | 0.0 | 40.0 | 1 |
| sb3_dqn | 18.4234124435779 | 1.5554159852695086 | 76.66666666666667 | 3 |

### Interprétation Des Résultats Agrégés
- `reward_moyenne`: SB3 est un peu meilleur en performance brute.
- `crash_rate_moyen_pct`: le custom DQN est nettement meilleur en sécurité (moins d'accidents).
- `ecart_type_reward` côté SB3: variabilité non négligeable entre seeds.
- Attention: `custom_dqn` n'a qu'un seul modèle évalué ici (`nb_modeles=1`), donc il faut rester prudent sur la robustesse statistique.

## Résultats Par Modèle
| algorithme | modele | chemin | n_eval | reward_moyenne | reward_std | longueur_moyenne | longueur_std | crash_rate_pct | nb_crash | seeds_echec_apercu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom_dqn | dqn_v1_big_buffer_2000_episodes_last.pth | results/dqn/dqn_v1_big_buffer_2000_episodes_last.pth | 50 | 17.721299998870197 | 7.19345242377722 | 22.62 | 9.522373653664301 | 40.0 | 20 | 102,104,105,108,111,112,113,117,124,125 |
| sb3_dqn | sb3_seed_1_last.zip | results/sb3/sb3_seed_1_last.zip | 50 | 20.126932840184512 | 8.581400530522426 | 21.46 | 8.651496980291908 | 66.0 | 33 | 101,103,104,105,106,108,110,111,112,115 |
| sb3_dqn | sb3_seed_2_last.zip | results/sb3/sb3_seed_2_last.zip | 50 | 17.07887178605279 | 9.04316239351216 | 18.1 | 8.962700485902673 | 84.0 | 42 | 100,101,102,104,105,106,107,108,110,111 |
| sb3_dqn | sb3_seed_3_last.zip | results/sb3/sb3_seed_3_last.zip | 50 | 18.06443270449639 | 8.37307681181434 | 19.06 | 8.080618788186955 | 80.0 | 40 | 101,103,104,105,106,108,109,110,111,112 |

### Interprétation Par Modèle
- `SB3 seed 1` est le meilleur en reward, mais avec un crash rate élevé (66%).
- `SB3 seed 2` et `SB3 seed 3` confirment un profil plus risqué (80%+ de crash).
- Le `custom_dqn` a une reward un peu plus basse, mais un crash rate bien plus faible (40%), ce qui peut être préférable si la sécurité est prioritaire.
- Les `seeds_echec_apercu` permettent d'identifier des cas de défaillance à rejouer pour l'analyse qualitative.

## Conclusion Opérationnelle
- Si l'objectif principal est la reward: SB3 est légèrement devant.
- Si l'objectif inclut fortement la sécurité: le custom DQN est plus convaincant sur ce benchmark.
- Pour renforcer la conclusion finale, il faudrait ajouter au moins 2 autres runs custom DQN (3 seeds au total), afin d'avoir une symétrie complète avec SB3.

## Notes De Reproductibilité
- Même configuration environnementale pour toutes les évaluations.
- Même liste de seeds déterministes pour tous les modèles.
- Métadonnées complètes dans `benchmark_metadata.json`.
