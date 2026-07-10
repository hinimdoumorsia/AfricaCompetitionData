# Détection de Fraude Transactionnelle — Pipeline de Stacking Multi-Niveaux

Solution de Machine Learning pour la détection de transactions frauduleuses sur un dataset transactionnel à grande échelle (1,29M lignes), combinant une ingénierie de variables comportementales approfondie, une sélection rigoureuse des features par Drop Column Importance, une optimisation d'hyperparamètres via Optuna, et deux architectures de stacking complémentaires (cascade séquentielle et stacking Out-of-Fold) fusionnées par ensembling.

---

# Contexte

## La compétition

Le jeu de données représente un historique de transactions financières (`train.csv` / `test.csv`, dataset `tour-data`), où chaque enregistrement décrit un mouvement d'argent entre un compte d'origine et un compte de destination : montant, type d'opération, soldes avant/après transaction, et une période temporelle discrète (`period`). L'objectif est de prédire la probabilité qu'une transaction soit frauduleuse (`fraud_flag`).

## Dimensions du problème

| Élément | Valeur |
|---|---|
| Lignes d'entraînement | 1 290 081 |
| Lignes de test | 430 100 |
| Colonnes brutes (train) | 11 (10 features + cible) |
| Colonnes brutes (test) | 10 |
| Valeurs manquantes | Aucune, ni en train ni en test |
| Mémoire train | 394,93 Mo |
| Mémoire test | 128,39 Mo |

## Variable cible

`fraud_flag` est binaire, avec une répartition **modérément déséquilibrée** :

| Classe | Effectif | Proportion |
|---|---|---|
| Non-fraude (0) | 1 160 539 | 89,96 % |
| Fraude (1) | 129 542 | 10,04 % |

Un taux de fraude de 10 % rend l'accuracy inutilisable comme métrique de pilotage : un modèle prédisant systématiquement « non-fraude » atteindrait déjà ~90 % d'accuracy sans aucune valeur ajoutée.

## Métrique d'évaluation

La métrique retenue tout au long du pipeline est la **PR-AUC (Precision-Recall Area Under Curve)**, calculée via `average_precision_score` de scikit-learn et directement optimisée comme `eval_metric` dans CatBoost (`PRAUC`) et XGBoost (`aucpr`). Ce choix est cohérent avec un problème à classe minoritaire rare, où la ROC-AUC a tendance à surestimer la qualité du modèle en raison du grand nombre de vrais négatifs.

## Contraintes identifiées

- **Haute cardinalité** : `origin_account` (13 431 valeurs uniques) et `destination_account` (15 818 valeurs uniques) interdisent un one-hot encoding direct — l'information doit être capturée par agrégation.
- **Concentration du signal sur `operation`** : la quasi-totalité des fraudes est associée au type d'opération `op_03`, ce qui constitue un signal fort mais risqué en termes de généralisation (distribution shift potentiel).
- **Volumétrie** : 1,29M lignes × jusqu'à 113 features en phase d'ingénierie imposent l'usage du GPU pour les entraînements répétés (validation croisée, sélection de variables, recherche d'hyperparamètres).

---

# Vue d'ensemble de la solution

La philosophie du pipeline repose sur trois piliers :

1. **Extraction maximale du signal comportemental des comptes**, plutôt que de se limiter aux variables brutes de la transaction. Les comptes d'origine et de destination sont caractérisés par des dizaines de statistiques historiques (moyenne, médiane, quantiles, entropie, skewness, kurtosis) apprises exclusivement sur le train, afin de détecter les écarts anormaux d'une transaction par rapport au comportement habituel du compte.

2. **Sélection agressive et validée empiriquement** des variables. Plutôt que de conserver l'intégralité des 113 features générées, une procédure de sélection en 7 étapes (CatBoost + Drop Column Importance) démontre qu'un sous-ensemble compact de 50 variables surpasse le jeu complet en validation croisée — un résultat contre-intuitif qui valide l'hypothèse que le bruit ajouté par les variables redondantes dégrade la généralisation.

3. **Stacking à deux architectures complémentaires**, comparées plutôt que choisies a priori :
   - **Pipeline A** : une cascade séquentielle LightGBM → XGBoost → CatBoost, où chaque modèle enrichit le suivant avec sa probabilité prédite comme méta-feature, sur des sous-échantillons disjoints du train pour limiter le leakage.
   - **Pipeline B** : un stacking classique par prédictions Out-of-Fold (OOF), où CatBoost est entraîné en validation croisée à 5 folds pour générer des méta-features sans fuite d'information, avant qu'un XGBoost final ne les exploite.

Les deux pipelines sont ensuite moyennés pour produire une soumission d'ensemble, exploitant la diversité des erreurs entre les deux architectures.

## Points forts du pipeline

- Apprentissage cost-sensitive par pondération d'échantillons (`sample_weight`) plutôt que par sur/sous-échantillonnage, évitant la distorsion de la distribution native des données.
- Zéro fuite d'information dans le feature engineering : toutes les statistiques d'agrégation sont apprises sur le train puis réappliquées telles quelles au test.
- Accélération GPU systématique pour CatBoost et XGBoost, avec un choix assumé de garder LightGBM sur CPU.
- Auto-critique intégrée au notebook : une section dédiée identifie explicitement le risque de leakage dans les architectures de stacking naïves et justifie le passage à une approche OOF pour le Pipeline B.

---

# Architecture du projet

```
.
├── README.md
├── troisieme-notebook5.ipynb        # Notebook principal (pipeline complet)
├── data/
│   ├── train.csv                    # 1 290 081 lignes × 11 colonnes
│   └── test.csv                     # 430 100 lignes × 10 colonnes
├── outputs/
│   ├── submission.csv               # Sortie du stacking CatBoost→XGBoost (Optuna)
│   ├── submission_xgb_only.csv      # Pipeline B (XGBoost final)
│   ├── submission_pipeline_A.csv    # Pipeline A (cascade complète)
│   └── submission_ensemble.csv      # Moyenne Pipeline A + Pipeline B
└── figures/
    ├── confusion_catboost.png
    ├── confusion_xgb.png
    ├── confusion_A_lightgbm.png
    ├── confusion_A_xgboost.png
    ├── confusion_A_catboost_final.png
    └── confusion_B_xgboost.png
```

---

# Pipeline Machine Learning

```
                    df_train.csv / df_test.csv
                              │
                              ▼
                  Analyse exploratoire (EDA)
        (distribution cible, histogrammes, boxplots,
         fraude par catégorie, fraude par période)
                              │
                              ▼
                 Feature Engineering (5 vagues)
        build_features → add_group_features →
        build_extra_agg_features → build_systematic_stats →
        add_advanced_groupby_features
              (113 features générées, 0 NaN restant)
                              │
                              ▼
              Sélection de variables (CatBoost)
        CV 5-fold + importance globale + test de seuils
              Top-K + Drop Column Importance
                    → 50 features retenues
                              │
                              ▼
                 Validation (StratifiedKFold /
                  train_test_split stratifié)
                              │
                              ▼
        ┌─────────────────────┴─────────────────────┐
        ▼                                             ▼
  PIPELINE A                                    PIPELINE B
  Cascade LightGBM → XGBoost                    OOF CatBoost (5 folds)
  → CatBoost (splits 50/25/10/15)               → XGBoost final
        │                                             │
        └─────────────────────┬─────────────────────┘
                              ▼
                    Moyenne des probabilités
                    (Ensemble Pipeline A + B)
                              │
                              ▼
                    submission_ensemble.csv
```

---

# Prétraitement des données

## Valeurs manquantes

L'audit initial confirme l'**absence totale de valeurs manquantes** dans `df_train` et `df_test`. Aucune stratégie d'imputation n'est donc nécessaire sur les colonnes brutes. En revanche, le processus de feature engineering génère mécaniquement des valeurs manquantes pour les comptes présents dans le test mais absents (ou peu représentés) dans le train — ces NaN sont traités au fil de l'eau (voir section Feature Engineering).

## Imputation intelligente des features dérivées

Une fonction `smart_impute` est appliquée aux nouvelles variables agrégées susceptibles de contenir des NaN après jointure (comptes non vus, historiques insuffisants). La stratégie est **adaptative par variable** :

- Pour les variables catégorielles : imputation par le **mode**.
- Pour les variables numériques : un test de normalité (`scipy.stats.normaltest`, échantillonné à 5000 observations si nécessaire) détermine si l'imputation se fait par la **moyenne** (si p > 0.05, distribution jugée normale) ou par la **médiane** (sinon, cas majoritaire pour des variables financières asymétriques).

Cette approche évite le biais classique de l'imputation systématique par la moyenne sur des distributions à queue lourde, très fréquentes ici (montants, soldes).

## Transformation logarithmique

`log_amount = log1p(amount)` est appliqué pour compresser la plage de valeurs du montant, qui varie de 0,37 à plus de 2,5 millions selon l'analyse descriptive initiale.

## Gestion des valeurs extrêmes

Aucune suppression d'outliers n'est effectuée. L'analyse exploratoire (boxplots par classe) montre que les valeurs extrêmes sont potentiellement porteuses de signal de fraude — les transactions atypiques sont précisément ce que le modèle doit apprendre à détecter. La stratégie retenue est de **caractériser** l'anomalie via des z-scores robustes et des indicateurs de dépassement de quantiles plutôt que de l'éliminer.

## Encodage

- `operation` (5 modalités) est encodée en **one-hot** (`pd.get_dummies`, préfixe `op_`).
- `origin_account` et `destination_account`, à trop haute cardinalité pour un encodage direct, ne sont **jamais one-hot encodées** ; elles servent uniquement de clés de groupement pour générer des statistiques agrégées, puis sont supprimées de la matrice finale.

## Optimisation mémoire

Les datasets intermédiaires de stacking sont systématiquement castés en `float32` (fonction `to_float32`) avant l'entraînement des modèles en aval, réduisant l'empreinte mémoire de moitié par rapport au `float64` par défaut de pandas — un choix significatif compte tenu du volume (1,29M lignes × jusqu'à 113 colonnes).

---

# Feature Engineering

Le pipeline construit ses variables en **cinq vagues successives**, chacune s'appuyant sur les colonnes produites par la précédente, pour un total de **113 features** avant sélection.

| Famille | Feature(s) représentative(s) | Objectif | Impact attendu |
|---|---|---|---|
| Montant | `log_amount`, `amount_to_origin_balance_ratio`, `drains_account` | Normaliser l'échelle du montant et détecter les transactions qui vident le compte source | Fort — le ratio montant/solde est un signal classique de fraude par vidage de compte |
| Cohérence comptable | `origin_balance_error`, `dest_balance_error`, `accounting_mismatch` | Vérifier que `solde_après = solde_avant ± montant` | Détecte les incohérences comptables typiques de manipulations frauduleuses |
| État des soldes | `origin_goes_negative`, `origin_already_negative`, `no_balance_change_origin/dest`, `dest_was_empty` | Caractériser les états limites des comptes | Un compte qui bascule en négatif ou reste inchangé signale des comportements atypiques |
| Opération | `op_op_01` … `op_op_05` (one-hot) | Encoder le type de transaction | `op_03` concentre la quasi-totalité des fraudes observées |
| Temporel | `period_normalized`, `period_sin`, `period_cos`, `period_since_start` | Capturer la position dans le temps, y compris de façon cyclique | Permet au modèle de capter d'éventuelles saisonnalités sans supposer une périodicité fixe |
| Vélocité simple | `low_balance_high_amount` | Repérer une grosse transaction depuis un compte à faible solde | Combinaison à risque élevé |
| Statistiques comportementales (origine) | `origin_avg_amount`, `origin_median_amount`, `origin_std_amount`, `origin_skew_amount`, `origin_kurt_amount`, quantiles q10/q25/q75/q90, IQR, MAD, `origin_tx_count` | Construire un profil historique complet de chaque compte source | Permet de détecter un montant hors norme *pour ce compte spécifique*, plus discriminant qu'un seuil global |
| Anomalie relative | `origin_robust_zscore`, `amount_zscore`, `amount_above_q90`, `amount_below_q10`, `amount_in_q25_q75`, `amount_mad_ratio`, `amount_skewness_ratio` | Quantifier l'écart de la transaction courante par rapport à l'historique du compte | Cœur du système de détection d'anomalie comportementale |
| Interaction réseau | `dest_receive_frequency`, `origin_unique_destinations`, `origin_to_dest_pair_count` | Caractériser la connectivité du réseau de comptes | Une destination rare ou une paire origine-destination inhabituelle est suspecte |
| Combinaisons à risque | `high_risk_combo`, `large_to_rare_dest`, `negative_origin_large_tx` | Combiner plusieurs signaux faibles en un signal fort | Capture des patterns multi-critères difficiles à isoler individuellement |
| Statistiques de groupe (operation × amount) | `origin_balance_before_group_mean/median`, `dest_balance_*_group_mean/median`, `abs_amount_minus_mean_over_amount` | Situer chaque transaction par rapport aux transactions de même type et montant similaire | Capture des micro-patterns locaux invisibles à l'échelle globale |
| Agrégations avancées (10 nouvelles features) | `origin_n_unique_operations`, `origin_pct_drains`, `origin_pct_negative_after`, `origin_time_since_last_tx`, `origin_amount_range_ratio`, `dest_n_unique_origins`, `dest_avg_amount_received`, `dest_pct_was_empty_before`, `origin_to_dest_pair_count`, `amount_rank_in_origin_history` | Enrichir le profil comportemental avec diversité d'usage, récurrence temporelle et rang historique | Ces variables capturent la *régularité* du comportement d'un compte, complémentaire aux stats de position |
| Statistiques systématiques | `{colonne}_mean/median/std/var/skew/kurt/mad` pour les 4 colonnes de solde, groupées par `origin_account` | Généraliser l'approche statistique à toutes les variables de solde | Uniformise la couverture statistique sans sélection manuelle |
| Quantiles avancés + entropie | `orig_amt_q01…q99`, `dest_amt_q10…q99`, `origin_amount_entropy`, `dest_amount_entropy`, `origin_amount_max_bin_freq`, `origin_amount_unique_bins`, `origin_amount_skew/kurt/cv`, quantiles de solde avant/après (origine et destination) | Mesurer la diversité et la complexité de la distribution des montants par compte (via l'entropie de Shannon sur des bins de quantiles) | Un compte au comportement très régulier (entropie faible, peu de bins utilisés) contraste avec un compte erratique — utile pour isoler les comptes « mules » |

## Prévention du data leakage

Un principe est appliqué de façon systématique et vérifiable dans tout le code : **toutes les statistiques d'agrégation (moyennes, quantiles, fréquences, edges de bins) sont calculées uniquement sur `df_train`**, puis réutilisées sans recalcul sur `df_test` via des `dict`/`DataFrame` de correspondance (`origin_stats`, `dest_freq`, `group_stats`, `stats_bundle`, `bin_edges`, etc.). Les comptes non vus dans le train reçoivent des valeurs de repli issues de la transaction courante ou de statistiques globales.

## Colonnes exclues du modèle

`id`, `origin_account`, `destination_account`, `operation` (remplacée par son one-hot) et `fraud_flag` (la cible) sont retirées de la matrice de features finale via `DROP_COLS`.

---

# Sélection des variables

La sélection de variables constitue l'une des étapes les plus rigoureuses du pipeline, structurée en **sept étapes** utilisant CatBoost (GPU) et une validation croisée stratifiée à 5 folds.

## Procédure

1. **Baseline** : un CatBoost est entraîné en CV 5-fold sur les 113 features. Score de référence obtenu : **PR-AUC = 0,4069 ± 0,0044**.
2. **Importance globale** : les importances CatBoost sont moyennées sur les 5 folds pour établir un classement robuste.
3. **Test automatique de seuils Top-K** : les sous-ensembles Top-25, 50, 75, 100, 150, 200, 300 et « toutes » sont chacun réévalués en CV.
4. **Identification du meilleur seuil** : le sous-ensemble **Top-50** obtient le meilleur score, **PR-AUC = 0,4137 ± 0,0028**, soit un gain de +0,0068 par rapport à la baseline complète — la réduction de dimensionnalité améliore la généralisation.
5. **Drop Column Importance** sur le Top-100 : chaque feature est retirée individuellement puis le modèle est réentraîné ; la variation de PR-AUC mesure sa contribution réelle (plus fiable que l'importance native, qui peut favoriser des variables corrélées redondantes).
6. **Classification automatique** en 4 catégories selon le seuil de drop importance : *Très importante* (> 0,001), *Importante* (0,0005–0,001), *Faible* (0–0,0005), *Inutile* (≤ 0).
7. **Sélection finale**.

## Résultat empirique et arbitrage retenu

Fait notable : sur le Top-100, la majorité des variables affichent une **drop importance négative** — leur suppression *améliore* le score de validation, symptôme de redondance ou de bruit. La sélection stricte des seules variables à drop importance positive donne un jeu réduit dont le score en CV (**PR-AUC = 0,3984 ± 0,0019**) est **inférieur** à celui du Top-50 obtenu à l'étape 4.

Le pipeline retient donc, pour la suite de l'entraînement, le **jeu Top-50 issu du test de seuils** (étape 4) plutôt que le jeu filtré par Drop Column Importance — la métrique de validation croisée tranche en faveur de la solution la plus performante empiriquement, illustrant une discipline de sélection pilotée par la donnée plutôt que par une heuristique unique.

## Composition du jeu final (50 features)

Le jeu `X_train_selected` / `X_test_selected` combine :
- caractéristiques de transaction (`log_amount`, ratios, z-scores, MAD, skew/kurtosis) ;
- information temporelle (`period_sin`, `period_cos`, `period_since_start`, `origin_time_since_last_tx`) ;
- comportement de compte (statistiques d'origine et de destination, quantiles, fréquences) ;
- dynamique des soldes (erreurs comptables, deltas, indicateurs de compte vide) ;
- interactions réseau (fréquence de paire origine-destination) ;
- trois indicateurs one-hot de type d'opération (`op_op_03`, `op_op_04`, `op_op_05`).

---

# Validation

Deux régimes de validation coexistent selon l'objectif de l'étape :

| Étape | Stratégie | Justification |
|---|---|---|
| Sélection de variables (`cv_prauc`) | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | Préserve le ratio fraude/non-fraude dans chaque fold, indispensable sur une cible déséquilibrée à 10 % |
| Développement individuel des modèles (Optuna, CatBoost/XGBoost cost-sensitive) | `train_test_split(test_size=0.10 ou 0.15, stratify=y)` | Split stratifié simple, suffisant pour l'optimisation bayésienne itérative (coût GPU par trial) |
| Pipeline A (cascade) | Split stratifié séquentiel en 4 sous-ensembles disjoints (50 % / 25 % / 10 % / 15 %) | Chaque modèle de la cascade est entraîné sur une portion distincte des données pour limiter la fuite entre étages successifs |
| Pipeline B (OOF) | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` pour générer les prédictions Out-of-Fold de CatBoost | Garantit que chaque méta-feature `cat_proba` est produite par un modèle n'ayant jamais vu l'échantillon concerné — condition nécessaire à un stacking valide |

`RANDOM_STATE = 42` est fixé de façon cohérente dans l'ensemble du notebook, garantissant la reproductibilité des splits.

## Diagnostic explicite de leakage

Le notebook contient une section d'auto-critique (« Pipeline Review and Recommendations ») qui identifie directement le risque de fuite d'information dans une architecture de stacking naïve où un modèle prédit sur les données mêmes qui ont servi à l'entraîner — et motive le passage à une génération de méta-features par OOF pour le Pipeline B. Cette vigilance méthodologique est cohérente avec les bonnes pratiques de compétition.

---

# Modèles utilisés

## CatBoost

**Rôle** : premier niveau du Pipeline B (OOF) et étage final de la cascade du Pipeline A.

**Principe** : gradient boosting sur arbres symétriques (oblivious trees), avec un traitement natif et sans encodage préalable des variables catégorielles (non exploité ici puisque les variables catégorielles à haute cardinalité sont déjà transformées en statistiques numériques en amont).

**Hyperparamètres clés (config finale, Pipeline A)** : `iterations=1429`, `depth=10`, `learning_rate≈0,252`, `l2_leaf_reg≈0,0012`, `bagging_temperature≈0,794`, `task_type='GPU'`, `eval_metric='PRAUC'`.

**Forces** : robustesse au surapprentissage grâce au boosting ordonné, gestion native du calcul de la PR-AUC comme métrique de suivi, bonnes performances hors-boîte sur données tabulaires bruitées.

**Faiblesses observées** : la métrique PR-AUC n'est pas implémentée nativement sur GPU (message répété dans les logs : *« Metric PRAUC is not implemented on GPU »*), forçant un calcul CPU de l'évaluation à chaque itération — un compromis vitesse/précision assumé dans tout le pipeline.

**Pourquoi ce modèle ici** : sa robustesse en fait un excellent générateur de méta-features de premier niveau (probabilités fiables même sur un sous-échantillon réduit, comme le 10 % alloué en cascade).

## XGBoost

**Rôle** : deuxième étage de la cascade (Pipeline A) et modèle final consommant les méta-features CatBoost (Pipeline B).

**Principe** : gradient boosting histogram-based (`tree_method='hist'`), accéléré GPU (`device='cuda'`).

**Hyperparamètres clés (config finale)** : `n_estimators=1461`, `max_depth=9`, `learning_rate≈0,212`, `subsample≈0,947`, `colsample_bytree≈0,759`, `max_delta_step=3`, `eval_metric='aucpr'`.

**Forces** : rapidité d'entraînement sur GPU, excellente capacité à exploiter des méta-features à forte séparabilité (probabilités/logits d'un modèle amont).

**Faiblesses observées** : un écart marqué train/val (PR-AUC train = 0,8705 vs val = 0,4475 lors de la phase Optuna cost-sensitive) déclenche l'alerte de surapprentissage intégrée au code (`if pr_auc_train - pr_auc_val > 0.10`). Ce diagnostic, bien qu'automatisé, souligne un risque de sur-ajustement des configurations à forte profondeur/faible régularisation retenues par Optuna.

**Pourquoi ce modèle ici** : sa capacité à intégrer efficacement des méta-features probabilistes (probabilité + logit) en fait le choix naturel pour le rôle de méta-apprenant final dans les deux pipelines.

## LightGBM

**Rôle** : premier étage de la cascade du Pipeline A uniquement.

**Principe** : gradient boosting histogram-based avec croissance leaf-wise, généralement plus rapide que XGBoost à profondeur égale.

**Hyperparamètres clés** : `n_estimators=1237`, `max_depth=9`, `num_leaves=81`, `learning_rate≈0,0011` (volontairement faible), `subsample≈0,990`, `colsample_bytree≈0,822`, **`device='cpu'`**.

**Forces** : rapide, économe en mémoire, bon comportement sur des jeux de très grande taille.

**Faiblesses observées / choix d'implémentation** : contrairement à CatBoost et XGBoost, LightGBM est exécuté **sur CPU et non sur GPU** dans ce pipeline — un choix délibéré compte tenu de l'instabilité connue de son accélération GPU combinée à des poids d'échantillon (`sample_weight`) sur ce type de configuration.

**Pourquoi ce modèle ici** : positionné en tête de la cascade A, il bénéficie de la plus grande portion de données (50 %) et sert de filtre grossier rapide avant que XGBoost et CatBoost n'affinent la prédiction sur des sous-échantillons plus réduits.

---

# Optimisation des hyperparamètres

## Outil : Optuna

L'optimisation bayésienne est réalisée avec `optuna.samplers.TPESampler(seed=RANDOM_STATE)`, sur `N_TRIALS = 50` essais par modèle, avec pour objectif la maximisation de la PR-AUC de validation (`average_precision_score`).

## Paramètres optimisés par modèle

| Modèle | Espace de recherche |
|---|---|
| CatBoost | `iterations` [200, 1500], `depth` [3, 10], `learning_rate` (log, [1e-3, 0.3]), `l2_leaf_reg` (log, [1e-3, 10]), `bagging_temperature` [0, 2], `random_strength` (log, [1e-3, 10]), `border_count` [32, 255], `min_data_in_leaf` [1, 50] |
| XGBoost | `n_estimators` [200, 1500], `max_depth` [3, 10], `learning_rate` (log, [1e-3, 0.3]), `subsample` [0.5, 1.0], `colsample_bytree` [0.4, 1.0], `min_child_weight` [1, 10], `gamma` [0, 5], `reg_alpha`/`reg_lambda` (log, [1e-4, 10]), `max_delta_step` [0, 10] |

## Pourquoi ces paramètres

- `depth`/`max_depth` et `learning_rate` contrôlent conjointement le compromis biais-variance, particulièrement sensible sur une cible à 10 % de positifs.
- `l2_leaf_reg`, `reg_alpha`, `reg_lambda` régularisent explicitement pour compenser le risque de surapprentissage identifié empiriquement (écart train/val de XGBoost).
- `bagging_temperature` et `random_strength` (CatBoost) injectent de la stochasticité pour améliorer la robustesse sur un jeu de données à motifs répétitifs (comptes récurrents).
- Early stopping (`early_stopping_rounds=30`) est utilisé pendant la recherche Optuna pour accélérer chaque essai, mais **désactivé lors de l'entraînement final** sur les meilleurs paramètres trouvés, une fois le nombre d'itérations optimal déjà déterminé.

## Résultats obtenus (phase de calibrage cost-sensitive)

| Modèle | Meilleur PR-AUC validation | Seuil optimal (F1) |
|---|---|---|
| CatBoost | 0,4529 | 0,5005 |
| XGBoost | 0,4545 | 0,0474 |

Ces hyperparamètres optimaux sont ensuite **figés** (`BEST_LGB_PARAMS`, `BEST_XGB_A_PARAMS`, `BEST_CAT_A_PARAMS`, `BEST_CAT_B_PARAMS`, `BEST_XGB_B_PARAMS`) et réutilisés tels quels dans les Pipelines A et B, séparant clairement la phase de recherche d'hyperparamètres de la phase d'évaluation des architectures d'ensemble — une bonne pratique qui évite de reconduire une recherche coûteuse à chaque comparaison d'architecture.

---

# Stratégie d'ensemble

Le notebook implémente **deux architectures de stacking distinctes**, dont les prédictions sont ensuite moyennées.

## Pipeline A — Cascade séquentielle

```text
                    Features sélectionnées (50)
                              │
                              ▼
                   LightGBM (50 % du train)
                              │
                 + probabilité + logit LightGBM
                              │
                              ▼
                   XGBoost (25 % du train)
                              │
                 + probabilité + logit XGBoost
                              │
                              ▼
                  CatBoost (10 % du train)
                              │
                              ▼
                 Validation finale (15 % du train)
                              │
                              ▼
                    Prédictions sur test
```

Chaque modèle de la cascade est entraîné sur une portion **disjointe** du jeu d'entraînement (splits successifs 50 % / 25 % / 10 % / 15 %), afin qu'aucun modèle en aval ne voie de prédictions générées sur ses propres données d'entraînement. Les probabilités et logits de chaque modèle (`add_logit`, transformation logit inverse de la sigmoïde) sont concaténés comme nouvelles colonnes au jeu de features du modèle suivant — un enrichissement de type stacking séquentiel plutôt qu'une simple moyenne.

**Résultats Pipeline A** :

| Étage | PR-AUC | Seuil F1 | F1 |
|---|---|---|---|
| LightGBM | 0,4477 | 0,6735 | 0,4914 |
| XGBoost-A | 0,4473 | 0,1962 | 0,4913 |
| CatBoost final (cascade complète) | 0,4480 | 0,0081 | 0,4897 |

## Pipeline B — Stacking Out-of-Fold

```text
               Jeu d'entraînement complet
                         │
                         ▼
              StratifiedKFold à 5 plis
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
     Fold 1           Fold 2          ...  Fold 5
        │                │                  │
        ▼                ▼                  ▼
     CatBoost        CatBoost           CatBoost
        └────────────────┴──────────────────┘
                         ▼
              Prédictions Out-of-Fold
           (probabilité + logit CatBoost)
                         │
                         ▼
             Dataset augmenté (52 colonnes)
                         │
                         ▼
                  XGBoost final
                         │
                         ▼
               Prédictions test
```

Chaque pli génère une prédiction sur les données qu'il n'a **pas** vues pendant son entraînement (`oof_cat_B[vl_idx]`), garantissant l'absence de fuite dans la méta-feature `cat_proba`. Les 5 modèles CatBoost sont également moyennés pour produire la prédiction sur le jeu de test (`test_cat_B.mean(axis=1)`).

**Résultats Pipeline B** :

| Étape | Score |
|---|---|
| CatBoost — Fold 1 | PR-AUC = 0,4487 |
| CatBoost — Fold 2 | PR-AUC = 0,4479 |
| CatBoost — Fold 3 | PR-AUC = 0,4487 |
| CatBoost — Fold 4 | PR-AUC = 0,4502 |
| CatBoost — Fold 5 | PR-AUC = 0,4451 |
| **CatBoost OOF global** | **0,4481** |
| XGBoost final | PR-AUC = 0,7398, Seuil F1 = 0,7852, F1 = 0,6660 |

## Fusion finale

```text
Pipeline A (proba test) ──┐
                           ├── moyenne simple ──> submission_ensemble.csv
Pipeline B (proba test) ──┘
```

`ensemble_proba = (pipe_A_proba_test + pipe_B_proba_test) / 2`

Une moyenne arithmétique simple (non pondérée) est retenue pour combiner les deux architectures, exploitant leur diversité structurelle (cascade séquentielle vs. OOF) sans introduire de paramètre de pondération supplémentaire à calibrer.

---

# Génération des prédictions

```text
X_test (données brutes)
        │
        ▼
Feature engineering (mêmes fonctions, stats apprises sur train)
        │
        ▼
X_test_selected (50 features)
        │
        ├──────────────► Pipeline A : LightGBM → XGBoost → CatBoost
        │                        │
        │                        ▼
        │                 pipe_A_proba_test
        │
        └──────────────► Pipeline B : CatBoost (moyenne 5 folds) → XGBoost
                                 │
                                 ▼
                          pipe_B_proba_test
                                 │
                                 ▼
                  ensemble_proba = moyenne(A, B)
                                 │
                                 ▼
                     submission_ensemble.csv
                     (colonnes : id, target)
```

Trois fichiers de soumission distincts sont produits pour permettre une comparaison sur le leaderboard :

- `submission_xgb_only.csv` : sortie brute du Pipeline B.
- `submission_pipeline_A.csv` : sortie brute du Pipeline A.
- `submission_ensemble.csv` : moyenne des deux.

---

# Analyse des performances

## Synthèse comparative

| Approche | PR-AUC (validation) | Remarque |
|---|---|---|
| CatBoost seul (Optuna, cost-sensitive) | 0,4529 | Split unique 90/10 |
| XGBoost seul (Optuna, cost-sensitive) | 0,4545 | Split unique 90/10, écart train/val important (0,87 vs 0,45) |
| Pipeline A (cascade complète) | 0,4480 | Validation finale sur 15 % non vus par aucun étage |
| Pipeline B — CatBoost OOF | 0,4481 | Moyenne robuste sur 5 folds, faible variance inter-fold (0,4451–0,4502) |
| Pipeline B — XGBoost final | 0,7398 | **À interpréter avec prudence, voir ci-dessous** |

## Risques d'overfitting identifiés

Deux signaux de surapprentissage méritent d'être soulignés :

1. **XGBoost en phase Optuna** (cost-sensitive, avant les pipelines A/B) : le pipeline détecte lui-même l'écart entre PR-AUC train (0,8705) et validation (0,4475), déclenchant l'avertissement automatique intégré au code. La configuration retenue par Optuna privilégie une profondeur élevée (max_depth=9-10) qui favorise la mémorisation.

2. **XGBoost final du Pipeline B** : le score de validation de 0,7398 (bien supérieur au PR-AUC OOF de CatBoost de 0,4481 qui l'alimente) doit être interprété avec prudence. Le modèle final est entraîné sur `X_B_aug_df` (l'intégralité du jeu augmenté) puis évalué sur `X_B_xgb_vl`, un sous-ensemble *extrait de ce même jeu*. Le score obtenu reflète donc en partie une performance en re-substitution plutôt qu'une généralisation pure — ce point est développé dans les axes d'amélioration.

## Stabilité et robustesse

Le Pipeline B affiche une **excellente stabilité inter-fold** pour la génération OOF de CatBoost (écart-type implicite faible entre 0,4451 et 0,4502 sur les 5 folds), signe d'un modèle bien régularisé et peu sensible à la partition des données. Le Pipeline A, bien que construit sur des sous-échantillons plus réduits à chaque étage (jusqu'à 10 % pour CatBoost, ≈129 000 lignes), conserve des scores cohérents (0,4473–0,4480) à chaque niveau, ce qui valide la stratégie de cascade malgré la réduction progressive de la taille d'entraînement.

---

# Importance des variables

L'importance est calculée nativement par CatBoost (`get_feature_importance()`), moyennée sur les 5 folds de validation croisée lors de la phase de sélection.

## Variables dominantes (Top features, importance moyenne CatBoost)

| Feature | Importance moyenne |
|---|---|
| `dest_receive_frequency` | 31,65 |
| `origin_balance_before_group_median` | 21,34 |
| `log_amount` | 20,63 |
| `origin_n_unique_operations` | 10,44 |
| `period` | 7,37 |
| `dest_pct_was_empty_before` | 6,06 |
| `origin_max_amount` | 2,51 |

Une observation majeure : **un très petit nombre de variables concentre l'essentiel du pouvoir prédictif**. Une large proportion des 113 features affiche une importance native de 0 dans le classement CatBoost, tandis que la majorité des variables du Top-100 obtiennent une **drop importance négative ou nulle** lors de l'analyse par suppression individuelle — signe de redondance entre variables issues des mêmes agrégations sous-jacentes (moyenne/médiane/quantiles calculés sur la même colonne group by, fortement corrélés entre eux).

Ce constat corrobore la décision de sélection : le sous-ensemble Top-50 conserve le signal utile en éliminant le bruit combinatoire généré par les cinq vagues de feature engineering.

---

# Optimisations

| Optimisation | Mise en œuvre |
|---|---|
| GPU CatBoost | `task_type='GPU'`, `devices='0'` sur tous les entraînements CatBoost |
| GPU XGBoost | `tree_method='hist'`, `device='cuda'` sur tous les entraînements XGBoost |
| CPU LightGBM | Choix délibéré (`device='cpu'`) pour éviter l'instabilité GPU constatée avec les poids d'échantillon |
| Réduction mémoire | Cast systématique en `float32` des datasets augmentés de stacking (`to_float32`) |
| Réduction de dimensionnalité | Passage de 113 à 50 features, réduisant le coût de chaque entraînement en cascade et en CV |
| Séparation recherche / évaluation | Hyperparamètres Optuna figés en dictionnaires (`BEST_*_PARAMS`) réutilisés sans recalcul dans les Pipelines A/B, évitant de relancer 50 essais Optuna par comparaison d'architecture |
| Apprentissage cost-sensitive | Pondération d'échantillon (`sample_weight`) calculée par `compute_class_weights`/`make_sample_weights` plutôt que ré-échantillonnage, évitant la duplication de données et la charge mémoire associée |

---

# Reproductibilité

| Élément | Valeur |
|---|---|
| Seed globale | `RANDOM_STATE = 42`, appliquée de façon cohérente aux splits, aux folds et aux modèles |
| Sampler Optuna | `TPESampler(seed=RANDOM_STATE)` |
| Environnement | Kaggle Notebooks, GPU T4 (task_type/device GPU pour CatBoost et XGBoost) |
| Bibliothèques principales | `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `catboost`, `xgboost`, `lightgbm`, `optuna` |
| Installation | `pip install optuna xgboost catboost` (LightGBM et les bibliothèques scientifiques standards sont préinstallées sur l'image Kaggle) |

---

# Lancer le projet

```bash
# 1. Installer les dépendances spécifiques
pip install optuna xgboost catboost

# 2. Charger les données
#    df_train = pd.read_csv("data/train.csv")
#    df_test  = pd.read_csv("data/test.csv")

# 3. Exécuter les cellules du notebook dans l'ordre :
#    a. EDA (distribution cible, histogrammes, boxplots, analyses catégorielles/temporelles)
#    b. Feature engineering (5 vagues successives)
#    c. Sélection de variables (7 étapes, CatBoost + Drop Column Importance)
#    d. Calibrage individuel CatBoost / XGBoost via Optuna (optionnel — hyperparamètres déjà fournis)
#    e. Pipeline A (cascade LightGBM → XGBoost → CatBoost)
#    f. Pipeline B (OOF CatBoost → XGBoost)
#    g. Génération des soumissions et de l'ensemble final
```

> Le notebook attend un GPU disponible (`task_type='GPU'` / `device='cuda'`) pour CatBoost et XGBoost. En son absence, remplacer ces paramètres par leurs équivalents CPU (`task_type='CPU'`, `device='cpu'`ou suppression du paramètre) au prix d'un temps d'exécution significativement plus long sur 1,29M lignes.

---

# Structure des sorties

| Fichier | Contenu |
|---|---|
| `submission.csv` | Sortie du stacking cost-sensitive CatBoost → XGBoost calibré par Optuna |
| `submission_xgb_only.csv` | Sortie du Pipeline B (XGBoost final sur méta-features OOF CatBoost) |
| `submission_pipeline_A.csv` | Sortie du Pipeline A (cascade LightGBM → XGBoost → CatBoost) |
| `submission_ensemble.csv` | Moyenne des probabilités Pipeline A et Pipeline B — **soumission recommandée** |
| `confusion_catboost.png` | Matrices de confusion train/validation, CatBoost cost-sensitive |
| `confusion_xgb.png` | Matrices de confusion train/validation, XGBoost cost-sensitive |
| `confusion_A_lightgbm.png` | Matrice de confusion, étage LightGBM du Pipeline A |
| `confusion_A_xgboost.png` | Matrice de confusion, étage XGBoost du Pipeline A |
| `confusion_A_catboost_final.png` | Matrice de confusion, validation finale du Pipeline A |
| `confusion_B_xgboost.png` | Matrices de confusion train/validation, XGBoost final du Pipeline B |

Chaque fichier `submission_*.csv` contient deux colonnes : `id` (identifiant de transaction du test) et `target` (probabilité de fraude prédite, valeur continue entre 0 et 1).

---

# Axes d'amélioration

Les pistes ci-dessous sont ciblées sur les limites concrètement observées dans ce pipeline, et non des recommandations génériques :

1. **Corriger l'évaluation du XGBoost final du Pipeline B.** Le modèle est actuellement entraîné sur `X_B_aug_df` (jeu complet) puis évalué sur `X_B_xgb_vl`, un sous-ensemble de ce même jeu — ce qui explique l'écart entre le PR-AUC affiché (0,7398) et le PR-AUC OOF de CatBoost qui l'alimente (0,4481). Il faudrait soit évaluer XGBoost sur un jeu de hold-out totalement disjoint du jeu d'entraînement final, soit générer les prédictions de XGBoost également en OOF pour obtenir un score de validation non biaisé, cohérent avec ce qui est réellement attendu sur le leaderboard.

2. **Réduire la redondance du feature engineering en amont plutôt qu'en aval.** L'analyse Drop Column Importance révèle qu'une majorité des 113 features générées par les cinq vagues successives (notamment les statistiques systématiques et les quantiles avancés) sont soit inutiles, soit légèrement délétères. Une consolidation des fonctions de génération (fusion des logiques `build_features`, `add_group_features`, `build_extra_agg_features`, qui recalculent parfois des statistiques très proches sur les mêmes colonnes de solde) réduirait le coût de calcul de l'étape de feature engineering elle-même, actuellement la plus lourde du pipeline sur 1,29M lignes.

3. **Étendre l'OOF au Pipeline A.** Le Pipeline A limite le leakage par des splits séquentiels disjoints (50/25/10/15), mais chaque étage n'est entraîné que sur une fraction réduite des données (jusqu'à 129 008 lignes pour CatBoost, contre 1,29M disponibles). Une architecture en OOF à tous les étages, comme pour le Pipeline B, permettrait d'exploiter 100 % du volume d'entraînement à chaque niveau de la cascade, au prix d'un coût de calcul plus élevé (K entraînements par étage plutôt qu'un seul).

4. **Pondérer l'ensemble plutôt que le moyenner uniformément.** La moyenne simple entre Pipeline A (PR-AUC ≈ 0,448) et Pipeline B (PR-AUC OOF ≈ 0,448, mais score final biaisé) traite les deux architectures à égalité. Un blending pondéré, calibré sur un jeu de hold-out commun aux deux pipelines (actuellement absent), ou un méta-modèle de second niveau (stacking à deux étages), pourrait mieux exploiter la diversité structurelle des deux approches.

5. **Explorer une régularisation renforcée pour XGBoost.** L'écart train/val de 0,87 vs 0,45 détecté par le garde-fou intégré au code suggère que l'espace de recherche Optuna (profondeur jusqu'à 10, `max_delta_step` jusqu'à 10) autorise des configurations trop expressives pour la taille effective du jeu d'entraînement cost-sensitive. Réduire la profondeur maximale ou ajouter une contrainte plus stricte sur `reg_lambda` pourrait resserrer cet écart.

6. **Valider la robustesse temporelle via un split par `period`.** L'analyse exploratoire révèle une variable `period` fortement informative et une possible dérive temporelle des patterns de fraude. Un split de validation basé sur les périodes les plus récentes (plutôt qu'un split aléatoire stratifié) permettrait de mesurer la capacité du modèle à généraliser à des périodes futures — un test plus proche des conditions réelles de déploiement que la validation croisée aléatoire actuelle.

---

# Conclusion

Ce pipeline démontre une approche disciplinée de la détection de fraude sur données transactionnelles déséquilibrées : une ingénierie de variables comportementales approfondie et sans fuite d'information, une sélection de variables validée empiriquement plutôt qu'arbitraire (le Top-50 surpasse le jeu complet de 113 features), un apprentissage cost-sensitive maîtrisé, et une comparaison honnête entre deux architectures de stacking aux philosophies différentes — cascade séquentielle disjointe versus stacking Out-of-Fold classique — avant fusion en ensemble.

Sa principale force réside dans la traçabilité des choix : chaque décision de feature engineering, de sélection ou d'architecture est justifiée par un score de validation croisée mesuré, et le notebook intègre lui-même des garde-fous de détection de surapprentissage et de leakage. Sa principale limite, également identifiée en interne, concerne l'évaluation finale du Pipeline B, dont le score affiché doit être recalculé sur un jeu de validation strictement disjoint du jeu d'entraînement pour refléter fidèlement la performance attendue en production ou sur le leaderboard.

## Contributor

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/hinimdoumorsia">
        <img src="https://github.com/hinimdoumorsia.png" width="100px;" alt="Hinimdou Morsia Guitdam"/>
        <br />
        <sub><b>Hinimdou Morsia Guitdam</b></sub>
      </a>
      <br />
      <sub>AI & Data Science Engineer</sub>
    </td>
  </tr>
</table>
