# Vote d'éligibilité v2 — métadonnées du run

**29–30 septembre 2026.** Fiche de provenance du vote qui produit la ressource.
Existe parce que le vote d'avril 2026 n'est **pas** reproductible : il n'avait
consigné que des alias flottants, et les versions réellement servies ne sont plus
récupérables. Voir `docs/note-filtre-eligibilite.md` pour le raisonnement.

## Population votée

| | |
| --- | --- |
| Total | **1 922 papiers** |
| dont `positif_v1` | 1 890 — les POSITIVE de la majorité d'avril (`merged.parquet`) |
| dont `jamais_vote` | 32 — positifs SciBERT absents de `merged.parquet`, trou de lignée |
| Entrée | `input_1922.parquet` (bibkey, title, abstract, groupe) |
| Prompt envoyé | `Title: {title}\nAbstract: {abstract}` |

## Modèles — versions exactes résolues

Relevées le 30/09/2026 en interrogeant les API, **pas** déduites des alias.

| Votant | Alias appelé | **Version réellement servie** | Comment vérifié |
| --- | --- | --- | --- |
| Mistral | `mistral-large-latest` | **`mistral-large-2512`** | `models.list()` déclare l'alias |
| Google | `gemini-3-flash-preview` | **`3-flash-preview-12-2025`** | `models.get()` + `job.model` |
| DeepSeek | `deepseek-chat` | **`deepseek-flash`** = DeepSeek-V4.1-Flash | champ `model` de la réponse |

Trois réserves à connaître :

- `gemini-3-flash-preview` **n'apparaît plus** dans `models.list()` — c'est un
  preview en fin de vie qui route encore. Il peut disparaître sans préavis.
- `deepseek-chat` a été **retiré le 24/07/2026** ; il ne fonctionne plus que par
  compatibilité. `MODEL_MAP` a depuis été épinglé sur `deepseek-v4-pro`
  (commit `d3d6d58`), donc **un nouveau run ne reproduira pas ce vote**.
- `mistral-large-latest` continuera de dériver. Pour reproduire, appeler
  `mistral-large-2512` explicitement.

## Prompt

Version issue de `96182fb` (schemas.py, `_TAXONOMY_PROMPT`, 4 266 caractères).
Trois écarts avec l'ancien prompt, tous adossés aux 8 catégories de
`manual_exclusions.parquet` écrites le 12/06/2026 :

1. `relation classification` retiré des exemples positifs — il contredisait la
   catégorie d'exclusion du même nom
2. « discrete, fixed-set label » → « label set **FIXED FOR THE ENTIRE DATASET** »,
   avec trois corollaires (inventaire global, unité textuelle, label effectivement
   assigné)
3. Deux exclusions ajoutées en STEP 2 : `Non-Textual Input` et
   `Unit Classified Is Not a Text`

Ancien prompt : 2 730 caractères, récupérable par `git show ae03d1b~1`.

## Jobs

| Votant | Identifiant | Résultat |
| --- | --- | --- |
| Mistral | `3ba072bf-3e2d-440b-bf1c-27978cce10f0` | 1 922/1 922 |
| Google | `batches/gtmgc4xya1xzpwqmgjh4k9jew74qi0hq35n6` | 1 921/1 922 |
| DeepSeek | asynchrone, pas de job | 1 922/1 922, 0 erreur API |

**Un vote Google manquant** : `de-clercq-van-hoecke-2020-exploratory`, le modèle a
dégénéré et produit un JSON invalide de 62 000 caractères.
`google_batch_results` saute silencieusement les entrées invalides. Le comptage
traite un vote absent comme **absent**, pas comme un rejet.

## Résultats par votant

| Votant | Rejette | Sur-rejet sur les 130 vrais du gold | Captation des 20 hors-périmètre |
| --- | --- | --- | --- |
| Mistral | 294 (15,3 %) | 9/130 = 6,9 % [3,7–12,6] | 12/20 |
| Google | 492 (25,6 %) | 17/130 = 13,1 % [8,3–19,9] | 16/20 |
| DeepSeek Flash | 697 (36,3 %) | 30/130 = 23,1 % [16,7–31,0] | 18/20 |

Accord (nombre de POSITIVE sur 3) : `{0: 229, 1: 260, 2: 276, 3: 1157}`

## Règles d'agrégation comparées

| Règle de rejet | Sur-rejet des vrais | Captation | Corpus final |
| --- | --- | --- | --- |
| Unanimité 3/3 | 7/130 = 5,4 % [2,6–10,7] | 12/20 | 1 294 (−9,4 %) |
| Mistral seul | 9/130 = 6,9 % [3,7–12,6] | 12/20 | 1 259 (−11,8 %) |
| **Majorité 2/3** | 15/130 = 11,5 % [7,1–18,2] | 16/20 | **1 108 (−22,4 %)** |

Entrants parmi les 32 oubliés, sous majorité : **12**.

## Pourquoi le « sur-rejet » ci-dessus est surestimé

Deux examens qualitatifs indépendants disent que ces rejets sont majoritairement
**justes**, et que c'est le gold qui avait laissé passer des papiers hors périmètre :

- sur les 9 rejets de Mistral : 6 justes, 2 discutables (STEP 1.1, non modifié), 1 faux
- sur les 8 vrais benchmarks que la majorité écarte et que l'unanimité garde :
  **7 justes**, 1 faux (`ghosh-chowdhury-etal-2019-youtoo`)

Les 7 justes : corpus multimodal annoté en continu (SEMAINE), évaluation de
capacité de LLM (MiQA, BanStereoSet, XSTest), entrées de dictionnaire
(Wiktionary), relations prédicat-argument proches du SRL, papier sans nouveau
benchmark.

La case « est-ce un benchmark de classification ? » du gold était un filtre rapide
en tête d'annotation, pas un audit de périmètre — l'attention portait sur
l'assignation taxonomique. **Le gold n'est donc pas une référence fiable pour
arbitrer entre les règles d'agrégation.**

## Un piège de mesure à ne pas refaire

Le sur-rejet du vote d'avril paraît être de 0 %. C'est **circulaire** : le gold a
été tiré dans le corpus, et le corpus est fait des papiers qu'avril a acceptés.
Les papiers qu'avril rejetait à tort ne sont jamais entrés, donc jamais annotés.
Les taux individuels d'avril (Mistral 1,5 %, DeepSeek 3,8 %, Google 3,1 %) sont
bornés par la même contrainte : il fallait 2 POSITIVE sur 3 pour entrer, donc au
plus un votant pouvait dissentir. Comparer « 0 % en avril » à « 11,5 % aujourd'hui »
n'a pas de sens.

## Ce qui reste à décider

Relire 30 papiers rejetés par exactement 2 votants sur 3 (~180 dans le corpus).
Si ≥ 80 % des rejets sont justes, appliquer la majorité ; sinon l'unanimité.

## Fichiers

```
input_1922.parquet             la population votée
mistral.parquet                vote Mistral (colonne label_v2)
google.parquet                 vote Google  (colonne google_label)
deepseek.parquet               vote DeepSeek Flash (colonne deepseek_label)
majority_v2.parquet            les trois votes + n_pos + majorite
regression_gold150.parquet     Mistral sur les 150 du gold
smoke_gold40.parquet           premier test, 40 papiers
deepseek_pro_gold150.parquet   DeepSeek V4-Pro sur les 150 (18,5 % de sur-rejet)
```

Copie de sécurité : `data/_backups/20260928/eligibility_v2/`
