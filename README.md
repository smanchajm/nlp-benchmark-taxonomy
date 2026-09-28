# ClassDex — un index des benchmarks de classification de l'ACL Anthology

Projet de recherche au RALI, DIRO, Université de Montréal.

ClassDex est un inventaire des benchmarks de **classification de texte** publiés dans l'[ACL Anthology](https://aclanthology.org/), interrogeable par tâche, par langue et par domaine. En classification, les jeux de données ne vivent pas dans un registre mais dans la littérature : chaque sous-communauté produit les siens, souvent petits, attachés à un domaine ou à une langue, décrits avec son propre vocabulaire, et le tout se disperse dans plusieurs dizaines de milliers de papiers. Les hubs n'y changent pas grand-chose, parce qu'ils indexent les papiers plutôt que les ressources qu'ils introduisent : dans notre corpus ~75 % des papiers sont référencés sur Papers with Code, mais seuls 18 % de leurs benchmarks y apparaissent comme jeux de données.

Ce dépôt contient deux choses : **la ressource** (1 428 benchmarks, chacun placé dans une taxonomie de tâches construite à partir du corpus, avec domaine, langue et lien vers les données quand il a pu être résolu) et **la méthode** (une cascade de filtrage évaluée, qui n'a rien de spécifique à la classification).


| Lien                   |                                                                                                      |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| Interface de recherche | [https://classdex.samuelmanchajm.fr/](https://classdex.samuelmanchajm.fr/)                           |
| Serveur d'annotation   | [https://labelstudio.samuelmanchajm.fr/](https://labelstudio.samuelmanchajm.fr/)                     |
| Papier (Typst)         | [https://typst.app/project/woL0SlsAcS2cEWUh0d5oTI](https://typst.app/project/woL0SlsAcS2cEWUh0d5oTI) |
| Ressource publiée      | à venir (HF Datasets)                                                                                |




## L'entonnoir


| Étape                 | Ce qui en sort                           | Volume           |
| --------------------- | ---------------------------------------- | ---------------- |
| Corpus                | ACL Anthology, 2013+, 32 venues          | 59 213 papiers   |
| Filtre 1 — SciBERT    | candidats, seuil réglé pour le rappel    | 3 738            |
| Filtre 2 — vote 3 LLM | benchmarks de classification éligibles   | 1 890            |
| Périmètre taxonomie   | mono-tâche, suites multi-tâches écartées | 1 473            |
| Exclusions manuelles  | hors périmètre par règle (8 catégories)  | 1 428            |
| Clustering            | clusters HAC sur paraphrases             | 192              |
| Taxonomie manuelle    | tâches mères / feuilles                  | 25 / 43          |
| Liens ressource       | URL de données résolue                   | 69 % des records |


Le pipeline a tourné de bout en bout : la ressource existe. Ce qui reste, c'est l'évaluation, un dernier batch complet, le nettoyage du code et la rédaction.

## La ressource

Tout est dans une table unique, versionnée dans le dépôt :

```python
df = pd.read_parquet("data/corpus/single_task_benchmark_paper.parquet")
```

L'état du fichier courant est de 1 428 lignes × 27 colonnes


| Colonne                                                          | Contenu                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `bibkey`, `anthology_id`, `url`, `pdf_url`, `doi`                | identité ACL du papier                                                    |
| `title`, `abstract`, `authors`, `year`, `venues`, `venue_type`   | métadonnées Anthology                                                     |
| `numcitedby`                                                     | citations (Semantic Scholar, partiel)                                     |
| `mother_task`, `task`                                            | position dans la taxonomie manuelle (mère / feuille)                      |
| `paraphrase_task`                                                | description normalisée du mécanisme de jugement — le signal de clustering |
| `domain`, `domain_raw`, `paraphrase_domain`, `thematic_domain`   | domaine du benchmark                                                      |
| `benchmark_languages`, `language_evidence`, `language_reasoning` | langue(s), avec la citation du résumé qui la justifie                     |
| `dataset_url`, `dataset_host_type`                               | lien vers les données (69 % des records)                                  |
| `code_url`, `code_host_type`                                     | lien vers le code                                                         |


Quelques conventions à connaître avant d'interpréter les colonnes :

- `und` **ne veut pas dire anglais.** Quand le résumé ne donne pas la langue, on code `und` au lieu de supposer. Les ~43 % de `und` sont une borne basse assumée : un défaut anglais silencieux aurait noyé les benchmarks non anglophones.
- `task` **peut être vide** là où le benchmark s'arrête à une tâche mère.
- `single_task_benchmark_paper_enriched_links.parquet` est la même table avec les liens bruts extraits des PDF, avant nettoyage.



## Le pipeline

Chaque étage lit un fichier et en écrit un autre, sans état conservé d'un notebook au suivant : on peut relancer un étage isolément sans rejouer ceux d'avant.


| #   | Étage                    | Entrée → sortie                                                                                             | Coût     |
| --- | ------------------------ | ----------------------------------------------------------------------------------------------------------- | -------- |
| 0   | Corpus                   | `src/corpus/{fetch,clean,enrich}_anthology.py` → `data/raw/anthology_enriched.parquet`                      | ~1 h     |
| 1   | Jeu d'entraînement       | `src/corpus/regex_buckets.py` + `notebooks/1` → `data/classifier/ready/`                                    | API      |
| 2   | SciBERT                  | `src/classifier/{train,infer}.py` + `configs/` + `slurm/` → `data/classifier/predictions/inference.parquet` | GPU      |
| 3   | Vote LLM + extraction    | `notebooks/2` → `data/taxonomy/per_llm/*` → `merged.parquet`                                                |          |
| 4   | Clustering               | `src/taxonomy/` + `notebooks/3` → `data/taxonomy/hac_clustering/`                                           | ~30 min  |
| 5   | Arbre manuel             | `tools/taxonomy_builder/` → `data/taxonomy/manual_taxonomy_tree.json`                                       | manuel   |
| 6   | Assignation des feuilles | `notebooks/4` → `data/taxonomy/leaf_assignments/`                                                           |          |
| 7   | Liens et couverture      | `src/extraction/` (GROBID), `src/coverage/` + `notebooks/5–6`                                               | h        |
| 8   | Ressource finale         | `src/corpus/build_single_task_benchmark_paper.py` → `data/corpus/single_task_benchmark_paper.parquet`       | ~5 min   |
| 9   | Évaluation               | `src/evaluation/` (Label Studio) + `notebooks/8`                                                            | en cours |


Les notebooks 7 et 8 sont les notebooks d'analyse ; les notebooks 1 à 6 orchestrent les étages et contiennent encore de la logique qui devrait vivre dans `src/` (en cours de nettoyage).  

## Arborescence


| Chemin                                  | Rôle                                                                                          |
| --------------------------------------- | --------------------------------------------------------------------------------------------- |
| `src/corpus/`                           | fetch, clean, enrich de l'Anthology, buckets regex, build final                               |
| `src/classifier/`                       | filtre SciBERT — `train`, `infer`, `model`, plus `configs/` (YAML) et `slurm/` (jobs cluster) |
| `src/taxonomy/`                         | embeddings, utilitaires HAC, providers LLM, schémas structurés                                |
| `src/extraction/`                       | extraction des liens données/code depuis les PDF via GROBID-TEI                               |
| `src/coverage/`                         | appariement aux hubs (PwC, HuggingFace), résolution des identifiants arXiv / S2               |
| `src/evaluation/`                       | pipeline d'annotation Label Studio et échantillonnage du gold                                 |
| `src/paths.py`, `src/logging_config.py` | racine de `data/` et logging partagé                                                          |
| `notebooks/`                            | orchestration des étages, numérotés 1–8 dans l'ordre d'exécution                              |
| `tools/taxonomy_builder/`               | app locale d'une page pour construire la taxonomie à la main (voir son `README.md`)           |
| `scripts/`                              | helpers de synchro avec le cluster (non versionnés : ils portent des identifiants)            |




## Installation

Le projet utilise **uv** et Python 3.13+.

```bash
uv sync
uv run python src/corpus/fetch_anthology.py    # ~1 h, clone le dépôt XML de l'Anthology
```

Les clés d'API vont dans un `.env` à la racine, et ne sont nécessaires que pour relancer les étages concernés :

```
ANTHROPIC_API_KEY=       # vote LLM, assignation des feuilles
MISTRAL_API_KEY=
GOOGLE_API_KEY=
OPENAI_API_KEY=          # optionnel selon le provider utilisé
OPENROUTER_API_KEY=
DEEPSEEK_API_KEY=
GROBID_URL=              # serveur GROBID pour l'extraction des liens
RESOURCE_LINKS_CONTACT_EMAIL=   # pour des appels aux API publiques
```



### Ce qui est dans `data/`

`data/` fait 3,7 Go et est ignoré par défaut, **sauf** les ~16 Mo qui ne se régénèrent pas : annotations humaines, labels votés par les LLM, arbre de taxonomie construit à la main, et les livrables finaux. L'allow-list est commentée dans le [.gitignore](.gitignore). Concrètement, après un `git clone` vous avez déjà :

- `data/corpus/single_task_benchmark_paper.parquet` — la ressource
- `data/taxonomy/manual_taxonomy_tree.json`, `merged.parquet`, `manual_exclusions.parquet`
- `data/classifier/{splits,ready,predictions}/` — splits gelés, labels LLM, inférence
- `data/labelling/`, `data/evaluation/` — annotations et exports/configs Label Studio
- `data/extraction/`, `data/coverage/` — liens extraits, couverture des hubs, caches d'API

Tout le reste se reconstruit en relançant l'étage correspondant.

## Décisions figées

Ces choix sont actés et justifiés dans le papier ; ils appartiennent au socle.

- **Éligibilité.** Une tâche relève de la classification si (c1) son entrée est un texte ou une paire de textes et si (c2) sa sortie est un ou plusieurs labels pris dans un inventaire fini, fixe pour la tâche et identique d'une instance à l'autre. Un papier compte comme ressource si (c3) il introduit des données annotées qu'on peut isoler. Le QA à choix multiples tombe donc hors scope (ses candidats changent à chaque instance), la vérification de faits reste dedans (jugement ternaire fixe).
- **Les suites multi-tâches sont écartées.** Éligibles au sens de la définition, mais on ne peut pas les placer en un nœud unique. C'est ce qui fait passer de 1 890 à 1 473, puis à 1 428 après les exclusions manuelles.
- **La paraphrase est le seul signal de regroupement.** On n'extrait pas « la tâche », ce qui nous rendrait les noms de tâches de la littérature, mais une description normalisée du mécanisme de jugement, débarrassée des noms de familles de tâches, des labels, du domaine et de la cardinalité.

