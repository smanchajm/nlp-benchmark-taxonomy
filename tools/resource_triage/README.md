# Resource Triage (offline)

Interface locale pour choisir **un lien dataset par papier** (option B : radio + URL manuelle), saisir licence/format/accès/notes, et produire un parquet enrichi.

## Prérequis

1. Pipeline PDF/GROBID + extraction (CSV triage) déjà lancé.
2. Générer la file de travail :

```bash
uv run python tools/resource_triage/prepare_queue.py
```

Produit `data/corpus/resource_links/papers_queue.json` (titre, lien Anthology, candidats, `n_cand=0` inclus).

## Lancer l'UI

1. Ouvrir `tools/resource_triage/index.html` (double-clic), **ou**
2. `python -m http.server` puis `http://localhost:8000/tools/resource_triage/index.html`

Charger `papers_queue.json` (bouton ou glisser-déposer).

## Workflow

- **Gauche** : file des papiers (filtre non revus, `n_cand=0` seulement).
- **Centre** : contexte papier, **abstract** dans une zone scrollable, **un radio** pour le dataset, URLs candidats **cliquables**, URL manuelle, métadonnées (`code_url`, licence, format, accès, notes).
- **Aucune URL de ressource** : premier choix radio (ou bouton **Aucune ressource**) — `dataset_url` vide, `no_dataset_url=true`.
- **Exporter CSV** → `resource_annotations.csv` (auto-save des **annotations** dans `localStorage` — pas `papers_queue.json`, trop gros).
- Après un **refresh** : recharger `papers_queue.json` ; les annotations locales restent.
- Si chargement bloqué : **Reset local** puis recharger le JSON.

Raccourcis : `J` suivant · `K` précédent · `O` ouvrir l'URL ressource · `S` enregistrer & suivant.

## Parquet enrichi

```bash
# Copier le CSV exporté vers data/corpus/resource_links/ (ou --annotations)
uv run python tools/resource_triage/merge_annotations.py
```

Sortie par défaut : `data/corpus/single_task_benchmark_paper_enriched_resources.parquet`

Colonnes ajoutées :

| Colonne | Description |
|---------|-------------|
| `dataset_url` | Lien de la ressource (1 par papier) |
| `no_dataset_url` | `true` si aucun lien dataset retenu |
| `code_url` | Optionnel |
| `resource_license` | ex. CC-BY-4.0 |
| `resource_format` | ex. json, tsv, repo |
| `resource_access` | open, registration, request, unknown |
| `resource_notes` | texte libre |

Pas de `resource_host` (déductible de l'URL). Pas de `resource_review_status` : la progression « à faire » est gérée dans l'UI (`localStorage` + filtre non revus).

## Fichiers

| Fichier | Rôle |
|---------|------|
| `prepare_queue.py` | Parquet + triage CSV → `papers_queue.json` |
| `index.html` / `app.js` / `style.css` | UI offline |
| `merge_annotations.py` | CSV → parquet enrichi |
