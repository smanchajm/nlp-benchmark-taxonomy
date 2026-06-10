# Taxonomy Builder (offline)

Outil local mono-page pour construire manuellement la taxonomie NLP a partir des feuilles HAC, avec audit trail exportable.

## Contraintes respectees

- Pas de SaaS, pas de serveur obligatoire, pas de CDN, pas de build.
- Fichiers versionnables: `tree.json`, `decisions.csv`, `assignments.csv`.
- Offline complet.
- Etat auto-sauvegarde en `localStorage`.

## Lancer

Option 1:

1. Ouvrir `tools/taxonomy_builder/index.html` directement (double clic / `file://`).

Option 2:

1. `python -m http.server`
2. Ouvrir `http://localhost:8000/tools/taxonomy_builder/index.html`

## Entree attendue: `leaves.json`

Liste JSON de:

```json
[
  {
    "id": "leaf_001",
    "label": "stance_detection",
    "description": "Determine whether text expresses support/opposition toward a target.",
    "freq": 12,
    "contexts": ["...", "..."]
  }
]
```

- `id`: identifiant stable feuille
- `label`: nom court de la feuille (affichage principal)
- `description`: description de la feuille (affichage principal)
- `freq`: poids/frequence
- `contexts`: exemples contextuels optionnels

## Fonctionnalites cle

- Panneau gauche: feuilles HAC filtrables, drag source.
- Panneau centre: arbre editable, drag target, avec deux vues:
  - vue liste (DOM)
  - vue D3 (`d3.tree`) verticale avec zoom/pan
  - vue Sunburst D3 (aires proportionnelles au nombre de papiers/freq)
  - option d'affichage des feuilles assignees dans l'arbre (label + description)
- Panneau droite: inspecteur, log decisions, conditions Nickerson live.
- Toute action structurelle est bloquee sans modale de justification (`<= 20 mots`).
- Assignation evidente feuille -> noeud: log silencieux dans `assignments.csv`.
- Option "Marquer assignation frontiere": force une decision modale + ligne `decisions.csv`.
- Marquage feuille "en attente" (reviewee mais non assignee), avec filtre dedie.
- Snapshot/Compare via `localStorage`.
- Round manager:
  - sampling stratifie 30 / 60 / reste
  - filtre par round courant
  - cloture de round + suivi des changements structurels
- Mode C->E guide:
  - requete conceptuelle
  - candidates triees par similarite lexicale
  - assignation directe vers le noeud selectionne
- Export review-ready:
  - `taxonomy_review.html` statique avec arbre, rounds, decisions, snapshot Nickerson

## Import / Export session

- Export: telecharge
  - `tree.json`
  - `tree_with_leaves.json`
  - `decisions.csv`
  - `assignments.csv`
  - `pending.csv`
  - `rounds.csv`
- Export session (bouton `Exporter session`):
  - telecharge les memes artefacts avec prefixe horodate
  - inclut aussi `${prefix}_taxonomy_review.html`
- Export rapide (bouton `Export tree+leaves`):
  - telecharge uniquement `tree_with_leaves.json`
- Import: bouton "Charger session", selectionner les 3 fichiers en meme temps.

## Note de modelisation

L'arbre exporte (`tree.json`) contient uniquement la structure taxonomique:

```json
{ "id": "...", "label": "...", "description": "...", "children": [...] }
```

Les assignations feuille -> noeud sont dans `assignments.csv`.
Les decisions de structure sont dans `decisions.csv`.

## Dependance locale (offline)

- D3 est vendore localement: `tools/taxonomy_builder/vendor/d3.v7.min.js`
- Aucun CDN n'est utilise.

## Conversion HAC -> leaves.json

Utiliser `tools/taxonomy_builder/convert_hac_to_leaves.py`.
