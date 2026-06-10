"""Extraction des liens ressource depuis le TEI GROBID (couche 2).

Consomme le cache TEI produit par ``grobid_tei.py`` et en tire, par papier, les
candidats URL avec leur zone, leur contexte (phrase porteuse / appel de footnote)
et un score de **pré-filtre** (il ORDONNE les candidats, il n'exclut jamais).
L'arbitrage final — quel lien est le dataset/code officiel — est fait par le batch
Mistral ``link_arbiter`` (voir ``extract_resource_links_pipeline.py``).

Bibliothèque pure : pas de CLI. Point d'entrée : ``extract_links(tei_paths, pdf_paths)``.
"""

import logging
import re
from collections import Counter
from pathlib import Path

from lxml import etree

logger = logging.getLogger(__name__)

TEI_NS = {"t": "http://www.tei-c.org/ns/1.0"}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
# Zones de corps (la footnote est traitée à part pour récupérer sa phrase appelante).
BODY_ZONE_XPATHS: list[tuple[str, str]] = [
    ("abstract", ".//t:profileDesc//t:abstract//t:p"),
    ("body", ".//t:text/t:body//t:p"),
    ("back", ".//t:text/t:back//t:div"),
]

URL_RE = re.compile(r"(?i)\bhttps?://[^\s)\]}>\",;]+")
# Signaux de DISPONIBILITÉ (ODDPub) — uniquement des verbes/locutions de release, JAMAIS les noms
# nus (dataset/corpus/repository/benchmark) : ceux-ci saturent les phrases de méthodo (« trained
# using a dataset », « Experimental Dataset and Splits ») et donnaient +2 à toute dépendance citée.
SCORE_AVAIL_RE = re.compile(
    r"(?i)\b(available|released?|publicly|"
    r"can be downloaded|download(?:able|ed)?|"
    r"we (?:release|make|provide|gather|collect|construct|introduce|present|publish|share))\b"
)
SCORE_NEG_RE = re.compile(
    r"(?i)\b(not (?:publicly )?available|upon request|on request|"
    r"available to (?:researchers|users) who|under (?:a )?(?:data )?agreement|"
    r"was available|no longer available|cannot be (?:shared|released)|we reuse)\b"
)
SCORE_SECTION_RE = re.compile(
    r"(?i)\b(data|code|resource|artifact)s?\s+(availability|release|access)|"
    r"\bavailability\b|\bwe release\b"
)

# Sous-chaînes dans l'URL normalisée — bruit stable, pas des ressources benchmark.
DENY_URL = (
    "doi.org/10.18653",  # DOI proceedings ACL = le papier lui-même
    "doi.org/10.3115",
    "dx.doi.org/10.18653",
    "dx.doi.org/10.3115",
    "doi.org/10.1145/",  # ACM proceedings (idno header)
    "doi.org/10.14288/",  # Sockeye / compute ack
    "creativecommons.org",
    "w3.org",
    "gdpr-info",
    "eur-lex.europa.eu",
    "aclanthology.org",
    "aclweb.org",
    "t.co/",
    "developer.twitter.com",
    "keras.io",
    "pytorch.org",
    "pypi.org",
    "fasttext.cc",
    "nominatim.openstreetmap.org",
    "sentiwordnet",
    "code.google.com/archive",
    "radimrehurek.com",
    "computecanada.ca",
    "westgrid.ca",
    "ethnologue.com",
    "internetlivestats.com",
    "who.int",
    "unctad.org",
    "arabsocialmediareport.com",
    "business-standard.com",
    "www.google.com",
)

KNOWN_HOST = {
    "github": "github",
    "githubusercontent": "github",
    "github.io": "github_pages",
    "huggingface": "hf",
    "hf.co": "hf",
    "zenodo": "zenodo",
    "figshare": "figshare",
    "osf": "osf",
    "codalab": "codalab",
    "kaggle": "kaggle",
    "gitlab": "gitlab",
    "drive.google": "gdrive",
    "dropbox": "dropbox",
    "ldc": "ldc",
    "dataverse": "dataverse",
    "doi.org": "doi",
    "mendeley": "mendeley",
}


# --------------------------------------------------------------------------- #
# Nettoyage du texte PDF (artefacts qui cassent les URLs)                       #
# --------------------------------------------------------------------------- #

_DOI_PATH = r"10\.\d+/[\w./-]+"
_RE_DX_DOI_SPLIT = re.compile(rf"https?://dx\.?\s*doi\.org/({_DOI_PATH})", re.I)
_RE_BARE_DOI = re.compile(rf"(?<![\w/])doi\.org/({_DOI_PATH})", re.I)
_RE_SCHEME_GAP = re.compile(r"https?:\s*/\s*/")
_RE_HYPHEN_BREAK = re.compile(r"-\s*\n\s*")
_RE_SPACED_URL = re.compile(r"https?://(?:[^\s]|\s(?=[a-z0-9~/~._#$-]))+", re.I)
_RE_MULTI_SPACE = re.compile(r"\s+")


def normalize_grobid_text(text: str) -> str:
    """Répare les URLs cassées par l'extraction PDF (l'ordre des passes est intentionnel).

    1. césure ``-\\n`` ; 2. DOI dx coupé (avant le recollage de schéma) ;
    3. schéma ``https: //`` recollé ; 4. DOI nu préfixé ``https://`` ;
    5. tilde PDF / fragment ``$#$`` ; 6. espaces intra-URL supprimés ;
    7. compaction des espaces restants.
    """
    text = _RE_HYPHEN_BREAK.sub("", text)
    text = _RE_DX_DOI_SPLIT.sub(r"https://doi.org/\1", text)
    text = _RE_SCHEME_GAP.sub("https://", text)
    text = _RE_BARE_DOI.sub(r"https://doi.org/\1", text)
    text = text.replace("\u02dc", "~").replace("\u00cb\u0153", "~").replace("$#$", "#")  # tilde PDF (+variante mojibake) -> ~, fragment GROBID -> #
    text = _RE_SPACED_URL.sub(lambda m: _RE_MULTI_SPACE.sub("", m.group(0)), text)
    return _RE_MULTI_SPACE.sub(" ", text)


def _normalize_url(url: str) -> str:
    """https par défaut ; minuscule sur l'hôte seul (paths sensibles à la casse)."""
    url = url.strip().rstrip(".,;:)]}>'\"")
    url = re.sub(r"#{2,}", "#", url)  # double-hash artefact PDF (##frag -> #frag), aligne footnote/pdf_annot
    if not url.lower().startswith("http"):
        url = "https://" + url
    match = re.match(r"(https?://)([^/]+)(/.*)?$", url, re.I)
    if match:
        url = match.group(1).lower() + match.group(2).lower() + (match.group(3) or "")
    if "#" in url or url.endswith((".html", ".htm", ".php", ".json", ".xml")):
        return url
    return url.rstrip("/")


def _host_type(url: str) -> str:
    lower = url.lower()
    for key, label in KNOWN_HOST.items():
        if key in lower:
            return label
    return "other"


def _is_denied(url: str) -> bool:
    lower = url.lower()
    return any(fragment in lower for fragment in DENY_URL)


# --------------------------------------------------------------------------- #
# Candidats + contexte                                                         #
# --------------------------------------------------------------------------- #


def pdf_annotation_links(pdf_path: Path) -> tuple[list[tuple[str, int]], int]:
    """Liens cliquables PDF -> [(url, index_page)] + nb total de pages.

    GROBID rate les hyperliens embarqués (texte surligné, logo, raccourci dans le titre). La page
    est un signal fort : un lien ressource en **page 1** (footnote d'intro « code available at … »)
    est presque toujours celui des auteurs ; les citations cliquables sont disséminées plus loin.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        logger.warning("%s — annotations PDF: %s", pdf_path.name, exc)
        return [], 0
    out: list[tuple[str, int]] = []
    for page_index, page in enumerate(reader.pages):
        for annot in page.get("/Annots") or []:
            try:
                uri = annot.get_object().get("/A", {}).get("/URI")
                if uri and str(uri).lower().startswith("http"):  # écarte mailto:/tel:/file:
                    out.append((str(uri), page_index))
            except Exception:
                continue
    return out, len(reader.pages)


def _sentence_around(text: str, anchor_pos: int) -> str:
    """Phrase portant l'ancre + la phrase précédente (le verbe décisif est souvent avant)."""
    t = normalize_grobid_text(text)
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+", t)] + [len(t)]
    for i in range(1, len(bounds)):
        if bounds[i - 1] <= anchor_pos < bounds[i]:
            start = bounds[i - 2] if i > 1 else bounds[i - 1]
            return t[start : bounds[i]].strip()
    return t[:300]


def _footnote_context(root: etree._Element, note: etree._Element) -> str:
    """Phrase du corps qui appelle la note, accolée au texte de la note."""
    note_text = normalize_grobid_text("".join(note.itertext()))
    nid = note.get(XML_ID)
    if not nid:
        return note_text
    ref = root.find(f".//t:text/t:body//t:ref[@target='#{nid}']", TEI_NS)
    parent = ref.getparent() if ref is not None else None
    if parent is None:
        return note_text
    ptext = normalize_grobid_text("".join(parent.itertext()))
    if not ptext:
        return note_text
    ref_text = normalize_grobid_text("".join(ref.itertext()))
    calling = _sentence_around(ptext, max(ptext.find(ref_text), 0) if ref_text else 0)
    return f"{calling} [note] {note_text}".strip() if calling else note_text


# Coupe une URL extraite du TEXTE là où elle se colle à la phrase suivante sans séparateur.
# Motifs : marqueur de note ``foot_N`` ; ``.`` ou un run de minuscules(>=6) suivi d'une Majuscule
# (``308.The``, ``underanOpen``) = mot de prose. N'altère PAS les vraies URL longues — hôtes en
# minuscules (``comparativeagendas``), ``handbook_2011_version_4.pdf``, GUID hex (``F81005F8-...``)
# n'ont pas ce motif. (Pas de règle chiffre->Majuscule : elle découperait les GUID.)
_RE_GLUE = re.compile(r"foot_\d.*$|(?:\.|(?<=[a-z]{6}))(?=[A-Z]).*$")


def _trim_glued_url(url: str) -> str:
    """Tronque la prose collée à une URL non séparée par un espace (artefact PDF)."""
    return _RE_GLUE.split(url, 1)[0]


def build_candidates(tei_path: Path, pdf_path: Path | None) -> list[dict]:
    """Tous les candidats URL d'un papier (zones TEI + footnotes + idno/ref + annot PDF).

    Un seul parse TEI, dédup par URL via ``seen``. Chaque candidat porte
    ``url`` / ``host_type`` / ``zone`` / ``context`` (le score est ajouté plus tard).
    Les URL issues du TEXTE (body/footnote) sont rognées (``trim``) ; celles issues d'un
    attribut GROBID propre (``ref target`` / DOI) ou des annotations PDF ne le sont pas.
    """
    root = etree.parse(str(tei_path)).getroot()
    cands: list[dict] = []
    seen: set[str] = set()

    def add(url: str, zone: str, context: str, trim: bool = False, first_page: bool | None = None) -> None:
        for piece in re.split(r"(?=https?://)", url):  # sépare les URLs soudées par un artefact PDF
            u = _normalize_url(_trim_glued_url(piece) if trim else piece)
            if len(u) < 15 or _is_denied(u) or u in seen:
                continue
            seen.add(u)
            cand = {"url": u, "host_type": _host_type(u), "zone": zone, "context": context.strip()}
            if first_page is not None:
                cand["first_page"] = first_page
            cands.append(cand)

    for zone, xpath in BODY_ZONE_XPATHS:
        for node in root.findall(xpath, TEI_NS):
            if zone == "body" and node.xpath("boolean(ancestor::t:note)", namespaces=TEI_NS):
                continue  # les notes du corps sont traitées dans la passe footnote
            text = normalize_grobid_text("".join(node.itertext()))
            for m in URL_RE.finditer(text):
                add(m.group(0), zone, _sentence_around(text, m.start()), trim=True)

    for note in root.findall(".//t:note[@place='foot']", TEI_NS):
        ctx = _footnote_context(root, note)
        for m in URL_RE.finditer(normalize_grobid_text("".join(note.itertext()))):
            add(m.group(0), "footnote", ctx, trim=True)

    # idno DOI + ref[@type='url'] : on fait CONFIANCE à l'attribut GROBID (URL propre, frontière
    # déjà calculée depuis l'hyperlien PDF) plutôt qu'au texte visuel collé.
    for idno in root.findall(".//t:idno[@type='DOI']", TEI_NS):
        if doi := (idno.text or "").strip():
            add(f"https://doi.org/{doi}", "metadata", f"TEI idno DOI {doi}")
    for ref in root.findall(".//t:ref[@type='url']", TEI_NS):
        target = (ref.get("target") or "").strip()
        if not target.lower().startswith("http"):  # écarte mailto:/ftp: (emails d'auteurs taggés ref)
            continue
        if ref.xpath("boolean(ancestor::t:note[@place='foot'])", namespaces=TEI_NS):
            zone = "footnote"
        elif ref.xpath("boolean(ancestor::t:back)", namespaces=TEI_NS):
            zone = "back"  # URL dans une entrée de biblio -> pénalisée, pas un lien inline
        else:
            zone = "ref"
        # Contexte = la PHRASE locale autour de ce ref (et non le bloc entier) : plusieurs refs dans
        # un même paragraphe ont chacun leur phrase, pas un contexte partagé dupliqué.
        parent = ref.getparent()
        block = normalize_grobid_text("".join(parent.itertext())) if parent is not None else ""
        ref_text = normalize_grobid_text("".join(ref.itertext()))
        context = _sentence_around(block, max(block.find(ref_text), 0)) if block else ""
        add(target, zone, context)

    if pdf_path is not None and pdf_path.exists():
        annots, npages = pdf_annotation_links(pdf_path)
        for url, page_index in annots:
            add(url, "pdf_annot", f"[lien cliquable PDF, page {page_index + 1}/{npages}]",
                first_page=page_index == 0)

    return cands


# Zones dont l'URL vient d'un attribut GROBID résolu (hyperlien PDF) ou d'une annotation PDF :
# frontière fiable. Une URL « plus profonde » issue du TEXTE n'est alors que de la prose collée.
_AUTHORITATIVE_ZONES = {"ref", "metadata", "pdf_annot"}


def dedup_by_prefix(cands: list[dict]) -> list[dict]:
    """Fusionne les URL dont l'une est le préfixe de l'autre.

    Cas normal (même fiabilité) : on garde la plus PROFONDE (``/geopy`` -> ``/geopy/geocoders``).

    Cas « URL fiable + queue collée » : une URL d'une zone fiable (``ref``/``metadata``/
    ``pdf_annot``, frontière calculée par GROBID/PDF) préfixe une URL du TEXTE. La version texte
    est alors la même ressource avec de la prose collée — que la queue soit un faux ``/segment``
    (``theyworkforyou.com`` vs ``…/underan``) ou des mots soudés (``…/DAICT`` vs
    ``…/DAICTSection5with…``). On garde la propre et on lui TRANSFÈRE le contexte de la collée
    (souvent c'est elle qui porte le verbe de release), puis on jette la collée.
    """
    by_url = {c["url"]: c for c in cands}
    urls = list(by_url)
    drop: set[str] = set()
    for child in urls:
        for parent in urls:
            if parent == child or not child.startswith(parent):
                continue
            rest = child[len(parent):]
            slash_child = child.startswith(parent.rstrip("/") + "/")
            clean_parent = (
                by_url[parent]["zone"] in _AUTHORITATIVE_ZONES
                and by_url[child]["zone"] not in _AUTHORITATIVE_ZONES
            )
            # queue collée à une URL fiable : faux /segment, mot soudé (Majuscule) ou longue prose
            if clean_parent and (slash_child or any(ch.isupper() for ch in rest) or len(rest) >= 12):
                if not by_url[parent]["context"] and by_url[child]["context"]:
                    by_url[parent]["context"] = by_url[child]["context"]
                drop.add(child)
            elif slash_child:
                drop.add(parent)  # cas normal : la plus profonde gagne
    return [c for c in cands if c["url"] not in drop]


# --------------------------------------------------------------------------- #
# Score (pré-filtre, n'exclut jamais) + point d'entrée                          #
# --------------------------------------------------------------------------- #


def compute_url_freq(candidates_by_paper: dict[str, list[dict]]) -> dict[str, int]:
    """Fréquence inter-papiers d'une URL (une URL très partagée = probable dépendance)."""
    counts: Counter[str] = Counter()
    for cands in candidates_by_paper.values():
        for c in cands:
            counts[c["url"]] += 1
    return dict(counts)


def score_candidate(cand: dict, freq: dict[str, int]) -> float:
    """Tri grossier « plausiblement une ressource » vs bruit. Pénalités douces, pas de cutoff."""
    ctx = cand.get("context", "")
    s = 0.0
    if SCORE_AVAIL_RE.search(ctx):
        s += 2.0
    if SCORE_NEG_RE.search(ctx):
        s -= 3.0  # gate négation/restriction (ODDPub) — dégrade, ne supprime pas
    if SCORE_SECTION_RE.search(ctx):
        s += 1.0
    if cand.get("host_type") not in ("other", "doi"):
        s += 1.0
    zone = cand.get("zone", "")
    # ref = URL citée inline, neutre : ce n'est plus le verbe nu « dataset » qui la gonflait (AVAIL_RE
    # resserré) et son contexte est désormais la phrase LOCALE — donc pas besoin de la pénaliser (ça
    # tuait de vrais liens « available at … » que GROBID balise en ref). back/biblio restent pénalisés.
    s += {"footnote": 1.0, "abstract": 1.0, "back": -1.0}.get(zone, 0.0)
    if zone == "pdf_annot":
        # lien cliquable : en page 1 = footnote ressource des auteurs (fort signal) ; ailleurs = souvent
        # une citation cliquable -> neutre, on laisse la fréquence inter-papiers les départager.
        s += 2.0 if cand.get("first_page") else 0.0
    f = freq.get(cand["url"], 1)
    if f > 20:
        s -= 2.0
    elif f > 5:
        s -= 1.0
    if cand["url"].endswith(("/tree", "/blob", "/raw")):
        s -= 0.5  # URL tronquée par le PDF
    return s


def extract_links(
    tei_paths: dict[str, Path | None],
    pdf_paths: dict[str, Path | None] | None = None,
) -> dict[str, list[dict]]:
    """TEI (+ annot PDF) -> candidats scorés et triés par ``anthology_id`` (entrée publique)."""
    pdf_paths = pdf_paths or {}
    raw_by_paper: dict[str, list[dict]] = {}
    for aid, tei_path in tei_paths.items():
        if tei_path is None:
            raw_by_paper[aid] = []
            continue
        try:
            raw_by_paper[aid] = dedup_by_prefix(build_candidates(tei_path, pdf_paths.get(aid)))
        except etree.XMLSyntaxError as exc:
            logger.warning("%s — TEI invalide: %s", aid, exc)
            raw_by_paper[aid] = []

    freq = compute_url_freq(raw_by_paper)
    by_paper = {
        aid: sorted(
            ({**c, "score": score_candidate(c, freq)} for c in raw),
            key=lambda c: (-c["score"], c["url"]),
        )
        for aid, raw in raw_by_paper.items()
    }
    logger.info("Papiers avec >=1 candidat: %d / %d", sum(1 for c in by_paper.values() if c), len(by_paper))
    return by_paper
