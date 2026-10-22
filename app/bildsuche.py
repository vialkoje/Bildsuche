#!/usr/bin/env python3
"""
Bildsuche v3 – dreistufige Bildsuche für die Synology.

  Stufe 1  Gesamtbild   kleines mehrsprachiges CLIP-Modell, 1 Vektor pro Bild      (alle Bilder)
  Stufe 2  Kacheln      dasselbe Modell auf 4 überlappenden Bildvierteln            (alle Bilder, wahrscheinlichste zuerst)
  Stufe 3  Prüfung      starkes Modell (SigLIP 2, 384 px) auf Ganzbild + 4 Vierteln (nur die besten Kandidaten je Suche)

Alle Vektoren landen in /data/index.sqlite. Jede Stufe ist unterbrechbar und macht nach einem
Neustart dort weiter, wo sie war. Reports werden laufend aktualisiert:
  "Bestätigt"  = vom starken Modell geprüft und über der Schwelle
  "Vorläufig"  = aus Stufe 1/2, noch nicht geprüft
Suchbegriffe kosten keine Neuberechnung – die Bilder sind schon "angeschaut".
"""
import csv
import hashlib
import html
import io
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import threading
import time
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

DATA_DIR = os.environ.get("DATA", "/data")
READY_MARKER = os.path.join(DATA_DIR, "models", ".ready")
MODEL_NAME = os.environ.get("MODEL", "xlm-roberta-base-ViT-B-32")
MODEL_PRETRAINED = os.environ.get("PRETRAINED", "laion5b_s13b_b90k")


def ready_models():
    try:
        with open(READY_MARKER, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    except OSError:
        return set()


# Offline-Start, sobald das Basismodell lokal liegt (keine Rückfragen bei Hugging Face).
OFFLINE = (os.environ.get("BILDSUCHE_ONLINE") != "1"
           and f"{MODEL_NAME} {MODEL_PRETRAINED}" in ready_models())
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageFile, ImageOps  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pillow_heif = None
try:
    import rawpy
except Exception:  # pragma: no cover
    rawpy = None

VERSION = "2026-10-21e"
CONFIG_PATH = os.environ.get("CONFIG", "/config/searches.yaml")
REPORT_DIR = os.environ.get("REPORTS", "/reports")
THREADS = int(os.environ.get("THREADS", "2"))
WEB_PORT = int(os.environ.get("WEB_PORT", "8091"))   # 0 = Webseite aus

STD_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".gif"}
RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2", ".raf", ".srw", ".pef", ".nrw", ".rwl"}
CROP_LABELS = ["ganzes Bild", "Viertel oben links", "Viertel oben rechts",
               "Viertel unten links", "Viertel unten rechts"]


class _OnlyOwnInfo(logging.Filter):
    def filter(self, record):
        return record.name == "bildsuche" or record.levelno >= logging.WARNING


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
_handler.addFilter(_OnlyOwnInfo())
_log_handlers = [_handler]
LOG_FILE = os.path.join(DATA_DIR, "logs", "bildsuche.log")
try:   # Protokoll zusätzlich als Datei -> auf der Webseite unter /protokoll lesbar, übersteht Neustarts
    import logging.handlers
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    _fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    _fh.setFormatter(_handler.formatter)
    _fh.addFilter(_OnlyOwnInfo())
    _log_handlers.append(_fh)
except OSError:
    pass
logging.basicConfig(level=logging.INFO, handlers=_log_handlers)
for _n in ("httpx", "httpcore", "urllib3", "huggingface_hub", "transformers", "filelock", "timm"):
    logging.getLogger(_n).setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger("bildsuche")
os.umask(0o002)


def _log_uncaught(exc_type, exc, tb):
    log.critical("Unerwarteter Fehler – Programm beendet", exc_info=(exc_type, exc, tb))


sys.excepthook = _log_uncaught


def _log_thread_exc(args):
    if issubclass(args.exc_type, SystemExit):
        return
    log.error("Fehler im Hintergrund-Thread '%s'", getattr(args.thread, "name", "?"),
              exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


threading.excepthook = _log_thread_exc


# ================================================================ Konfiguration
DEFAULTS = {
    "photo_root": "/volume1/photo/Bildarchiv",
    "display_root": "",                  # leer = wie photo_root
    "rescan_interval_hours": 6,
    "report_minutes": 20,
    "batch_size": 16,
    "exclude_dirs": ["@eaDir", "#recycle", "#snapshot", ".@__thumb", "@Recycle"],
    "min_file_kb": 20,
    "default_negatives": ["ein Foto"],
    "skip_raw_with_sibling": True,
    "random_order": True,
    "dedupe_similarity": 0.95,
    "tiles": True,
    "rerank": True,
    "rerank_model": "ViT-B-16-SigLIP2-384",
    "rerank_pretrained": "webli",
    "preview_max": 300,
    # Klickbarer Netzwerk-Link unter jedem Vorschaubild (entspricht photo_root im Container)
    "link_root": "",
    "web_base": "http://diskstation.local:8091",   # für die Bild-Links in der CSV (Adresse der NAS)
    # Synology Photos (Alben auf Knopfdruck). Zugangsdaten stehen in /data/photos.env
    "photos_url": "http://192.168.1.10:5000",   # Adresse der NAS (IP, nicht .local – im Container nicht auflösbar)
    "photos_folder_prefix": "auto",              # Ordner von photo_root im "Freigegebenen Bereich"; auto = aus photo_root
    "photos_share_path": "/volume1/photo",       # NAS-Pfad, der in Photos der "Freigegebene Bereich" ist
    "photos_share_with": [],   # Album wird mit diesen DSM-Benutzern geteilt, z. B. ["anna", "ben"]
    "photos_share_role": "manage",   # manage = bearbeiten | upload | download | view = nur ansehen
    "photos_album_prefix": "Bildsuche: ",
    # Adaptive Kandidatenmenge: Nach der Prüfung der ersten N Kandidaten wird geschaut, wie viele
    # der LETZTEN 500 noch bestätigt wurden. Liegt die Quote über expand_if_rate, gibt es offenbar
    # noch Treffer weiter hinten -> Menge um expand_step vergrößern. Wiederholt sich, bis die
    # Quote abfällt oder candidates_max erreicht ist.
    "expand_if_rate": 0.02,
    "expand_step": 1000,
    "candidates_max": 30000,
}


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return parse_config(cfg)


def parse_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("Die Datei muss aus den Abschnitten 'settings:' und 'searches:' bestehen.")
    user_settings = cfg.get("settings") or {}
    settings = {**DEFAULTS, **user_settings}
    settings["photo_root"] = str(settings["photo_root"]).rstrip("/") or "/"
    disp = str(user_settings.get("display_root") or "").rstrip("/")
    if not os.path.isdir(settings["photo_root"]) and disp and os.path.isdir(disp):
        # Übergang: alte Konfiguration (photo_root: /photos) nach Umstellung der Einbindung auf
        # /volume1/photo -> der echte NAS-Pfad steht in display_root und ist jetzt direkt erreichbar.
        settings["photo_root"] = disp
    if not user_settings.get("display_root") or settings["photo_root"] == disp:
        settings["display_root"] = settings["photo_root"]   # Pfade sind im Container gleich wie auf der NAS
    searches = []
    for s in cfg.get("searches") or []:
        if not s or not s.get("name") or not s.get("terms"):
            continue
        terms = s["terms"] if isinstance(s["terms"], list) else [s["terms"]]
        thr = float(s.get("threshold", 0.5))
        searches.append({
            "name": str(s["name"]).strip().replace("/", "_"),
            "terms": [str(t) for t in terms],
            "negatives": [str(n) for n in (s.get("negatives") or settings["default_negatives"])],
            "threshold": thr,
            "coarse_threshold": float(s.get("coarse_threshold", thr)),
            "candidates": int(s.get("candidates", 2500)),
            "candidates_max": int(s.get("candidates_max", settings.get("candidates_max", 30000))),
            "max_results": int(s.get("max_results", 1000)),
        })
    return settings, searches


def validate_config_text(text):
    """Prüft einen neuen Konfigurationstext, bevor er gespeichert wird. Gibt (ok, Meldung) zurück."""
    try:
        cfg = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" (Zeile {mark.line + 1}, Spalte {mark.column + 1})" if mark else ""
        return False, f"YAML-Fehler{where}: {getattr(e, 'problem', None) or e}"
    try:
        settings, searches = parse_config(cfg or {})
    except Exception as e:
        return False, f"Ungültiger Wert: {e}"
    raw = (cfg or {}).get("searches") or []
    if not os.path.isdir(settings["photo_root"]):
        return False, (f"Bildpfad (photo_root) '{settings['photo_root']}' existiert im Container nicht – "
                       "Tippfehler? Oder der Ordner ist in compose.yaml nicht eingebunden "
                       "(eingebunden ist z. B. /volume1/photo).")
    raw_names = [str(r.get("name", "")).strip() for r in raw if isinstance(r, dict) and r.get("name")]
    bad = [n for n in raw_names if n.startswith(".") or any(c in n for c in '\\/:*?"<>|')]
    if bad:
        return False, "Ungültige Suchnamen (keine Sonderzeichen wie / : * ?): " + ", ".join(bad)
    if not isinstance(raw, list) or not searches:
        return False, "Es ist keine gültige Suche eingetragen (jede braucht 'name' und 'terms')."
    skipped = len(raw) - len(searches)
    names = [q["name"] for q in searches]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        return False, "Doppelte Suchnamen: " + ", ".join(dup)
    msg = f"In Ordnung: {len(searches)} Suchen ({', '.join(names)})."
    if skipped:
        msg += f" Achtung: {skipped} Eintrag/Einträge ohne 'name' oder 'terms' werden ignoriert."
    return True, msg


def save_config_text(text):
    """Speichert die Konfiguration atomar und legt vorher eine Sicherung an (die letzten 10 bleiben)."""
    cfg_dir = os.path.dirname(CONFIG_PATH)
    bdir = os.path.join(cfg_dir, "sicherungen")
    os.makedirs(bdir, exist_ok=True)
    if os.path.exists(CONFIG_PATH):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(CONFIG_PATH, "rb") as a, open(os.path.join(bdir, f"searches-{stamp}.yaml"), "wb") as b:
            b.write(a.read())
        for old in sorted(os.listdir(bdir))[:-10]:
            try:
                os.remove(os.path.join(bdir, old))
            except OSError:
                pass
    if not text.endswith("\n"):
        text += "\n"
    write_atomic(CONFIG_PATH, text)


def config_mtime():
    try:
        return os.path.getmtime(CONFIG_PATH)
    except OSError:
        return 0


def restart_online(reason):
    log.warning("%s – starte neu mit Internetzugang, um fehlende Modelldateien zu laden.", reason)
    env = dict(os.environ, BILDSUCHE_ONLINE="1")
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


# ================================================================ Modelle
class Model:
    def __init__(self, name, pretrained):
        import torch
        import open_clip
        torch.set_num_threads(THREADS)
        self.torch = torch
        self.key = f"{name} {pretrained}"
        if OFFLINE and self.key not in ready_models():
            restart_online(f"Modell {name} liegt noch nicht lokal")
        log.info("Lade Modell %s (%s) …", name, pretrained or "ohne Gewichte")
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained or None, cache_dir=os.path.join(DATA_DIR, "models"))
            self.tokenizer = open_clip.get_tokenizer(name)
        except Exception as e:
            if OFFLINE:
                restart_online(f"Offline-Start von {name} fehlgeschlagen ({e})")
            raise
        self.model.eval()
        scale = getattr(self.model, "logit_scale", None)
        self.logit_scale = float(scale.detach().exp()) if scale is not None else 100.0
        self._text = {}
        self.dim = int(self.encode_text_raw("test").shape[-1])
        if pretrained:
            os.makedirs(os.path.dirname(READY_MARKER), exist_ok=True)
            if self.key not in ready_models():
                with open(READY_MARKER, "a", encoding="utf-8") as f:
                    f.write(self.key + "\n")
        log.info("Modell %s bereit.", name)

    def encode_text_raw(self, text):
        with self.torch.no_grad():
            f = self.model.encode_text(self.tokenizer([text]))
            f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy().astype(np.float32)

    def embed_text(self, text):
        if text not in self._text:
            self._text[text] = self.encode_text_raw(text)
        return self._text[text]

    def text_matrix(self, search):
        pos = np.mean([self.embed_text(t) for t in search["terms"]], axis=0)
        pos /= np.linalg.norm(pos)
        return np.vstack([pos] + [self.embed_text(n) for n in search["negatives"]])

    def embed_images(self, pil_images, bs=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(pil_images), bs):
                batch = self.torch.stack([self.preprocess(im) for im in pil_images[i:i + bs]])
                f = self.model.encode_image(batch)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.cpu().numpy().astype(np.float32))
        return np.vstack(out)


def pos_prob(E, T, scale):
    """Wahrscheinlichkeit, dass Vektoren E eher zum Suchbegriff (T[0]) passen als zu den Gegenkategorien."""
    logits = scale * (E @ T.T)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p[..., 0] / p.sum(axis=-1)


# ================================================================ Bilder
def load_image(path, max_side=1024):
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        if rawpy is None:
            raise RuntimeError("rawpy nicht verfügbar")
        with rawpy.imread(path) as raw:
            try:
                thumb = raw.extract_thumb()
                im = (Image.open(io.BytesIO(thumb.data)) if thumb.format == rawpy.ThumbFormat.JPEG
                      else Image.fromarray(thumb.data))
            except Exception:
                im = Image.fromarray(raw.postprocess(half_size=True, use_camera_wb=True))
    else:
        im = Image.open(path)
        im.draft("RGB", (max_side, max_side))
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def safe_load(path):
    try:
        return load_image(path), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:300]


def quadrants(im, frac=0.6):
    w, h = im.size
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    return [im.crop(b) for b in ((0, 0, cw, ch), (w - cw, 0, w, ch), (0, h - ch, cw, h), (w - cw, h - ch, w, h))]


def padded(im):
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    return canvas


# ================================================================ Datenbank
def index_file(root):
    """Eigene Index-Datei pro Bildpfad – ein Wechsel des Pfads legt einen weiteren Index an,
    ein Zurückwechseln nutzt den alten weiter."""
    root = root.rstrip("/") or "/"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", root.strip("/"))[:60] or "root"
    h = hashlib.sha1(root.encode("utf-8")).hexdigest()[:8]
    return os.path.join(DATA_DIR, "indexes", f"{slug}_{h}.sqlite")


class DB:
    """Index eines Bildpfads. Gespeichert werden Pfade RELATIV zum Bildpfad – im Programm absolut."""

    def __init__(self, path, root):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.root = root.rstrip("/")
        self.file = path
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS images(
            path TEXT PRIMARY KEY, mtime REAL, size INTEGER, emb BLOB, error TEXT)""")
        self.db.execute("CREATE TABLE IF NOT EXISTS tiles(path TEXT PRIMARY KEY, emb BLOB, error TEXT)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS rerank(
            path TEXT PRIMARY KEY, model TEXT, emb BLOB, error TEXT)""")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('root', ?)", (self.root,))
        self.db.commit()

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass

    def rel(self, p):
        return p[len(self.root):].lstrip("/") if p.startswith(self.root + "/") else p

    def abs(self, r):
        return self.root + "/" + r

    def rows(self, sql):
        """Wie execute(), aber die erste Spalte (Pfad) kommt absolut zurück."""
        for row in self.db.execute(sql):
            yield (self.abs(row[0]),) + tuple(row[1:])

    def known(self):
        return {self.abs(p): (m, s) for p, m, s in self.db.execute("SELECT path, mtime, size FROM images")}

    def count(self):
        return self.db.execute("SELECT COUNT(*) FROM images WHERE emb IS NOT NULL").fetchone()[0]

    def drop(self, paths):
        rows = [(self.rel(p),) for p in paths]
        for t in ("images", "tiles", "rerank"):
            self.db.executemany(f"DELETE FROM {t} WHERE path=?", rows)
        self.db.commit()

    def put(self, table, rows):
        q = {"images": "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?)",
             "tiles": "INSERT OR REPLACE INTO tiles VALUES (?,?,?)",
             "rerank": "INSERT OR REPLACE INTO rerank VALUES (?,?,?,?)"}[table]
        self.db.executemany(q, [(self.rel(r[0]),) + tuple(r[1:]) for r in rows])
        self.db.commit()


def list_indexes():
    """Alle vorhandenen Indizes: [(Bildpfad, Anzahl Bilder, Datei)]."""
    out = []
    d = os.path.join(DATA_DIR, "indexes")
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not fn.endswith(".sqlite"):
            continue
        try:
            con = sqlite3.connect(f"file:{os.path.join(d, fn)}?mode=ro", uri=True, timeout=2)
            root = (con.execute("SELECT value FROM meta WHERE key='root'").fetchone() or ["?"])[0]
            n = con.execute("SELECT COUNT(*) FROM images WHERE emb IS NOT NULL").fetchone()[0]
            con.close()
            out.append((root, n, fn))
        except Exception:
            continue
    return out


def migrate_legacy_index(settings):
    """Einmalig: alter Gesamt-Index (data/index.sqlite, Pfade unter /photos) -> Index pro Bildpfad."""
    legacy = os.path.join(DATA_DIR, "index.sqlite")
    if not os.path.exists(legacy):
        return
    target_root = str(settings.get("legacy_index_root") or "/volume1/photo/Bildarchiv").rstrip("/")
    prefix = str(settings.get("legacy_index_prefix") or "/photos").rstrip("/")
    target = index_file(target_root)
    if os.path.exists(target):
        log.info("Alter Index wird nicht übernommen – für %s gibt es schon einen Index.", target_root)
        return
    log.info("Übernehme alten Index (%s/…) als Index für %s – einmalig, dauert einen Moment …",
             prefix, target_root)
    new = DB(target, target_root)
    new.db.execute("ATTACH DATABASE ? AS old", (legacy,))
    n0 = len(prefix) + 2
    like = prefix + "/%"
    for t, cols in (("images", "mtime, size, emb, error"), ("tiles", "emb, error"),
                    ("rerank", "model, emb, error")):
        new.db.execute(f"INSERT OR REPLACE INTO main.{t} SELECT substr(path, {n0}), {cols} "
                       f"FROM old.{t} WHERE path LIKE ?", (like,))
    new.db.commit()
    new.db.execute("DETACH DATABASE old")
    n = new.count()
    new.close()
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(legacy + ext):
            os.replace(legacy + ext, legacy + ".uebernommen" + ext)
    log.info("Alter Index übernommen: %d Bilder für %s.", n, target_root)


def scan(settings):
    root = settings["photo_root"]
    excl = set(settings["exclude_dirs"])
    min_size = int(settings["min_file_kb"]) * 1024
    skip_raw = bool(settings.get("skip_raw_with_sibling", True))
    found, raw_skipped = {}, 0
    t0 = last = time.time()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excl and not d.startswith(".")]
        if time.time() - last > 30:
            last = time.time()
            log.info("Scan läuft … %d Bilder bisher (%s)", len(found), dirpath[len(root):] or "/")
        stems = ({os.path.splitext(f)[0].lower() for f in filenames
                  if os.path.splitext(f)[1].lower() in STD_EXT} if skip_raw else set())
        for fn in filenames:
            if fn.startswith("."):
                continue
            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext not in STD_EXT and ext not in RAW_EXT:
                continue
            if ext in RAW_EXT and stem.lower() in stems:
                raw_skipped += 1
                continue
            full = os.path.join(dirpath, fn)
            try:
                st = os.stat(full)
            except OSError:
                continue
            if st.st_size >= min_size:
                found[full] = (st.st_mtime, st.st_size)
    if raw_skipped:
        log.info("%d RAW-Dateien übersprungen (JPG/HEIC gleichen Namens vorhanden).", raw_skipped)
    log.info("Scan: %d Bilder gefunden (%.0fs).", len(found), time.time() - t0)
    return found


# ================================================================ Arbeitsspeicher-Index
class Store:
    """Alle Vektoren im RAM (float16). Index i ist für alle drei Stufen gleich."""

    def __init__(self, found, db, dim, rerank_key):
        self.found = found
        self.paths = sorted(found)
        self.idx = {p: i for i, p in enumerate(self.paths)}
        n = len(self.paths)
        self.G = np.zeros((n, dim), np.float16)
        self.has_g = np.zeros(n, bool)
        self.err_g = np.zeros(n, bool)
        self.T = np.zeros((n, 4, dim), np.float16)
        self.has_t = np.zeros(n, bool)
        self.err_t = np.zeros(n, bool)
        self.R = {}            # i -> (5, dim_r) float32
        self.err_r = set()
        t0 = time.time()
        for p, e, err in db.rows("SELECT path, emb, error FROM images"):
            i = self.idx.get(p)
            if i is None:
                continue
            if e is not None:
                self.G[i] = np.frombuffer(e, np.float16)
                self.has_g[i] = True
            elif err:
                self.err_g[i] = True
        for p, e, err in db.rows("SELECT path, emb, error FROM tiles"):
            i = self.idx.get(p)
            if i is None:
                continue
            if e is not None:
                self.T[i] = np.frombuffer(e, np.float16).reshape(4, dim)
                self.has_t[i] = True
            elif err:
                self.err_t[i] = True
        for p, model, e, err in db.rows("SELECT path, model, emb, error FROM rerank"):
            i = self.idx.get(p)
            if i is None or model != rerank_key:
                continue
            if e is not None:
                v = np.frombuffer(e, np.float16).astype(np.float32)
                self.R[i] = v.reshape(5, -1)
            elif err:
                self.err_r.add(i)
        log.info("Index geladen: %d Gesamtbild, %d Kacheln, %d geprüft (%.0fs).",
                 int(self.has_g.sum()), int(self.has_t.sum()), len(self.R), time.time() - t0)

    def coarse(self, T, scale, chunk=20000):
        """Grobscore je Bild = max(Gesamtbild, beste Kachel) + welche Ansicht ihn geliefert hat."""
        n = len(self.paths)
        score = np.zeros(n, np.float32)
        crop = np.zeros(n, np.int8)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            g = pos_prob(self.G[a:b].astype(np.float32), T, scale)
            g[~self.has_g[a:b]] = 0
            t = pos_prob(self.T[a:b].astype(np.float32), T, scale)       # (m, 4)
            t[~self.has_t[a:b]] = 0
            tb = t.argmax(axis=1)
            tm = t[np.arange(b - a), tb]
            use_t = tm > g
            score[a:b] = np.where(use_t, tm, g)
            crop[a:b] = np.where(use_t, tb + 1, 0)
        return score, crop


# ================================================================ Engine
KNOWN_POOL = {}
CURRENT_ROOT = [""]


class Engine:
    def __init__(self, base, db, store, settings, searches):
        self.base, self.db, self.store = base, db, store
        self.settings, self.searches = settings, searches
        self.strong = None
        self.pool = ThreadPoolExecutor(max_workers=max(1, THREADS))
        s = store
        self.todo_g = [i for i in range(len(s.paths)) if not s.has_g[i] and not s.err_g[i]]
        if settings.get("random_order", True):
            random.shuffle(self.todo_g)
        self.todo_t, self.todo_r = [], []
        self.cand_dirty = True
        self.cand_time = 0
        self.tiles_since_refresh = 0
        self.coarse_cache = {}
        self.known_pool = KNOWN_POOL          # bleibt über Neu-Scans erhalten -> keine Wiederholungen im Log
        self.stats = {"g": [0, time.time()], "t": [0, time.time()], "r": [0, time.time()]}
        self.last_progress = 0

    # ---------------- Kandidaten & Prioritäten
    def final_scores(self, q):
        """Score des starken Modells für alle bereits geprüften Bilder."""
        s = self.store
        final_s, final_crop = {}, {}
        if s.R and self.settings.get("rerank", True):
            if self.strong is None:
                self.strong = Model(self.settings["rerank_model"], self.settings["rerank_pretrained"])
            Ts = self.strong.text_matrix(q)
            ids = list(s.R)
            p = pos_prob(np.stack([s.R[i] for i in ids]), Ts, self.strong.logit_scale)  # (k, 5)
            for i, row in zip(ids, p):
                final_s[i] = float(row.max())
                final_crop[i] = int(row.argmax())
        return final_s, final_crop

    def pool_size(self, q, order, score):
        """Kandidatenmenge adaptiv vergrößern, solange am Ende der Liste noch Treffer auftauchen."""
        s = self.store
        size = q["candidates"]
        rate_min = float(self.settings.get("expand_if_rate", 0.02))
        step = int(self.settings.get("expand_step", 1000))
        cap = int(q.get("candidates_max") or self.settings.get("candidates_max", 30000))
        n_valid = int((score > 0).sum())
        if rate_min <= 0 or not s.R:
            return min(size, n_valid)
        final_s = None
        while size < min(cap, n_valid):
            tail = [int(i) for i in order[max(0, size - 500):size]]
            if any(i not in s.R and i not in s.err_r for i in tail):
                break                                   # Ende der Liste noch nicht geprüft
            if final_s is None:
                final_s, _ = self.final_scores(q)
            hits = sum(1 for i in tail if final_s.get(i, 0) >= q["threshold"])
            rate = hits / max(1, len(tail))
            if rate < rate_min:
                break
            size = min(size + step, cap)
            if size > self.known_pool.get(q["name"], 0):
                self.known_pool[q["name"]] = size
                log.info("Suche '%s': %d von %d der letzten Kandidaten bestätigt (%.1f %%) – "
                         "erweitere Prüfung auf die besten %d Bilder.", q["name"], hits, len(tail),
                         100 * rate, size)
        return min(size, cap, n_valid)

    def refresh_candidates(self):
        s = self.store
        self.coarse_cache = {}
        self.pool_sizes = {}
        cand_rank = {}
        prio = np.zeros(len(s.paths), np.float32)
        for q in self.searches:
            T = self.base.text_matrix(q)
            score, crop = s.coarse(T, self.base.logit_scale)
            self.coarse_cache[q["name"]] = (score, crop)
            prio = np.maximum(prio, score)
            full_order = np.argsort(-score)
            size = self.pool_size(q, full_order, score)
            self.pool_sizes[q["name"]] = size
            for r, i in enumerate(full_order[:size]):
                if score[i] > 0:
                    cand_rank[int(i)] = min(cand_rank.get(int(i), 10 ** 9), r)
        self.todo_r = [i for i, _ in sorted(cand_rank.items(), key=lambda kv: kv[1])
                       if i not in s.R and i not in s.err_r]
        self.candidates = set(cand_rank)
        if self.settings.get("tiles", True):
            pending = np.where(s.has_g & ~s.has_t & ~s.err_t)[0]
            self.todo_t = list(pending[np.argsort(-prio[pending])])
        self.cand_dirty = False
        self.cand_time = time.time()
        self.tiles_since_refresh = 0
        log.info("Kandidaten aktualisiert: %d zu prüfen, %d Bilder warten auf Kacheln.",
                 len(self.todo_r), len(self.todo_t))

    # ---------------- Aufgaben
    def next_task(self):
        if self.todo_g:
            bs = int(self.settings["batch_size"])
            batch, self.todo_g = self.todo_g[:bs], self.todo_g[bs:]
            if not self.todo_g:
                self.cand_dirty = True
            return "g", batch
        if self.cand_dirty or (self.tiles_since_refresh >= 2000) or \
                (self.tiles_since_refresh and time.time() - self.cand_time > 3600):
            self.refresh_candidates()
        if self.settings.get("rerank", True) and self.todo_r:
            batch, self.todo_r = self.todo_r[:2], self.todo_r[2:]
            return "r", batch
        if self.settings.get("tiles", True) and self.todo_t:
            batch, self.todo_t = self.todo_t[:4], self.todo_t[4:]
            return "t", batch
        if time.time() - self.cand_time > 5:
            # Nichts mehr zu tun? Einmal neu bewerten – Kacheln oder eine größere
            # Kandidatenmenge können neue Prüfaufträge ergeben haben.
            self.refresh_candidates()
            return self.next_task()
        return None, None

    def run(self, kind, batch):
        s = self.store
        paths = [s.paths[i] for i in batch]
        loaded = list(self.pool.map(safe_load, paths))
        good = [(i, im) for i, (im, err) in zip(batch, loaded) if im is not None]
        bad = [(i, err) for i, (im, err) in zip(batch, loaded) if im is None]

        if kind == "g":
            rows = []
            if good:
                embs = self.base.embed_images([im for _, im in good])
                for (i, _), e in zip(good, embs):
                    s.G[i] = e
                    s.has_g[i] = True
                    m, z = s.found[s.paths[i]]
                    rows.append((s.paths[i], m, z, e.astype(np.float16).tobytes(), None))
            for i, err in bad:
                s.err_g[i] = True
                m, z = s.found[s.paths[i]]
                rows.append((s.paths[i], m, z, None, err))
            self.db.put("images", rows)

        elif kind == "t":
            rows = []
            if good:
                crops = [c for _, im in good for c in quadrants(im)]
                embs = self.base.embed_images(crops).reshape(len(good), 4, -1)
                for (i, _), e in zip(good, embs):
                    s.T[i] = e
                    s.has_t[i] = True
                    rows.append((s.paths[i], e.astype(np.float16).tobytes(), None))
            for i, err in bad:
                s.err_t[i] = True
                rows.append((s.paths[i], None, err))
            self.db.put("tiles", rows)
            self.tiles_since_refresh += len(batch)

        elif kind == "r":
            if self.strong is None:
                self.strong = Model(self.settings["rerank_model"], self.settings["rerank_pretrained"])
            key = self.strong.key
            rows = []
            if good:
                crops = [c for _, im in good for c in [padded(im)] + quadrants(im)]
                embs = self.strong.embed_images(crops, bs=5).reshape(len(good), 5, -1)
                for (i, _), e in zip(good, embs):
                    s.R[i] = e
                    rows.append((s.paths[i], key, e.astype(np.float16).tobytes(), None))
            for i, err in bad:
                s.err_r.add(i)
                rows.append((s.paths[i], key, None, err))
            self.db.put("rerank", rows)

        self.stats[kind][0] += len(batch)
        self.progress()

    # ---------------- Fortschritt
    def counts(self):
        s = self.store
        n = len(s.paths)
        g_done = int((s.has_g | s.err_g).sum())
        t_done = int((s.has_t | s.err_t).sum())
        cand = getattr(self, "candidates", set())
        r_done = len([i for i in cand if i in s.R or i in s.err_r])
        return n, g_done, t_done, len(cand), r_done

    def progress(self, force=False):
        if not force and time.time() - self.last_progress < 120:
            return
        self.last_progress = time.time()
        n, g, t, c, r = self.counts()

        def rate(k, remaining):
            done, t0 = self.stats[k]
            rps = done / max(1.0, time.time() - t0)
            eta = f", Rest ca. {remaining / rps / 3600:.1f} h" if rps > 0 and remaining else ""
            return f"{rps:.2f}/s{eta}" if done else "–"
        log.info("Fortschritt | Gesamtbild %d/%d (%s) | Kacheln %d/%d (%s) | Geprüft %d/%d (%s)",
                 g, n, rate("g", n - g), t, n, rate("t", n - t), r, c, rate("r", c - r))

    # ---------------- Reports
    def report(self, final):
        s = self.store
        os.makedirs(REPORT_DIR, exist_ok=True)
        if not s.has_g.any():
            log.info("Noch keine Bilder im Index – keine Reports.")
            return
        # Immer frisch rechnen: Seit dem letzten Report sind neue Vektoren dazugekommen.
        # (Kostet nur Sekunden – die teure Arbeit steckt in den Vektoren, nicht im Vergleich.)
        self.refresh_candidates()
        summary = []
        for q in self.searches:
            t0 = time.time()
            score_c, crop_c = self.coarse_cache.get(q["name"]) or s.coarse(
                self.base.text_matrix(q), self.base.logit_scale)
            # Stufe 3 – geprüfte Kandidaten
            final_s, final_crop = self.final_scores(q)
            # Keine Kappung mehr: CSV enthält ALLE Treffer, die HTML-Seite zeigt die besten als Bild.
            confirmed = sorted((i for i, v in final_s.items() if v >= q["threshold"]),
                               key=lambda i: -final_s[i])
            rejected = sorted((i for i, v in final_s.items() if v < q["threshold"]
                               and score_c[i] >= q["coarse_threshold"]), key=lambda i: -score_c[i])
            order = np.argsort(-score_c)
            prelim = []
            for i in order:
                if score_c[i] < q["coarse_threshold"]:
                    break
                if int(i) not in final_s and int(i) not in s.err_r:
                    prelim.append(int(i))
            write_report(q, s, confirmed, prelim, rejected, final_s, final_crop, score_c, crop_c,
                         self.settings, self.counts(), final, self.pool_sizes.get(q["name"]))
            summary.append((q["name"], len(confirmed), len(prelim)))
            log.info("Report '%s': %d bestätigt, %d vorläufig, %d verworfen (%.0fs)",
                     q["name"], len(confirmed), len(prelim), len(rejected), time.time() - t0)
        write_status(self.counts(), summary, final)


# ================================================================ Report-Dateien
def display_path(path, settings):
    root = settings["photo_root"].rstrip("/")
    return settings["display_root"].rstrip("/") + path[len(root):] if path.startswith(root) else path


def view_link(search, path, settings, absolute=False):
    """Link zum Anzeigen des Originalbilds im Browser (über die Webseite des Containers)."""
    root = settings["photo_root"].rstrip("/")
    if not path.startswith(root):
        return ""
    rel = path[len(root):].lstrip("/")
    q = "/bild?s=" + urllib.parse.quote(search) + "&p=" + urllib.parse.quote(rel)
    return (settings.get("web_base", "").rstrip("/") + q) if absolute else q


def smb_link(path, settings, folder=False):
    """smb://-Link auf die Datei (oder ihren Ordner), Sonderzeichen URL-kodiert."""
    base = (settings.get("link_root") or "").rstrip("/")
    root = settings["photo_root"].rstrip("/")
    if not base or not path.startswith(root):
        return ""
    rel = path[len(root):]
    if folder:
        rel = os.path.dirname(rel) + "/"
    return base + urllib.parse.quote(rel, safe="/")


def thumb_for(path):
    tdir = os.path.join(DATA_DIR, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    out = os.path.join(tdir, hashlib.sha1(path.encode("utf-8")).hexdigest() + ".jpg")
    if not os.path.exists(out):
        try:
            load_image(path, max_side=240).save(out, "JPEG", quality=80)
        except Exception:
            return None
    return out


def group_near_duplicates(hits, G, thr):
    rep_of, reps = {}, []
    for i in hits:
        if thr > 0 and reps:
            sims = G[reps].astype(np.float32) @ G[i].astype(np.float32)
            j = int(np.argmax(sims))
            if sims[j] >= thr:
                rep_of[i] = reps[j]
                continue
        rep_of[i] = i
        reps.append(i)
    return rep_of


def write_report(q, s, confirmed, prelim, rejected, final_s, final_crop, score_c, crop_c,
                 settings, counts, final, pool=None):
    out_dir = os.path.join(REPORT_DIR, q["name"])
    tdir = os.path.join(out_dir, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    thr = float(settings.get("dedupe_similarity") or 0)
    rep_c = group_near_duplicates(confirmed, s.G, thr)
    rep_p = group_near_duplicates(prelim, s.G, thr)

    def fmt(x):
        return f"{x:.3f}".replace(".", ",")

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path + ".tmp", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["status", "rang", "score_geprueft", "score_grob", "fundstelle", "pfad", "ordner",
                    "datei", "aehnlich_wie_rang", "link"])
        for status, lst, rep in (("bestätigt", confirmed, rep_c), ("vorläufig", prelim, rep_p)):
            rank_of = {i: r for r, i in enumerate(lst, 1)}
            for r, i in enumerate(lst, 1):
                dp = display_path(s.paths[i], settings)
                fs = final_s.get(i)
                where = CROP_LABELS[final_crop[i]] if fs is not None else CROP_LABELS[crop_c[i]]
                w.writerow([status, r, fmt(fs) if fs is not None else "", fmt(score_c[i]), where, dp,
                            os.path.dirname(dp), os.path.basename(dp),
                            "" if rep[i] == i else rank_of[rep[i]],
                            view_link(q["name"], s.paths[i], settings, absolute=True)])
        for i in rejected:
            dp = display_path(s.paths[i], settings)
            w.writerow(["verworfen", "", fmt(final_s[i]), fmt(score_c[i]), "", dp,
                        os.path.dirname(dp), os.path.basename(dp), "",
                        view_link(q["name"], s.paths[i], settings, absolute=True)])


    os.replace(csv_path + ".tmp", csv_path)
    wanted = set()

    root = settings["photo_root"].rstrip("/")

    def rel(i):
        return s.paths[i][len(root):].lstrip("/")

    def cards(lst, rep, use_final, limit):
        dup, members = {}, {}
        for i in lst:
            members.setdefault(rep[i], []).append(rel(i))
            if rep[i] != i:
                dup[rep[i]] = dup.get(rep[i], 0) + 1
        out = []
        for r, i in enumerate(lst, 1):
            if rep[i] != i:
                continue
            if len(out) >= limit:
                break
            src = thumb_for(s.paths[i])
            img = ""
            if src:
                name = os.path.basename(src)
                wanted.add(name)
                dst = os.path.join(tdir, name)
                if not os.path.exists(dst):
                    with open(src, "rb") as a, open(dst, "wb") as b:
                        b.write(a.read())
                img = f'<img loading="lazy" src="thumbs/{name}" alt="">'
            sc = final_s[i] if use_final else score_c[i]
            where = CROP_LABELS[final_crop[i]] if use_final else CROP_LABELS[crop_c[i]]
            more = f' · <i>+{dup[i]} ähnliche</i>' if dup.get(i) else ""
            dp = html.escape(display_path(s.paths[i], settings))
            f_link = html.escape(view_link(q["name"], s.paths[i], settings), quote=True)
            links = f'<br><a class="open" href="{f_link}">Bild öffnen</a>' if f_link else ""
            grp = [rel(i)] + [m for m in members.get(i, []) if m != rel(i)]   # gezeigtes Bild zuerst
            data_p = html.escape("|".join(grp), quote=True)
            out.append(f'<figure data-p="{data_p}">{img}<figcaption><b>#{r}</b> · {sc:.2f} · {where}{more}'
                       f'<br><span title="{dp}">{dp}</span>{links}</figcaption></figure>')
        return out

    c_cards = cards(confirmed, rep_c, True, q["max_results"])
    c_note = (f" – die besten {len(c_cards)} Motive als Bild, alle in results.csv"
              if len(c_cards) >= q["max_results"] else "")
    p_cards = cards(prelim, rep_p, False, int(settings.get("preview_max", 300)))
    for old in os.listdir(tdir):  # veraltete Vorschauen aufräumen
        if old not in wanted:
            try:
                os.remove(os.path.join(tdir, old))
            except OSError:
                pass

    n, g, t, c, r = counts
    pct = lambda a, b: f"{100 * a / b:.0f} %" if b else "–"  # noqa: E731
    now = datetime.now().strftime("%d.%m.%Y %H:%M")
    state = "" if final else '<p class="warn">Zwischenstand – Berechnung läuft noch, der Report wird laufend besser.</p>'
    terms = ", ".join(html.escape(x) for x in q["terms"])
    p_note = (f" – die besten {len(p_cards)} von {len(prelim)} gezeigt, alle in results.csv"
              if len(prelim) > len(p_cards) else "")
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bildsuche: {html.escape(q['name'])}</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#222}}
@media (prefers-color-scheme:dark){{body{{background:#1c1c1e;color:#eee}}figure{{background:#2c2c2e!important}}}}
h1{{margin:0 0 4px}} h2{{margin:28px 0 4px}} .meta{{color:#888;font-size:14px;line-height:1.5}}
.warn{{color:#b8860b;font-weight:600}} .hint{{color:#888;font-size:13px;margin:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-top:12px}}
figure{{margin:0;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px #0002}}
figure img{{width:100%;height:180px;object-fit:cover;display:block;background:#ddd}}
figcaption{{padding:6px 8px;font-size:12px;word-break:break-all}}
figcaption a{{color:#2a6fdb;text-decoration:none}} figcaption a:hover{{text-decoration:underline}}
.prelim figure{{opacity:.85}}
.album{{margin:16px 0;padding:12px 14px;border-radius:8px;background:#eef3fb;font-size:14px}}
@media (prefers-color-scheme:dark){{.album{{background:#23304a}}}}
.simopt{{display:block;margin-top:8px;font-size:13px;cursor:pointer}}
.album button{{font:inherit;padding:6px 12px;margin-right:8px;border-radius:6px;border:1px solid #2a6fdb;background:#2a6fdb;color:#fff;cursor:pointer}}
.album button.sec{{background:transparent;color:#2a6fdb}} .album button.del{{background:transparent;color:#c0392b;border-color:#c0392b}} .album button:disabled{{opacity:.5;cursor:default}}
#albummsg{{margin-top:8px;white-space:pre-line}}
.album input[type=text]{{font:inherit;padding:5px 8px;border-radius:6px;border:1px solid #aab;min-width:280px;margin:0 8px 8px 0}}
.selbar{{position:sticky;top:0;z-index:5;padding:8px 0;background:inherit}}
figure{{cursor:pointer;position:relative;outline:3px solid transparent;transition:outline-color .1s}}
figure.sel{{outline-color:#2a6fdb}}
figure.sel::after{{content:"✓";position:absolute;top:6px;right:6px;width:26px;height:26px;border-radius:13px;
  background:#2a6fdb;color:#fff;font-weight:700;display:flex;align-items:center;justify-content:center}}
</style></head><body>
<p class="hint"><a href="../index.html">← Übersicht</a> · Version {VERSION}</p>
<h1>{html.escape(q['name'])}</h1>
<div class="meta">Begriffe: {terms}<br>
Bildpfad {html.escape(settings['photo_root'])} · Schwelle geprüft {q['threshold']:.2f} · Schwelle vorläufig {q['coarse_threshold']:.2f} · Stand {now}<br>
Fortschritt: Gesamtbild {g}/{n} ({pct(g, n)}) · Kacheln {t}/{n} ({pct(t, n)}) · Geprüft {r}/{c} Kandidaten ({pct(r, c)})</div>
{state}
{album_box(q["name"])}
<h2>Bestätigt – {len(c_cards)} Motive</h2>
<p class="hint">Vom starken Modell geprüft. {len(confirmed)} Treffer inkl. ähnlicher{c_note}.
Geprüft werden die besten {pool or q['candidates']} Bilder dieser Suche – die Menge wächst automatisch, solange am Ende noch Treffer auftauchen.</p>
<div class="grid">{''.join(c_cards)}</div>
<h2>Vorläufig – noch nicht geprüft</h2>
<p class="hint">Aus Gesamtbild und Kacheln{p_note}. Diese Bilder werden nach und nach geprüft.</p>
<div class="grid prelim">{''.join(p_cards)}</div>
{LIGHTBOX}
</body></html>"""
    write_atomic(os.path.join(out_dir, "report.html"), page)


LIGHTBOX = r"""<div id="lb" hidden>
<div id="lbbar"><button id="lbx" aria-label="Schließen">✕ Schließen</button>
<span id="lbn"></span>
<button id="lbsel">Auswählen</button></div>
<button id="lbp" class="lbnav" aria-label="Vorheriges Bild">‹</button>
<div id="lbwrap"><div id="lbload">Bild wird geladen …</div><img id="lbimg" alt=""></div>
<button id="lbnx" class="lbnav" aria-label="Nächstes Bild">›</button>
<div id="lbcap"></div>
</div>
<style>
#lb{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.92);color:#eee;display:flex;align-items:center;justify-content:center}
#lb[hidden]{display:none}
#lbbar{position:absolute;top:0;left:0;right:0;display:flex;gap:10px;align-items:center;padding:10px 12px;
  padding-top:max(10px, env(safe-area-inset-top))}
#lbbar button{font:inherit;font-size:15px;padding:8px 14px;border-radius:8px;border:1px solid #666;background:#222;color:#eee;cursor:pointer}
#lbsel.on{background:#2a6fdb;border-color:#2a6fdb;color:#fff}
#lbn{flex:1;text-align:center;font-size:14px;color:#aaa}
#lbwrap{max-width:calc(100vw - 120px);max-height:calc(100vh - 140px);display:flex;align-items:center;justify-content:center}
#lbimg{max-width:calc(100vw - 120px);max-height:calc(100vh - 140px);object-fit:contain;border-radius:4px}
#lbload{position:absolute;color:#aaa;font-size:14px}
.lbnav{position:absolute;top:50%;transform:translateY(-50%);font-size:44px;line-height:1;width:52px;height:84px;
  border:none;border-radius:10px;background:rgba(255,255,255,.08);color:#fff;cursor:pointer}
#lbp{left:6px} #lbnx{right:6px}
#lbcap{position:absolute;bottom:0;left:0;right:0;padding:10px 16px;padding-bottom:max(10px, env(safe-area-inset-bottom));
  font-size:13px;color:#ccc;text-align:center;word-break:break-word}
@media (max-width:700px){#lbwrap,#lbimg{max-width:100vw} .lbnav{display:none}}
</style>
<script>
(function(){
  const lb = document.getElementById('lb'), img = document.getElementById('lbimg'), cap = document.getElementById('lbcap'),
        sel = document.getElementById('lbsel'), num = document.getElementById('lbn'), load = document.getElementById('lbload');
  let cur = -1, x0 = null;
  const links = () => Array.from(document.querySelectorAll('a.open'));
  function fig(){ const a = links()[cur]; return a ? a.closest('figure') : null; }
  function paintSel(){ const f = fig(); const on = !!(f && f.classList.contains('sel'));
    sel.textContent = on ? 'Ausgewählt ✓' : 'Auswählen'; sel.classList.toggle('on', on); }
  function show(i){
    const a = links(); if (!a.length) return;
    cur = (i + a.length) % a.length;
    const f = a[cur].closest('figure');
    load.style.display = 'block'; img.style.visibility = 'hidden';
    img.src = a[cur].href;
    const fc = f ? f.querySelector('figcaption') : null;
    cap.textContent = fc ? fc.innerText.replace(/\s*Bild öffnen\s*/g, '').replace(/\n+/g, ' · ') : '';
    num.textContent = (cur + 1) + ' / ' + a.length;
    paintSel();
    if (lb.hidden) { lb.hidden = false; document.body.style.overflow = 'hidden';
      try { history.pushState({lb: 1}, ''); } catch (e) {} }
  }
  function hideNow(){ lb.hidden = true; img.removeAttribute('src'); document.body.style.overflow = ''; }
  function close(){ if (history.state && history.state.lb) history.back(); else hideNow(); }
  img.addEventListener('load', () => { load.style.display = 'none'; img.style.visibility = 'visible'; });
  img.addEventListener('error', () => { load.textContent = 'Bild konnte nicht geladen werden.'; });
  window.addEventListener('popstate', () => { if (!lb.hidden) hideNow(); });
  document.addEventListener('click', ev => {
    const a = ev.target.closest('a.open'); if (!a) return;
    ev.preventDefault(); ev.stopPropagation(); load.textContent = 'Bild wird geladen …';
    show(links().indexOf(a));
  }, true);
  document.getElementById('lbx').onclick = close;
  document.getElementById('lbp').onclick = () => show(cur - 1);
  document.getElementById('lbnx').onclick = () => show(cur + 1);
  sel.onclick = () => { const f = fig(); if (!f) return; f.classList.toggle('sel');
    if (typeof update === 'function') update(); paintSel(); };
  lb.addEventListener('click', ev => { if (ev.target === lb || ev.target.id === 'lbwrap') close(); });
  document.addEventListener('keydown', ev => { if (lb.hidden) return;
    if (ev.key === 'Escape') close(); else if (ev.key === 'ArrowLeft') show(cur - 1); else if (ev.key === 'ArrowRight') show(cur + 1); });
  lb.addEventListener('touchstart', ev => { x0 = ev.touches[0].clientX; }, {passive: true});
  lb.addEventListener('touchend', ev => { if (x0 === null) return; const dx = ev.changedTouches[0].clientX - x0; x0 = null;
    if (Math.abs(dx) > 50) show(cur + (dx < 0 ? 1 : -1)); });
})();
</script>"""


def write_atomic(path, text):
    """Erst in eine Hilfsdatei schreiben, dann umbenennen – so liefert die Webseite nie halbe Dateien aus."""
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def write_status(counts, summary, final):
    n, g, t, c, r = counts
    lines = [f"Stand: {datetime.now():%d.%m.%Y %H:%M}",
             f"Status: {'fertig, wartet auf neue Bilder' if final else 'Berechnung läuft'}",
             f"Stufe 1 Gesamtbild: {g}/{n}",
             f"Stufe 2 Kacheln:    {t}/{n}",
             f"Stufe 3 Geprüft:    {r}/{c} Kandidaten", ""]
    lines += [f"{name}: {a} bestätigt, {b} vorläufig  ->  {name}/report.html" for name, a, b in summary]
    write_atomic(os.path.join(REPORT_DIR, "_status.txt"), "\n".join(lines) + "\n")
    write_index(counts, summary, final)


def write_index(counts, summary, final):
    """Übersichtsseite reports/index.html – im Browser über die Webseite oder als Datei."""
    n, g, t, c, r = counts
    pct = lambda a, b: f"{100 * a / b:.0f} %" if b else "–"  # noqa: E731
    pw = lambda a, b: f"{100 * a / b:.1f}%" if b else "0%"  # noqa: E731
    idx_rows = "".join(
        f'<tr><td>{html.escape(r)}{" <b>(aktiv)</b>" if r == CURRENT_ROOT[0] else ""}</td>'
        f'<td class="num">{n}</td></tr>' for r, n, _ in list_indexes()) or '<tr><td colspan="2">–</td></tr>'
    rows = "".join(
        f'<tr><td><a href="{urllib.parse.quote(name)}/report.html">{html.escape(name)}</a></td>'
        f'<td class="num">{a}</td><td class="num">{b}</td></tr>' for name, a, b in summary)
    active = {name for name, _, _ in summary}
    orphans = []
    try:
        for d in sorted(os.listdir(REPORT_DIR)):
            csv_p = os.path.join(REPORT_DIR, d, "results.csv")
            if d in active or d.startswith(".") or not os.path.isfile(csv_p):
                continue
            try:
                with open(csv_p, encoding="utf-8-sig") as f:
                    n_ok = sum(1 for line in f if line.startswith("bestätigt;"))
            except OSError:
                n_ok = 0
            orphans.append((d, datetime.fromtimestamp(os.path.getmtime(csv_p)), n_ok))
    except OSError:
        pass
    old_rows = "".join(
        f'<tr id="old-{html.escape(d, quote=True)}"><td><a href="{urllib.parse.quote(d)}/report.html">{html.escape(d)}</a></td>'
        f'<td>{ts:%d.%m.%Y %H:%M}</td><td class="num">{k}</td>'
        f'<td class="num"><button class="del" data-name="{html.escape(d, quote=True)}" onclick="delReport(this)">Löschen</button></td></tr>'
        for d, ts, k in orphans)
    old_block = (f"""<h2 style="font-size:16px;margin-top:28px">Ältere Reports</h2>
<p class="meta">Reports von Suchen, die es in der Konfiguration nicht mehr gibt (z. B. nach dem Umbenennen).
Sie werden nicht mehr aktualisiert. „Löschen“ entfernt nur den Report-Ordner – Index und Bilder bleiben unberührt.</p>
<table><tr><th>Report</th><th>Stand</th><th class="num">bestätigt</th><th></th></tr>{old_rows}</table>
<p class="meta" id="oldmsg"></p>""" if orphans else "")
    state = ("fertig – wartet auf neue Bilder" if final
             else "Berechnung läuft – die Reports werden laufend besser")
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="300">
<title>Bildsuche</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#222;max-width:900px}}
@media (prefers-color-scheme:dark){{body{{background:#1c1c1e;color:#eee}}table{{background:#2c2c2e!important}}td,th{{border-color:#444!important}}}}
h1{{margin:0 0 6px}} .meta{{color:#888;font-size:14px;line-height:1.6}}
table{{border-collapse:collapse;width:100%;margin-top:18px;background:#fff;border-radius:8px;overflow:hidden}}
th,td{{padding:10px 12px;border-bottom:1px solid #e5e5e5;text-align:left}} th{{font-size:13px;color:#888;font-weight:600}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
a{{color:#2a6fdb;text-decoration:none}} a:hover{{text-decoration:underline}}
.bar{{height:6px;background:#ddd;border-radius:3px;overflow:hidden;margin:2px 0 8px;max-width:420px}}
.bar>div{{height:100%;background:#2a6fdb}}
button.del{{font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid #c0392b;background:transparent;color:#c0392b;cursor:pointer}}
</style></head><body>
<h1>Bildsuche</h1>
<p><a href="/config">⚙︎ Konfiguration ansehen &amp; bearbeiten</a> · <a href="/protokoll">☰ Protokoll</a></p>
<div class="meta">Bildpfad: <b>{html.escape(CURRENT_ROOT[0])}</b> · Stand {datetime.now():%d.%m.%Y %H:%M} · {state}<br>
Stufe 1 Gesamtbild: {g} / {n} ({pct(g, n)})<div class="bar"><div style="width:{pw(g, n)}"></div></div>
Stufe 2 Kacheln: {t} / {n} ({pct(t, n)})<div class="bar"><div style="width:{pw(t, n)}"></div></div>
Stufe 3 Geprüft: {r} / {c} Kandidaten ({pct(r, c)})<div class="bar"><div style="width:{pw(r, c)}"></div></div>
</div>
<table><tr><th>Suche</th><th class="num">bestätigt</th><th class="num">vorläufig</th></tr>{rows}</table>
<h2 style="font-size:16px;margin-top:28px">Vorhandene Indizes</h2>
<p class="meta">Pro Bildpfad wird ein eigener Index geführt. Wechselst du in der Konfiguration <code>photo_root</code>,
wird für neue Pfade ein weiterer Index angelegt; beim Zurückwechseln wird der vorhandene weiterbenutzt.</p>
<table><tr><th>Bildpfad</th><th class="num">berechnete Bilder</th></tr>{idx_rows}</table>
{old_block}
<script>
async function delReport(b){{
  const name = b.dataset.name, msg = document.getElementById('oldmsg');
  if (location.protocol === 'file:') {{ msg.textContent = 'Das geht nur über die Webseite.'; return; }}
  if (!confirm('Alten Report „' + name + '“ löschen?\\n\\nEntfernt wird nur der Report-Ordner, nicht der Index und keine Bilder.')) return;
  b.disabled = true;
  try {{
    const r = await fetch('/api/report/delete?search=' + encodeURIComponent(name), {{method: 'POST'}});
    const j = await r.json(); msg.textContent = j.message;
    if (r.ok) {{ const row = document.getElementById('old-' + name); if (row) row.remove(); }} else b.disabled = false;
  }} catch (e) {{ msg.textContent = 'Fehler: ' + e; b.disabled = false; }}
}}
</script>
</body></html>"""
    write_atomic(os.path.join(REPORT_DIR, "index.html"), page)


def album_box(name):
    """Album-Steuerung: alle bestätigten Treffer oder eine Auswahl per Klick in ein Photos-Album.
    Funktioniert nur, wenn der Report über die Webseite geöffnet ist."""
    n = html.escape(name, quote=True)
    default_title = html.escape(DEFAULTS["photos_album_prefix"] + name, quote=True)
    return f"""<div class="album">
<b>Synology Photos – Album</b><br>
Bilder durch Anklicken auswählen (Umschalt-Klick wählt einen ganzen Bereich).
Existiert das Album schon, werden die Bilder hinzugefügt.<br><br>
<input type="text" id="albumtitle" value="{default_title}" aria-label="Albumname">
<div class="selbar">
<button onclick="albumJob('all')">Alle bestätigten hinzufügen</button>
<button onclick="albumJob('sel')" id="selbtn" disabled>Auswahl hinzufügen (0)</button>
<button class="sec" onclick="selAll(true)">Alle bestätigten auswählen</button>
<button class="sec" onclick="selAll(false)">Auswahl aufheben</button>
<button class="sec" onclick="albumJob('test')">Probelauf (5 Bilder)</button>
<button class="del" onclick="albumJob('delete')">Album löschen</button>
</div>
<label class="simopt"><input type="checkbox" id="withsim" onchange="update()"> ähnliche Bilder mitnehmen
(„+n ähnliche“ – sonst nur das gezeigte Bild)</label>
<div id="albummsg"></div></div>
<script>
const SEARCH = "{n}", KEY = "bildsuche-sel-" + SEARCH;
let last = null;
function figs(){{ return Array.from(document.querySelectorAll('figure[data-p]')); }}
function withSim(){{ const c = document.getElementById('withsim'); return !!(c && c.checked); }}
function selPaths(){{ return figs().filter(f => f.classList.contains('sel'))
  .flatMap(f => withSim() ? f.dataset.p.split('|') : [f.dataset.p.split('|')[0]]); }}
function save(){{ try {{ localStorage.setItem(KEY, JSON.stringify(figs().filter(f => f.classList.contains('sel')).map(f => f.dataset.p))); }} catch(e) {{}} }}
function update(){{
  const n = selPaths().length;
  const b = document.getElementById('selbtn'); b.textContent = 'Auswahl hinzufügen (' + n + ')'; b.disabled = !n; save();
}}
function selAll(on){{   // "Alle auswählen" = alle bestätigten; Vorläufige nur per Einzelklick
  figs().forEach(f => {{ if (!on || !f.closest('.prelim')) f.classList.toggle('sel', on); }}); update(); }}
document.addEventListener('click', ev => {{
  if (ev.target.closest('a') || ev.target.closest('.album')) return;
  const f = ev.target.closest('figure[data-p]'); if (!f) return;
  const all = figs();
  if (ev.shiftKey && last) {{
    const [a, b] = [all.indexOf(last), all.indexOf(f)].sort((x, y) => x - y);
    const on = !f.classList.contains('sel');
    all.slice(a, b + 1).forEach(x => x.classList.toggle('sel', on));
  }} else f.classList.toggle('sel');
  last = f; update();
}});
document.addEventListener('DOMContentLoaded', () => {{
  try {{ const saved = new Set(JSON.parse(localStorage.getItem(KEY) || '[]'));
        figs().forEach(f => {{ if (saved.has(f.dataset.p)) f.classList.add('sel'); }}); }} catch(e) {{}}
  update();
}});
function msg(t){{ document.getElementById('albummsg').textContent = t; }}
function busy(b){{ document.querySelectorAll('.album button').forEach(x => x.disabled = b); if (!b) update(); }}
async function albumJob(mode){{
  if (location.protocol === 'file:') {{ msg('Das geht nur über die Webseite (Port 8091 der NAS).'); return; }}
  const title = document.getElementById('albumtitle').value.trim();
  if (!title) {{ msg('Bitte einen Albumnamen eingeben.'); return; }}
  let paths = null;
  if (mode === 'sel') paths = selPaths();
  const what = mode === 'all' ? (withSim() ? 'alle bestätigten Bilder (mit ähnlichen)' : 'alle bestätigten Motive (ohne ähnliche)') : mode === 'sel' ? paths.length + ' ausgewählte Bilder' : '5 Bilder (Probelauf)';
  if (mode === 'delete') {{
    if (!confirm('Album „' + title + '“ in Synology Photos löschen?\\n\\nEs wird nur das Album gelöscht, keine Fotos. Das geht nur bei Alben, die die Bildsuche angelegt hat.')) return;
  }} else if (mode !== 'test' && !confirm(what + ' zum Album „' + title + '“ hinzufügen?')) return;
  busy(true); msg('Auftrag wird gestartet …');
  try {{
    const r = await fetch('/api/album?search=' + encodeURIComponent(SEARCH), {{method:'POST',
      headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{mode, title, paths, similar: withSim()}})}});
    const j = await r.json(); msg(j.message);
    if (!r.ok) {{ busy(false); return; }}
    poll();
  }} catch(e) {{ msg('Fehler: ' + e); busy(false); }}
}}
async function poll(){{
  try {{
    const j = await (await fetch('/api/album/status?search=' + encodeURIComponent(SEARCH))).json();
    msg(j.message);
    if (j.running) setTimeout(poll, 2000); else busy(false);
  }} catch(e) {{ msg('Fehler: ' + e); busy(false); }}
}}
</script>"""


class PhotosError(Exception):
    def __init__(self, msg, code=None):
        super().__init__(msg)
        self.code = code


class PhotosClient:
    """Minimaler Client für die (inoffizielle) Synology-Photos-Schnittstelle. Nur Lesen + Alben."""

    AUTH_ERRORS = {400: "Benutzername oder Passwort falsch",
                   401: "Konto ist deaktiviert",
                   402: "keine Berechtigung für diese Anmeldung",
                   403: "2-Faktor-Code verlangt – für den Hilfsbenutzer abschalten",
                   404: "2-Faktor-Code falsch",
                   406: "2-Faktor-Anmeldung ist Pflicht – für den Hilfsbenutzer ausnehmen",
                   407: "IP-Adresse gesperrt (zu viele Fehlversuche) – in DSM unter Sicherheit → Schutz entsperren",
                   408: "Passwort abgelaufen – neu setzen",
                   409: "Passwort abgelaufen – neu setzen",
                   410: "Passwort muss geändert werden – einmal im Browser anmelden und ändern"}

    def __init__(self, base_url, user, password):
        import http.cookiejar
        import ssl
        import urllib.request
        ctx = ssl.create_default_context()
        ctx.check_hostname = False              # NAS im Heimnetz mit selbstsigniertem Zertifikat
        ctx.verify_mode = ssl.CERT_NONE
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
            urllib.request.HTTPSHandler(context=ctx))
        self.sid = self.token = None
        root = base_url.rstrip("/")
        # Wichtig: session="webui" – mit anderen Sitzungsnamen (z. B. "SynologyPhotos") antwortet DSM
        # mit Fehler 402, sogar für Administratoren. Nach der Anmeldung laufen alle Aufrufe über entry.cgi.
        last = None
        for login_path, call_path, version in (
                ("/webapi/auth.cgi", "/webapi/entry.cgi", 6),
                ("/webapi/entry.cgi", "/webapi/entry.cgi", 6),
                ("/photo/webapi/auth.cgi", "/photo/webapi/entry.cgi", 3)):
            self.base = root + login_path
            try:
                r = self.call("SYNO.API.Auth", "login", version=version, account=user, passwd=password,
                              format="sid", enable_syno_token="yes", session="webui")
                self.sid, self.token = r.get("sid"), r.get("synotoken")
                self.auth_version = version
                self.base = root + call_path
                log.info("Photos-Anmeldung über %s erfolgreich.", login_path)
                return
            except PhotosError as e:
                last = e
                log.info("Anmeldung über %s: %s", login_path, e)
                if e.code in (400, 401, 407, 408, 409, 410):   # weitere Versuche würden nur sperren
                    break
            except Exception as e:                              # z. B. 404, wenn der Pfad nicht existiert
                last = PhotosError(f"{login_path}: {e}")
                log.info("Anmeldung über %s nicht möglich: %s", login_path, e)
        code = getattr(last, "code", None)
        hint = self.AUTH_ERRORS.get(code)
        raise PhotosError(f"Anmeldung bei Synology Photos fehlgeschlagen: {hint or last}"
                          + (f" (Fehlercode {code})" if code else ""))

    def call(self, api, method, version=1, **params):
        import json as _json
        import urllib.request
        q = {"api": api, "method": method, "version": version}
        for k, v in params.items():
            q[k] = v if isinstance(v, str) else _json.dumps(v)
        if self.sid:
            q["_sid"] = self.sid
        req = urllib.request.Request(self.base, data=urllib.parse.urlencode(q).encode())
        if self.token:
            req.add_header("X-SYNO-TOKEN", self.token)
        with self.opener.open(req, timeout=120) as resp:
            j = _json.loads(resp.read().decode("utf-8"))
        if not j.get("success"):
            code = j.get("error", {}).get("code")
            raise PhotosError(f"{api}.{method}: Fehlercode {code}", code=code)
        return j.get("data") or {}

    def logout(self):
        try:
            self.call("SYNO.API.Auth", "logout", version=getattr(self, "auth_version", 6), session="webui")
        except Exception:
            pass

    def paged(self, api, key="list", **params):
        out, off = [], 0
        while True:
            d = self.call(api, "list", offset=off, limit=1000, **params)
            lst = d.get(key) or []
            out += lst
            if len(lst) < 1000:
                return out
            off += 1000


_REPORT_PATHS = {}


def report_paths(search):
    """Alle Bildpfade (relativ zu photo_root) aus results.csv einer Suche – zwischengespeichert."""
    csv_path = os.path.join(REPORT_DIR, search, "results.csv")
    try:
        mt = os.path.getmtime(csv_path)
    except OSError:
        return set()
    cached = _REPORT_PATHS.get(search)
    if cached and cached[0] == mt:
        return cached[1]
    settings, _ = load_config()
    disp_root = settings["display_root"].rstrip("/")
    paths = set()
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["pfad"].startswith(disp_root):
                paths.add(row["pfad"][len(disp_root):].lstrip("/"))
    _REPORT_PATHS[search] = (mt, paths)
    return paths


ALBUM_JOBS = {}
ALBUM_LOCK = threading.Lock()


def read_photos_env():
    path = os.path.join(DATA_DIR, "photos.env")
    env = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env.get("PHOTOS_USER"), env.get("PHOTOS_PASS")


def album_job(name, mode="all", title=None, selected=None, similar=False):
    """Läuft im Hintergrund: Treffer aus results.csv (alle bestätigten oder eine Auswahl) -> Photos-Album.
    mode: "all" | "sel" | "test". Existiert das Album, werden die Bilder hinzugefügt (Probelauf: ersetzt)."""
    test = mode == "test"
    job = ALBUM_JOBS[name]

    def say(t):
        job["message"] = t
        log.info("Album '%s': %s", name, t)

    try:
        settings, _ = load_config()
        user, pw = read_photos_env()
        if not user or not pw:
            raise PhotosError("Keine Zugangsdaten: Datei docker/bildsuche/data/photos.env fehlt "
                              "(PHOTOS_USER=… / PHOTOS_PASS=…).")
        if mode == "delete":
            title = (title or "").strip()
            if not title:
                raise PhotosError("Bitte einen Albumnamen angeben.")
            say(f"Suche Album „{title}“ …")
            pc = PhotosClient(settings["photos_url"], user, pw)
            try:
                # Browse.Album listet nur die eigenen Alben des Benutzers bildsuche;
                # fremde Alben (anderer Benutzer) tauchen hier nicht auf und kann Photos ihm auch nicht löschen lassen.
                own = [a for a in pc.paged("SYNO.Foto.Browse.Album") if a.get("name") == title]
                if not own:
                    raise PhotosError(f"Kein Album „{title}“ gefunden, das die Bildsuche angelegt hat. "
                                      f"Eigene Alben anderer Benutzer kann sie nicht löschen.")
                pc.call("SYNO.Foto.Browse.Album", "delete", id=[a["id"] for a in own])
                say(f"Fertig: Album „{title}“ gelöscht (die Fotos selbst bleiben erhalten). "
                    f"({datetime.now():%d.%m. %H:%M})")
            finally:
                pc.logout()
            return
        disp_root = settings["display_root"].rstrip("/")
        rels, known = [], set()
        with open(os.path.join(REPORT_DIR, name, "results.csv"), encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter=";"):
                if not row["pfad"].startswith(disp_root):
                    continue
                r = row["pfad"][len(disp_root):].lstrip("/")
                known.add(r)
                if row.get("status") == "bestätigt" and (similar or not row.get("aehnlich_wie_rang")):
                    rels.append(r)
        if mode == "sel":
            # nur Pfade übernehmen, die wirklich in diesem Report stehen
            rels = list(dict.fromkeys(p for p in (selected or []) if p in known))
        if test:
            rels = rels[:5]
        if not rels:
            raise PhotosError("Keine Bilder ausgewählt." if mode == "sel" else "Keine bestätigten Treffer in dieser Suche.")
        say(f"Melde mich bei Synology Photos an … ({len(rels)} Bilder)")
        pc = PhotosClient(settings["photos_url"], user, pw)
        try:
            # Ordner-IDs im Freigegebenen Bereich über den Pfad finden (Baum Ebene für Ebene)
            prefix = str(settings.get("photos_folder_prefix") or "auto")
            if prefix == "auto":
                share = settings.get("photos_share_path", "/volume1/photo").rstrip("/")
                if not settings["photo_root"].startswith(share):
                    raise PhotosError(f"Bildpfad {settings['photo_root']} liegt nicht im Freigegebenen Bereich "
                                      f"({share}) – Alben gehen nur für Bilder aus Synology Photos.")
                prefix = settings["photo_root"][len(share):] or "/"
            prefix = "/" + prefix.strip("/")
            children = {}

            def folder_id(path):
                parent, cur = 0, ""
                for part in path.strip("/").split("/"):
                    cur = cur + "/" + part
                    if parent not in children:
                        children[parent] = {f["name"]: f["id"] for f in
                                            pc.paged("SYNO.FotoTeam.Browse.Folder", id=parent)}
                    if cur not in children[parent]:
                        return None
                    parent = children[parent][cur]
                return parent

            by_dir = {}
            for r in rels:
                d, fn = os.path.split(r)
                by_dir.setdefault(d, []).append(fn)
            ids, missing, miss_names = [], 0, []
            for k, (d, files) in enumerate(sorted(by_dir.items()), 1):
                if k % 25 == 0:
                    say(f"Suche Bilder in Photos … Ordner {k}/{len(by_dir)}")
                fid = folder_id(prefix + ("/" + d if d else ""))
                if fid is None:
                    missing += len(files)
                    miss_names += [f"{d}/{fn} (Ordner in Photos nicht sichtbar)" for fn in files]
                    continue
                lst = pc.paged("SYNO.FotoTeam.Browse.Item", folder_id=fid,
                               sort_by="filename", sort_direction="asc")
                exact = {it["filename"]: it["id"] for it in lst}
                lower = {it["filename"].lower(): it["id"] for it in lst}
                # Photos fasst RAW+JPG und Live-Fotos zu EINEM Eintrag zusammen -> Abgleich über den Namensstamm
                stems = {}
                for it in lst:
                    stems.setdefault(os.path.splitext(it["filename"])[0].lower(), []).append(it["id"])
                for fn in files:
                    iid = exact.get(fn) or lower.get(fn.lower())
                    if iid is None:
                        cand = stems.get(os.path.splitext(fn)[0].lower()) or []
                        iid = cand[0] if len(cand) == 1 else None
                    if iid is not None:
                        ids.append(iid)
                    else:
                        missing += 1
                        miss_names.append(f"{d}/{fn}")
            ids = list(dict.fromkeys(ids))
            if miss_names:
                log.info("Album '%s': nicht in Photos gefunden (%d): %s", name, len(miss_names),
                         "; ".join(miss_names[:200]))
            if not ids:
                raise PhotosError(f"Keines der {len(rels)} Bilder in Synology Photos gefunden – ist "
                                  f"'{prefix}' im Freigegebenen Bereich indiziert?")
            if test:
                title = "Bildsuche-Test: " + name
            title = (title or (settings["photos_album_prefix"] + name)).strip()[:200]
            existing = next((a for a in pc.paged("SYNO.Foto.Browse.Album") if a.get("name") == title), None)
            album, added_to = {}, False
            if existing and test:
                pc.call("SYNO.Foto.Browse.Album", "delete", id=[existing["id"]])
                existing = None
            if existing:
                say(f"Füge {len(ids)} Bilder zum vorhandenen Album „{title}“ hinzu …")
                try:
                    pc.call("SYNO.Foto.Browse.NormalAlbum", "add_item", id=existing["id"], item=ids)
                    added_to = True
                except PhotosError as e:
                    # Fallback: Inhalt lesen, Album mit der Vereinigungsmenge neu anlegen
                    log.info("add_item nicht möglich (%s) – lege Album mit allen Bildern neu an.", e)
                    old = [it["id"] for it in pc.paged("SYNO.Foto.Browse.Item", album_id=existing["id"])]
                    ids = list(dict.fromkeys(old + ids))
                    pc.call("SYNO.Foto.Browse.Album", "delete", id=[existing["id"]])
            if not added_to:
                say(f"Lege Album „{title}“ mit {len(ids)} Bildern an …")
                album = pc.call("SYNO.Foto.Browse.NormalAlbum", "create",
                                name=json.dumps(title, ensure_ascii=False),   # API erwartet den Namen als JSON-String
                                item=ids).get("album", {})
            shared = ""
            targets = settings.get("photos_share_with") or []
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(",") if t.strip()]
            album_id = album.get("id") or (existing or {}).get("id")
            if targets and album_id:
                try:
                    lst = pc.call("SYNO.Foto.Sharing.Misc", "list_user_group",
                                  team_space_sharable_list=False).get("list", [])
                    uids = {u.get("name"): u["id"] for u in lst if u.get("type", "user") == "user"}
                    found = [t for t in targets if t in uids]
                    unknown = [t for t in targets if t not in uids]
                    if found:
                        pp = pc.call("SYNO.Foto.Sharing.Passphrase", "set_shared", policy="album",
                                     album_id=album_id, enabled=True).get("passphrase")
                        # gewünschte Rolle zuerst, bei Ablehnung schrittweise schwächer
                        order = ["manage", "upload", "download", "view"]
                        want = str(settings.get("photos_share_role") or "manage").strip().lower()
                        roles = [want] + [r for r in order[order.index(want) + 1:]] if want in order else [want] + order
                        used, last_err = None, None
                        for role in roles:
                            try:
                                pc.call("SYNO.Foto.Sharing.Passphrase", "update", passphrase=pp, expiration=0,
                                        permission=[{"role": role, "action": "update",
                                                     "member": {"type": "user", "id": uids[t]}} for t in found])
                                used = role
                                break
                            except PhotosError as e:
                                last_err = e
                                log.info("Teilen mit Rolle '%s' abgelehnt (%s)", role, e)
                        if used is None:
                            raise last_err or PhotosError("Teilen nicht möglich")
                        names = {"manage": "Verwalten", "upload": "Hochladen", "download": "Herunterladen",
                                 "view": "Anzeigen"}
                        shared = f", geteilt mit {' und '.join(found)} (Rolle: {names.get(used, used)})"
                    if unknown:
                        shared += f" – Benutzer nicht gefunden: {', '.join(unknown)}"
                except Exception as e:
                    shared = f" – Teilen fehlgeschlagen ({e})"
            miss = ""
            if missing:
                ex = ", ".join(miss_names[:3]) + (" …" if len(miss_names) > 3 else "")
                miss = (f". {missing} nicht in Photos gefunden: {ex} – meist noch nicht von Photos "
                        f"indiziert oder für den Benutzer {user} nicht lesbar (neue Ordner?). Liste im Protokoll")
            n_txt = "1 Bild" if len(ids) == 1 else f"{len(ids)} Bilder"
            what = (f"{n_txt} zum Album „{title}“ hinzugefügt (schon enthaltene bleiben einfach drin)"
                    if added_to else f"Album „{title}“ mit {'1 Bild' if len(ids) == 1 else f'{len(ids)} Bildern'} angelegt")
            say(f"Fertig: {what}{shared}{miss}. "
                f"({datetime.now():%d.%m. %H:%M})")
        finally:
            pc.logout()
    except Exception as e:
        say(f"Fehler: {e}")
        log.warning("Album '%s' fehlgeschlagen: %s", name, e)
    finally:
        job["running"] = False


def read_log_lines(n, newest_first=True):
    """Die letzten n Zeilen des Protokolls (über die rotierten Dateien hinweg)."""
    lines = []
    for i in range(0, 6):
        fn = LOG_FILE if i == 0 else f"{LOG_FILE}.{i}"
        try:
            with open(fn, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines() + lines
        except OSError:
            break
        if len(lines) >= n:
            break
    lines = lines[-n:]
    if newest_first:
        lines.reverse()
        return lines
    return "\n".join(lines) + "\n"


def protokoll_page(raw_path):
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_path).query)
    n = min(int((qs.get("n") or ["500"])[0] or 500), 20000)
    q = (qs.get("q") or [""])[0].strip()
    only_warn = (qs.get("warn") or [""])[0] == "1"
    hide_prog = (qs.get("fort") or [""])[0] == "1" if "f" in qs else True   # Standard: ausblenden
    rows = []
    for line in read_log_lines(n * 4 if (hide_prog or q or only_warn) else n):
        if hide_prog and " Fortschritt | " in line:
            continue
        if only_warn and not any(k in line for k in (" WARNING ", " ERROR ", " CRITICAL ")):
            continue
        if q and q.lower() not in line.lower():
            continue
        cls = "w" if (" WARNING " in line) else "e" if (" ERROR " in line or " CRITICAL " in line) else ""
        rows.append(f'<div class="{cls}">{html.escape(line)}</div>')
        if len(rows) >= n:
            break
    body = "".join(rows) or '<p class="meta">Keine Einträge (Protokolldatei entsteht ab Version 2026-10-21b).</p>'
    sel = lambda v: " selected" if n == v else ""  # noqa: E731
    return f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Bildsuche – Protokoll</title>
<style>
body{{font-family:-apple-system,system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#222}}
@media (prefers-color-scheme:dark){{body{{background:#1c1c1e;color:#eee}} #log{{background:#2c2c2e!important}}}}
a{{color:#2a6fdb;text-decoration:none}} .meta{{color:#888;font-size:14px}}
form{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:12px 0;font-size:14px}}
input[type=text],select{{font:inherit;padding:4px 8px;border-radius:6px;border:1px solid #aab}}
button{{font:inherit;padding:4px 12px;border-radius:6px;border:1px solid #2a6fdb;background:#2a6fdb;color:#fff;cursor:pointer}}
#log{{background:#fff;border-radius:8px;padding:10px 12px;font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;word-break:break-word}}
#log .w{{color:#b26a00}} #log .e{{color:#c0392b;font-weight:600}}
</style></head><body>
<p><a href="/">← Übersicht</a></p>
<h1 style="margin:0 0 4px">Protokoll</h1>
<div class="meta">Neueste Einträge oben · Version {VERSION} · Stand {datetime.now():%d.%m.%Y %H:%M:%S} ·
<a href="/protokoll.txt">als Textdatei herunterladen</a></div>
<form method="get"><input type="hidden" name="f" value="1">
<label>Zeilen <select name="n"><option{sel(200)}>200</option><option{sel(500)}>500</option><option{sel(2000)}>2000</option><option{sel(10000)}>10000</option></select></label>
<label><input type="checkbox" name="fort" value="1"{" checked" if hide_prog else ""}> Fortschritt ausblenden</label>
<label><input type="checkbox" name="warn" value="1"{" checked" if only_warn else ""}> nur Warnungen/Fehler</label>
<input type="text" name="q" value="{html.escape(q, quote=True)}" placeholder="Suchen, z. B. Album">
<button>Anzeigen</button>
</form>
<div id="log">{body}</div>
</body></html>"""


CONFIG_PAGE = """<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bildsuche – Konfiguration</title>
<style>
:root{--bg:#f6f6f4;--fg:#222;--muted:#888;--card:#fff;--line:#d8d8d8;--accent:#2a6fdb;--ok:#1d7a3a;--err:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#eee;--card:#2c2c2e;--line:#444;--ok:#5cc27a;--err:#ff8a80}}
body{font-family:-apple-system,system-ui,sans-serif;margin:24px;background:var(--bg);color:var(--fg)}
h1{margin:0 0 6px} a{color:var(--accent);text-decoration:none} .muted{color:var(--muted);font-size:14px}
textarea{width:100%;box-sizing:border-box;height:70vh;margin-top:12px;padding:12px;border-radius:8px;
  border:1px solid var(--line);background:var(--card);color:var(--fg);
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;tab-size:2;white-space:pre}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
button{font:inherit;padding:7px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
button.sec{background:transparent;color:var(--accent)} button:disabled{opacity:.5;cursor:default}
#msg{margin-top:10px;white-space:pre-line} .ok{color:var(--ok)} .err{color:var(--err)}
</style></head><body>
<p class="muted"><a href="/">← Übersicht</a></p>
<h1>Konfiguration</h1>
<div class="muted">searches.yaml · Änderungen wirken nach dem Speichern innerhalb von ca. 30 Sekunden – ohne Neustart.
Vor jedem Speichern wird geprüft und eine Sicherung in <code>config/sicherungen/</code> angelegt.</div>
<textarea id="t" spellcheck="false" autocapitalize="off" autocomplete="off"></textarea>
<div class="bar">
<button onclick="save()" id="sv">Speichern</button>
<button class="sec" onclick="check()">Nur prüfen</button>
<button class="sec" onclick="load()">Verwerfen &amp; neu laden</button>
<span class="muted" id="state"></span>
</div>
<div id="msg"></div>
<script>
const t = document.getElementById('t'); let saved = '';
function msg(text, ok){ const m = document.getElementById('msg'); m.textContent = text; m.className = ok === undefined ? '' : (ok ? 'ok' : 'err'); }
function dirty(){ document.getElementById('state').textContent = t.value !== saved ? 'ungespeicherte Änderungen' : ''; }
async function load(){
  if (t.value !== saved && !confirm('Ungespeicherte Änderungen verwerfen?')) return;
  const r = await fetch('/api/config'); const j = await r.json();
  t.value = saved = j.text || ''; dirty(); msg(j.message || '');
}
async function send(dry){
  const r = await fetch('/api/config' + (dry ? '?pruefen=1' : ''), {method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({text: t.value})});
  const j = await r.json(); msg(j.message, j.ok); return j.ok;
}
async function check(){ await send(true); }
async function save(){
  document.getElementById('sv').disabled = true;
  try { if (await send(false)) { saved = t.value; dirty(); } } finally { document.getElementById('sv').disabled = false; }
}
t.addEventListener('input', dirty);
t.addEventListener('keydown', e => {
  if (e.key === 'Tab') { e.preventDefault(); const s = t.selectionStart;
    t.setRangeText('  ', s, t.selectionEnd, 'end'); dirty(); }
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); save(); }
});
window.addEventListener('beforeunload', e => { if (t.value !== saved) { e.preventDefault(); e.returnValue = ''; } });
load();
</script></body></html>"""


def start_web():
    """Kleiner Webserver im Heimnetz: liefert NUR den Report-Ordner aus (Reports, Vorschaubilder, CSV)."""
    if WEB_PORT <= 0:
        return
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    class Handler(SimpleHTTPRequestHandler):
        def log_error(self, fmt, *args):       # HTTP-Fehler (z. B. 404, 500) ins Protokoll
            try:
                msg = fmt % args
            except Exception:
                msg = str(fmt)
            if any(f"code {c}" in msg for c in (400, 404, 408, 414)):
                return   # nicht gefundene Seiten, kaputte/fremde Anfragen (z. B. HTTPS auf dem HTTP-Port) – Rauschen
            log.warning("Webseite: %s – %s", getattr(self, "path", "?"), msg)

        def log_message(self, *args):          # kein Protokoll pro Seitenaufruf
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def list_directory(self, path):        # kein Durchstöbern von Ordnern
            self.send_error(404)
            return None

        def _json(self, code, obj):
            import json as _json
            body = _json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _placeholder(self):
            """Übersicht noch nicht erzeugt (kurz nach dem Start): vorhandene Reports trotzdem verlinken."""
            names = sorted(d for d in os.listdir(REPORT_DIR)
                           if os.path.exists(os.path.join(REPORT_DIR, d, "report.html")))
            links = "".join(f'<li><a href="{urllib.parse.quote(n)}/report.html">{html.escape(n)}</a></li>'
                            for n in names) or "<li>noch keine</li>"
            body = (f'<!doctype html><html lang="de"><head><meta charset="utf-8">'
                    f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                    f'<meta http-equiv="refresh" content="30"><title>Bildsuche</title></head>'
                    f'<body style="font-family:-apple-system,system-ui,sans-serif;margin:24px">'
                    f'<h1>Bildsuche</h1><p>Die Übersicht wird gerade erstellt – diese Seite lädt sich '
                    f'alle 30 Sekunden neu.</p><p><a href="/config">Konfiguration ansehen &amp; bearbeiten</a></p>'
                    f'<p>Vorhandene Reports:</p><ul>{links}</ul></body></html>'
                    ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, data, ctype, download=None):
            body = data.encode("utf-8") if isinstance(data, str) else data
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(body)

        def _search_param(self):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            name = (qs.get("search") or [""])[0]
            ok = name and os.path.exists(os.path.join(REPORT_DIR, name, "results.csv")) \
                and "/" not in name and not name.startswith(".")
            return name if ok else None, qs

        def _send_image(self):
            """Originalbild anzeigen – nur Bilder, die im Report der Suche stehen.
            JPG/PNG/WebP/GIF werden direkt geliefert, HEIC/RAW/TIFF als JPEG umgewandelt."""
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            search = (qs.get("s") or [""])[0]
            rel = (qs.get("p") or [""])[0]
            if not search or "/" in search or search.startswith(".") or rel not in report_paths(search):
                return self.send_error(404)
            settings, _ = load_config()
            full = os.path.join(settings["photo_root"], rel)
            if not os.path.isfile(full):
                return self.send_error(404)
            ext = os.path.splitext(full)[1].lower()
            direct = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                      ".webp": "image/webp", ".gif": "image/gif"}
            try:
                if ext in direct:
                    with open(full, "rb") as fh:
                        data = fh.read()
                    ctype = direct[ext]
                else:
                    buf = io.BytesIO()
                    load_image(full, max_side=3000).save(buf, "JPEG", quality=90)
                    data, ctype = buf.getvalue(), "image/jpeg"
            except Exception as e:
                log.warning("Bild konnte nicht angezeigt werden: %s (%s)", rel, e)
                return self.send_error(500)
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition",
                             "inline; filename*=UTF-8''" + urllib.parse.quote(os.path.basename(full)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/bild":
                return self._send_image()
            if path == "/protokoll":
                return self._send_bytes(protokoll_page(self.path), "text/html; charset=utf-8")
            if path == "/protokoll.txt":
                return self._send_bytes(read_log_lines(20000, newest_first=False), "text/plain; charset=utf-8",
                                        download="bildsuche-protokoll.txt")
            if path == "/config":
                body = CONFIG_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            if path == "/api/config":
                try:
                    with open(CONFIG_PATH, encoding="utf-8") as fh:
                        text = fh.read()
                    ok, m = validate_config_text(text)
                    return self._json(200, {"text": text, "message": "Geladen. " + m if ok else "Geladen – " + m})
                except OSError as e:
                    return self._json(500, {"text": "", "message": f"Konnte die Datei nicht lesen: {e}"})
            if path in ("/", "/index.html") and not os.path.exists(os.path.join(REPORT_DIR, "index.html")):
                return self._placeholder()
            if self.path.startswith("/api/album/status"):
                name, _ = self._search_param()
                job = ALBUM_JOBS.get(name) if name else None
                return self._json(200, job or {"running": False, "message": ""})
            return super().do_GET()

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path == "/api/config":
                try:
                    ln = int(self.headers.get("Content-Length") or 0)
                    text = json.loads(self.rfile.read(ln).decode("utf-8")).get("text", "") if 0 < ln < 2_000_000 else ""
                except Exception:
                    return self._json(400, {"ok": False, "message": "Ungültige Anfrage."})
                ok, m = validate_config_text(text)
                if not ok:
                    return self._json(200, {"ok": False, "message": "Nicht gespeichert – " + m})
                if "pruefen=1" in self.path:
                    return self._json(200, {"ok": True, "message": m})
                try:
                    save_config_text(text)
                except OSError as e:
                    return self._json(500, {"ok": False, "message": f"Speichern fehlgeschlagen: {e}"})
                log.info("Konfiguration über die Webseite gespeichert.")
                return self._json(200, {"ok": True, "message": "Gespeichert. " + m +
                                        " Die Reports werden in ca. 30 Sekunden neu erstellt."})
            if self.path.startswith("/api/report/delete"):
                name, _ = self._search_param()
                if not name:
                    return self._json(404, {"message": "Report nicht gefunden."})
                try:
                    _, searches = load_config()
                    if name in {q["name"] for q in searches}:
                        return self._json(409, {"message": f"„{name}“ ist eine aktive Suche und wird nicht gelöscht."})
                    base = os.path.realpath(REPORT_DIR)
                    target = os.path.realpath(os.path.join(REPORT_DIR, name))
                    if os.path.dirname(target) != base:
                        return self._json(400, {"message": "Ungültiger Name."})
                    shutil.rmtree(target)
                    log.info("Alter Report '%s' über die Webseite gelöscht.", name)
                    return self._json(200, {"message": f"Report „{name}“ gelöscht."})
                except Exception as e:
                    return self._json(500, {"message": f"Löschen fehlgeschlagen: {e}"})
            if not self.path.startswith("/api/album"):
                return self._json(404, {"message": "unbekannt"})
            name, qs = self._search_param()
            if not name:
                return self._json(400, {"message": "Unbekannte Suche."})
            body = {}
            try:
                ln = int(self.headers.get("Content-Length") or 0)
                if 0 < ln < 5_000_000:
                    body = json.loads(self.rfile.read(ln).decode("utf-8"))
            except Exception:
                return self._json(400, {"message": "Ungültige Anfrage."})
            mode = body.get("mode") or ("test" if (qs.get("test") or ["0"])[0] == "1" else "all")
            if mode not in ("all", "sel", "test", "delete"):
                return self._json(400, {"message": "Ungültiger Modus."})
            title = str(body.get("title") or "").strip() or None
            paths = body.get("paths") if isinstance(body.get("paths"), list) else None
            similar = bool(body.get("similar"))
            with ALBUM_LOCK:
                if any(j.get("running") for j in ALBUM_JOBS.values()):
                    return self._json(409, {"message": "Es läuft bereits ein Album-Auftrag – bitte kurz warten."})
                ALBUM_JOBS[name] = {"running": True, "message": "Auftrag angenommen …"}
            threading.Thread(target=album_job, args=(name, mode, title, paths, similar), daemon=True).start()
            return self._json(202, {"message": "Auftrag angenommen …"})

    try:
        class LoggingServer(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                exc = sys.exc_info()[1]
                if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
                    return                      # Browser hat die Verbindung abgebrochen – harmlos
                log.error("Fehler in der Webseite bei Anfrage von %s", client_address[0], exc_info=True)

        srv = LoggingServer(("0.0.0.0", WEB_PORT), partial(Handler, directory=REPORT_DIR))
    except OSError as e:
        log.warning("Webseite konnte nicht starten (Port %d): %s", WEB_PORT, e)
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log.info("Webseite läuft auf Port %d (http://<NAS>:%d/).", WEB_PORT, WEB_PORT)


# ================================================================ Hauptschleife
def main():
    log.info("Bildsuche Version %s", VERSION)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        log.error("Keine Konfiguration unter %s gefunden.", CONFIG_PATH)
        sys.exit(1)
    settings, searches = load_config()
    cfg_mtime = config_mtime()
    start_web()
    base = Model(MODEL_NAME, MODEL_PRETRAINED)
    try:
        migrate_legacy_index(settings)
    except Exception as e:
        log.error("Übernahme des alten Index fehlgeschlagen: %s", e)
    db, cur_root = None, None
    once = os.environ.get("RUN_ONCE") == "1"
    strong = None

    while True:
        if settings["photo_root"] != cur_root:
            if db:
                db.close()
            cur_root = settings["photo_root"]
            db = DB(index_file(cur_root), cur_root)
            KNOWN_POOL.clear()
            CURRENT_ROOT[0] = cur_root
            log.info("Bildpfad %s – Index %s (%d Bilder bereits berechnet).",
                     cur_root, os.path.basename(db.file), db.count())
        found = scan(settings)
        known = db.known()
        stale = [p for p, ms in known.items() if found.get(p) != ms]
        missing = sum(1 for p in known if p not in found)
        if known and missing > max(1000, 0.3 * len(known)):
            # Schutz: Wenn plötzlich ein Großteil fehlt, ist eher der Ordner nicht erreichbar oder die
            # Konfiguration falsch (photo_root, exclude_dirs) – dann den Index NICHT löschen.
            log.warning("Scan findet %d von %d bekannten Bildern nicht mehr – Index bleibt unverändert. "
                        "photo_root/exclude_dirs und die Einbindung von /photos prüfen.", missing, len(known))
            stale = [p for p in stale if p in found]          # nur echte Änderungen übernehmen
        if stale:
            db.drop(stale)
            gone = [p for p in stale if p not in found]
            changed = [p for p in stale if p in found]
            log.info("%d Bilder aus dem Index entfernt (%d nicht mehr vorhanden, %d geändert – "
                     "geänderte werden neu berechnet).", len(stale), len(gone), len(changed))
            if gone:
                log.info("Nicht mehr vorhanden: %s%s", "; ".join(db.rel(p) for p in gone[:50]),
                         " …" if len(gone) > 50 else "")
            if changed:
                log.info("Geändert: %s%s", "; ".join(db.rel(p) for p in changed[:50]),
                         " …" if len(changed) > 50 else "")
        rerank_key = f"{settings['rerank_model']} {settings['rerank_pretrained']}"
        store = Store(found, db, base.dim, rerank_key)
        eng = Engine(base, db, store, settings, searches)
        eng.strong = strong if (strong and strong.key == rerank_key) else None
        eng.progress(force=True)
        next_scan = time.time() + float(settings["rescan_interval_hours"]) * 3600
        if store.has_g.any():
            eng.report(final=False)          # Übersicht + Reports sofort nach dem Start aktualisieren
        last_report = time.time()
        dirty, rescan = True, False

        while time.time() < next_scan and not rescan:
            mt = config_mtime()
            if mt != cfg_mtime:
                cfg_mtime = mt
                try:
                    new_settings, searches = load_config()
                    rescan = any(new_settings[k] != settings[k] for k in
                                 ("photo_root", "exclude_dirs", "skip_raw_with_sibling", "min_file_kb",
                                  "rerank_model", "rerank_pretrained"))
                    settings = new_settings
                    eng.settings, eng.searches = settings, searches
                    eng.cand_dirty = True
                    log.info("Konfiguration geändert – Reports werden neu erstellt.")
                    if not rescan:
                        eng.report(final=False)
                        last_report = time.time()
                    continue
                except Exception as e:
                    log.error("Konfiguration fehlerhaft, behalte alte: %s", e)

            kind, batch = eng.next_task()
            if kind is None:
                if dirty:
                    eng.progress(force=True)
                    eng.report(final=True)
                    last_report = time.time()
                    dirty = False
                    if eng.todo_r or eng.todo_t:
                        continue            # Report-Bewertung hat neue Prüfaufträge ergeben
                    log.info("Alle Stufen abgeschlossen. Nächster Scan %s.",
                             datetime.fromtimestamp(next_scan).strftime("%d.%m. %H:%M"))
                if once:
                    return
                time.sleep(30)
                continue
            try:
                eng.run(kind, batch)
            except MemoryError:
                log.error("Zu wenig Arbeitsspeicher – mem_limit in compose.yaml erhöhen.")
                raise
            dirty = True
            if time.time() - last_report > float(settings["report_minutes"]) * 60:
                eng.report(final=False)
                last_report = time.time()
        strong = eng.strong


if __name__ == "__main__":
    main()
