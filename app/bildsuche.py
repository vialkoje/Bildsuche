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
import gc
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

VERSION = "2026-11-14b"
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
    "photos_url": "http://192.168.1.10:5000",   # Adresse der NAS (IP, .local löst der Container meist nicht auf)
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
        self.R = {}            # i -> (5, dim_r) float16 (wie in der Datenbank; halber Speicher gegenüber float32)
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
                self.R[i] = np.frombuffer(e, np.float16).reshape(5, -1)
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
            chunk = 20000                       # stückweise statt ganz R auf einmal als float32 zu kopieren
            for a in range(0, len(ids), chunk):
                part = ids[a:a + chunk]
                E = np.stack([s.R[i] for i in part]).astype(np.float32)
                p = pos_prob(E, Ts, self.strong.logit_scale)  # (k, 5)
                del E
                for i, row in zip(part, p):
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
                    s.R[i] = e.astype(np.float16)
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
<div id="lbextra" hidden></div>
</div>
<style>
#lb{position:fixed;inset:0;z-index:100;background:rgba(0,0,0,.92);color:#eee;display:flex;align-items:center;justify-content:center}
#lb[hidden]{display:none}
#lbbar{position:absolute;top:0;left:0;right:0;display:flex;gap:10px;align-items:center;padding:10px 12px;
  padding-top:max(10px, env(safe-area-inset-top))}
#lbbar button{font:inherit;font-size:15px;padding:8px 14px;border-radius:8px;border:1px solid rgba(0,0,0,.15);background:rgba(255,255,255,.82);color:#111;cursor:pointer}
#lbsel.on{background:#2a6fdb;border-color:#2a6fdb;color:#fff}
#lbn{margin:0 auto;font-size:14px;color:#111;background:rgba(255,255,255,.82);padding:4px 10px;border-radius:6px}
#lbwrap{max-width:calc(100vw - 120px);max-height:calc(100vh - 140px);display:flex;align-items:center;justify-content:center}
#lbimg{max-width:calc(100vw - 120px);max-height:calc(100vh - 140px);object-fit:contain;border-radius:4px}
#lbload{position:absolute;color:#aaa;font-size:14px}
.lbnav{position:absolute;top:50%;transform:translateY(-50%);font-size:44px;line-height:1;width:52px;height:84px;
  border:none;border-radius:10px;background:rgba(255,255,255,.75);color:#111;cursor:pointer}
#lbp{left:6px} #lbnx{right:6px}
#lbcap{position:absolute;bottom:max(10px, env(safe-area-inset-bottom));left:50%;transform:translateX(-50%);max-width:calc(100% - 32px);
  box-sizing:border-box;padding:6px 12px;border-radius:6px;background:rgba(255,255,255,.82);
  font-size:13px;color:#111;text-align:center;word-break:break-word}
#lbcap:empty{display:none}
#lbextra{position:absolute;top:calc(max(10px, env(safe-area-inset-top)) + 46px);left:50%;transform:translateX(-50%);
  max-width:calc(100% - 32px);box-sizing:border-box;padding:6px 12px;border-radius:6px;background:rgba(255,255,255,.88);
  font-size:13px;line-height:1.5;color:#111;text-align:center;z-index:2}
#lbextra[hidden]{display:none}
#lbextra a{color:#1f5bbd;font-weight:600}
@media (max-width:700px){#lbwrap,#lbimg{max-width:100vw} .lbnav{display:none}}
</style>
<script>
(function(){
  const lb = document.getElementById('lb'), img = document.getElementById('lbimg'), cap = document.getElementById('lbcap'),
        sel = document.getElementById('lbsel'), num = document.getElementById('lbn'), load = document.getElementById('lbload'),
        extra = document.getElementById('lbextra');
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
    const exs = f ? f.querySelector('.lbextra') : null;
    extra.innerHTML = exs ? exs.innerHTML : ''; extra.hidden = !exs;
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
  extra.addEventListener('click', ev => {          // Querverweis: Bildansicht schließen, dann zur Gruppe
    const a = ev.target.closest('a'); if (!a) return;
    ev.preventDefault(); ev.stopPropagation();
    const h = a.getAttribute('href'); hideNow();
    setTimeout(() => { location.hash = h; }, 0);
  });
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


LAST_INDEX = [None]      # letzte Daten der Übersicht – damit sie nach dem Löschen sofort neu geschrieben werden kann


INDEX_CSS = """
:root{--bg:#f4f4f1;--card:#fff;--ink:#1d1d1f;--mute:#6e6e73;--line:#e3e3e0;--acc:#2a6fdb;--acc2:#1f5bbd;
  --ok:#1e8e3e;--okbg:#e6f4ea;--warn:#b26a00;--warnbg:#fff4e0;--bad:#c0392b;--tint:#eef3fb;--track:#e6e6e2}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--card:#242426;--ink:#f2f2f2;--mute:#9a9a9f;--line:#38383b;
  --acc:#4c8dff;--acc2:#6aa0ff;--ok:#5cc27a;--okbg:#1f3326;--warn:#f0b04a;--warnbg:#3a2e17;--bad:#ff6b5e;--tint:#1f2a40;--track:#38383b}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{font-family:-apple-system,system-ui,"Segoe UI",sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.45}
main{max-width:980px;margin:0 auto;padding:max(20px,env(safe-area-inset-top)) max(16px,env(safe-area-inset-right))
  max(32px,env(safe-area-inset-bottom)) max(16px,env(safe-area-inset-left))}
a{color:var(--acc);text-decoration:none}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:18px}
h1{margin:0;font-size:30px;letter-spacing:-.02em}
.state{display:inline-flex;align-items:center;gap:8px;font-size:14px;color:var(--mute);margin-top:4px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--ok);flex:none}
.dot.run{background:var(--acc);animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{50%{opacity:.35}}
nav.links{display:flex;gap:8px;flex-wrap:wrap}
nav.links a{display:inline-flex;align-items:center;gap:6px;min-height:40px;padding:8px 14px;border-radius:999px;
  background:var(--card);border:1px solid var(--line);color:var(--ink);font-size:14px}
nav.links a:hover{border-color:var(--acc)}
.actions{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:22px}
.action{display:flex;gap:14px;align-items:center;padding:18px;border-radius:16px;min-height:84px;
  background:var(--card);border:1px solid var(--line);color:var(--ink);transition:transform .08s,border-color .15s}
.action:hover{border-color:var(--acc)} .action:active{transform:scale(.98)}
.action.pri{background:var(--acc);border-color:var(--acc);color:#fff}
.action.pri:hover{background:var(--acc2)}
.action .ic{font-size:26px;width:46px;height:46px;border-radius:12px;display:grid;place-items:center;flex:none;background:var(--tint)}
.action.pri .ic{background:rgba(255,255,255,.18)}
.action b{display:block;font-size:17px} .action small{display:block;font-size:13px;opacity:.75;margin-top:2px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);font-weight:600;margin:26px 4px 10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px 18px}
.stages{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
.stage .top{display:flex;justify-content:space-between;align-items:baseline;gap:8px;margin-bottom:6px}
.stage .lbl{font-size:14px;font-weight:500}
.stage .pc{font-size:20px;font-weight:650;font-variant-numeric:tabular-nums}
.stage .cnt{font-size:12px;color:var(--mute);margin-top:5px;font-variant-numeric:tabular-nums}
.bar{height:8px;background:var(--track);border-radius:4px;overflow:hidden}
.bar>div{height:100%;background:var(--acc);border-radius:4px}
.bar.done>div{background:var(--ok)}
.path{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);font-size:13px;color:var(--mute);word-break:break-all}
.path b{color:var(--ink);font-weight:500}
.searches{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:10px}
.search{display:grid;grid-template-columns:minmax(0,1fr) auto;column-gap:10px;row-gap:8px;align-items:center;
  padding:14px 16px;min-width:0;border-radius:14px;background:var(--card);border:1px solid var(--line);color:var(--ink)}
.search:hover{border-color:var(--acc)} .search:active{transform:scale(.99)}
.search .nm{min-width:0;font-weight:600;font-size:16px;overflow-wrap:anywhere}
.search .nm small{display:block;font-weight:400;font-size:12px;color:var(--mute);margin-top:1px}
.search .pills{grid-column:1;display:flex;gap:6px;flex-wrap:wrap}
.search .chev{grid-column:2;grid-row:1/3}
.pill{font-size:13px;font-variant-numeric:tabular-nums;padding:3px 9px;border-radius:999px;white-space:nowrap}
.pill.ok{background:var(--okbg);color:var(--ok)} .pill.pre{background:var(--warnbg);color:var(--warn)}
.pill.zero{background:var(--track);color:var(--mute)}
.chev{color:var(--mute);font-size:20px;flex:none}
.empty{color:var(--mute);font-size:14px}
details.card{padding:0;margin-top:10px}
details.card>summary{list-style:none;cursor:pointer;padding:14px 18px;min-height:52px;display:flex;align-items:center;gap:10px;font-weight:600}
details.card>summary::-webkit-details-marker{display:none}
details.card>summary::after{content:"›";display:inline-block;margin-left:auto;color:var(--mute);font-size:20px;transition:transform .15s}
details.card[open]>summary::after{transform:rotate(90deg)}
details.card>summary .cnt{font-weight:400;color:var(--mute);font-size:14px}
.dbody{padding:0 18px 14px}
.dbody p{font-size:13px;color:var(--mute);margin:0 0 10px}
.row{display:flex;align-items:center;gap:10px;padding:10px 0;border-top:1px solid var(--line);font-size:14px}
.row .grow{flex:1;min-width:0;word-break:break-word}
.row .sub{display:block;font-size:12px;color:var(--mute)}
.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.tag{display:inline-block;white-space:nowrap;font-size:11px;font-weight:600;color:var(--acc);background:var(--tint);padding:2px 7px;border-radius:999px;margin-left:6px}
button.del{font:inherit;font-size:13px;min-height:36px;padding:6px 12px;border-radius:8px;border:1px solid var(--bad);
  background:transparent;color:var(--bad);cursor:pointer;flex:none}
button.del:disabled{opacity:.5}
#oldmsg{font-size:13px;color:var(--mute);margin:8px 0 0}
footer{margin-top:28px;text-align:center;font-size:12px;color:var(--mute)}
@media (max-width:640px){
  h1{font-size:26px}
  header{align-items:flex-start}
  nav.links{width:100%} nav.links a{flex:1;justify-content:center}
  .actions{grid-template-columns:1fr;gap:10px}
  .action{min-height:72px;padding:14px 16px}
  .stages{grid-template-columns:1fr;gap:16px}
  .searches{grid-template-columns:minmax(0,1fr)}
}
"""

INDEX_JS = r"""
async function delReport(b){
  const name = b.dataset.name, msg = document.getElementById('oldmsg');
  if (location.protocol === 'file:') { msg.textContent = 'Das geht nur über die Webseite.'; return; }
  if (!confirm('Alten Report „' + name + '“ löschen?\n\nEntfernt wird nur der Report-Ordner, nicht der Index und keine Bilder.')) return;
  b.disabled = true;
  try {
    const r = await fetch('/api/report/delete?search=' + encodeURIComponent(name), {method: 'POST'});
    const j = await r.json(); msg.textContent = j.message;
    if (r.ok) { const row = document.getElementById('old-' + name); if (row) row.remove(); } else b.disabled = false;
  } catch (e) { msg.textContent = 'Fehler: ' + e; b.disabled = false; }
}
"""


def write_index(counts, summary, final):
    """Übersichtsseite reports/index.html – im Browser über die Webseite oder als Datei.
    Karten-Layout, auf dem Handy einspaltig; Farben wie Assistent und Themen (hell/dunkel)."""
    LAST_INDEX[0] = (counts, summary, final)
    n, g, t, c, r = counts
    esc = html.escape
    num = lambda x: f"{x:,}".replace(",", ".")  # noqa: E731
    pct = lambda a, b: f"{100 * a / b:.0f} %" if b else "–"  # noqa: E731
    pw = lambda a, b: f"{min(100, 100 * a / b):.1f}%" if b else "0%"  # noqa: E731

    def stage(label, a, b, unit):
        done = " done" if b and a >= b else ""
        return (f'<div class="stage"><div class="top"><span class="lbl">{label}</span><span class="pc">{pct(a, b)}</span></div>'
                f'<div class="bar{done}"><div style="width:{pw(a, b)}"></div></div>'
                f'<div class="cnt">{num(a)} von {num(b)} {unit}</div></div>')

    def pretty(name):                          # lego_modell -> Lego modell
        t = re.sub(r"[_\s]+", " ", name).strip()
        return t[:1].upper() + t[1:]

    def pill(k, cls, word):
        return f'<span class="pill {cls if k else "zero"}">{num(k)} {word}</span>'

    cards = "".join(
        f'<a class="search" href="{urllib.parse.quote(name)}/report.html">'
        f'<span class="nm">{esc(pretty(name))}<small>{esc(name)}</small></span><span class="chev">›</span>'
        f'<span class="pills">{pill(a, "ok", "bestätigt")}{pill(b, "pre", "vorläufig") if b else ""}</span></a>'
        for name, a, b in summary) or '<p class="empty">Noch keine Suchen – lege mit dem Assistenten die erste an.</p>'

    indexes = list_indexes()
    idx_rows = "".join(
        f'<div class="row"><span class="grow">{esc(p)}{"<span class=tag>aktiv</span>" if p == CURRENT_ROOT[0] else ""}</span>'
        f'<span class="num">{num(k)} Bilder</span></div>' for p, k, _ in indexes) or '<div class="row">–</div>'

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
        f'<div class="row" id="old-{esc(d, quote=True)}"><span class="grow"><a href="{urllib.parse.quote(d)}/report.html">{esc(d)}</a>'
        f'<span class="sub">Stand {ts:%d.%m.%Y %H:%M} · {num(k)} bestätigt</span></span>'
        f'<button class="del" data-name="{esc(d, quote=True)}" onclick="delReport(this)">Löschen</button></div>'
        for d, ts, k in orphans)
    old_block = (f"""<details class="card"><summary>Ältere Reports <span class="cnt">{len(orphans)}</span></summary>
<div class="dbody"><p>Reports von Suchen, die es in der Konfiguration nicht mehr gibt (z. B. nach dem Umbenennen).
Sie werden nicht mehr aktualisiert. „Löschen“ entfernt nur den Report-Ordner – Index und Bilder bleiben unberührt.</p>
{old_rows}<p id="oldmsg"></p></div></details>""" if orphans else "")

    state = ("Fertig – wartet auf neue Bilder" if final
             else "Berechnung läuft – die Reports werden laufend besser")
    page = f"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#f4f4f1" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#161618" media="(prefers-color-scheme: dark)">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Bildsuche">
<meta http-equiv="refresh" content="300">
<title>Bildsuche</title>
<style>{INDEX_CSS}</style></head><body><main>
<header>
  <div><h1>Bildsuche</h1>
  <div class="state"><span class="dot{"" if final else " run"}"></span>{state} · {datetime.now():%d.%m.%Y %H:%M}</div></div>
  <nav class="links"><a href="/config">⚙︎ Konfiguration</a><a href="/protokoll">☰ Protokoll</a></nav>
</header>
<div class="actions">
  <a class="action pri" href="/assistent"><span class="ic">＋</span><span><b>Neue Suche anlegen</b><small>Schritt für Schritt mit Probebildern</small></span></a>
  <a class="action" href="/themen"><span class="ic">🗂️</span><span><b>Bilder nach Themen erkunden</b><small>Das Archiv nach Gruppen durchstöbern</small></span></a>
</div>
<h2>Suchen</h2>
<div class="searches">{cards}</div>
<h2>Fortschritt</h2>
<div class="card">
  <div class="stages">
    {stage("Stufe 1 · Gesamtbild", g, n, "Bilder")}
    {stage("Stufe 2 · Kacheln", t, n, "Bilder")}
    {stage("Stufe 3 · Prüfung", r, c, "Kandidaten")}
  </div>
  <div class="path">Bildpfad: <b>{esc(CURRENT_ROOT[0])}</b></div>
</div>
<details class="card" style="margin-top:22px"><summary>Vorhandene Indizes <span class="cnt">{len(indexes)}</span></summary>
<div class="dbody"><p>Pro Bildpfad wird ein eigener Index geführt. Wechselst du in der Konfiguration <code>photo_root</code>,
wird für neue Pfade ein weiterer Index angelegt; beim Zurückwechseln wird der vorhandene weiterbenutzt.</p>
{idx_rows}</div></details>
{old_block}
<footer>Version {VERSION} · aktualisiert sich alle 5 Minuten</footer>
</main>
<script>{INDEX_JS}</script>
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


# ================================================================ Assistent für neue Suchen
ENGINE = [None]          # laufende Engine (für Probesuchen aus der Webseite)
PROBES = {}              # token -> {"items": [...], "time": t}
PROBE_LOCK = threading.Lock()
PROBE_RANKS = [0, 1, 2, 3, 5, 7, 10, 14, 20, 30, 45, 70, 110, 170, 260, 400]
STRICTNESS = {   # Strenge -> (Schwelle "Bestätigt", Abstand zur Grob-Schwelle, beta für die Kalibrierung)
    "vollstaendig": (0.30, 0.10, 2.0),
    "ausgewogen": (0.40, 0.10, 1.0),
    "treffsicher": (0.50, 0.15, 0.5),
}


def assist_slug(text):
    t = str(text or "").lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    return t[:40].strip("_") or "neue_suche"


def assist_clean(lines):
    out = []
    for x in lines or []:
        x = " ".join(str(x).split()).strip(" -•")
        if x and x.lower() not in {o.lower() for o in out}:
            out.append(x[:200])
    return out


def assist_search(draft, settings):
    """Entwurf aus dem Assistenten -> Suche im Format von parse_config (+ Rohdaten für die YAML-Datei)."""
    terms = assist_clean(draft.get("terms"))
    if not terms:
        raise ValueError("Bitte mindestens eine Formulierung angeben.")
    negs = assist_clean(draft.get("negatives"))
    kind = draft.get("kind") or "szene"
    if negs or kind == "szene":
        if "ein foto" not in {n.lower() for n in negs}:
            negs.append("ein Foto")
        own_negs = negs
    else:
        own_negs = None                       # Objekt ohne eigene Liste -> Standard-Gegenkategorien
    strict = draft.get("strict") if draft.get("strict") in STRICTNESS else "ausgewogen"
    thr, gap, _ = STRICTNESS[strict]
    return {
        "name": assist_slug(draft.get("name") or draft.get("motiv")),
        "terms": terms,
        "negatives": own_negs if own_negs else list(settings["default_negatives"]),
        "own_negatives": own_negs,
        "threshold": thr,
        "coarse_threshold": round(max(0.15, thr - gap), 2),
        "strict": strict,
    }


def assist_thumb(path, big=False):
    tdir = os.path.join(DATA_DIR, "thumbs")
    os.makedirs(tdir, exist_ok=True)
    size = 1600 if big else 520
    out = os.path.join(tdir, hashlib.sha1(f"{path}|{size}".encode("utf-8")).hexdigest() + ".jpg")
    if not os.path.exists(out):
        im = load_image(path, max_side=size)
        im.convert("RGB").save(out, "JPEG", quality=82)
    return out


def assist_probe(draft):
    eng = ENGINE[0]
    if eng is None or not eng.store.has_g.any():
        raise RuntimeError("Die Bildsuche ist noch nicht bereit (der Index wird gerade geladen). "
                           "Bitte in ein paar Minuten noch einmal versuchen.")
    settings = eng.settings
    q = assist_search(draft, settings)
    s = eng.store
    score, _ = s.coarse(eng.base.text_matrix(q), eng.base.logit_scale)
    order = np.argsort(-score)
    final_s = {}
    if eng.strong is not None and s.R:
        try:
            final_s, _ = eng.final_scores(q)
        except Exception as e:
            log.warning("Assistent: Prüf-Scores nicht verfügbar (%s)", e)
    picked, items = [], []
    thr = float(settings.get("dedupe_similarity") or 0.95)
    for r in PROBE_RANKS:
        k = r
        while k < len(order) and k < r + 40:   # Serienbilder überspringen
            i = int(order[k])
            if score[i] <= 0:
                break
            g = s.G[i].astype(np.float32)
            if all(float(g @ s.G[j].astype(np.float32)) < thr for j in picked):
                break
            k += 1
        if k >= len(order) or score[int(order[k])] <= 0:
            continue
        i = int(order[k])
        if i in picked:
            continue
        picked.append(i)
        items.append({"path": s.paths[i], "rank": k + 1, "coarse": round(float(score[i]), 3),
                      "final": round(final_s[i], 3) if i in final_s else None})
    token = hashlib.sha1(f"{time.time()}{random.random()}".encode()).hexdigest()[:16]
    with PROBE_LOCK:
        for t in [t for t, v in PROBES.items() if time.time() - v["time"] > 3 * 3600]:
            PROBES.pop(t, None)
        PROBES[token] = {"items": items, "time": time.time()}
    log.info("Assistent: Probesuche '%s' mit %d Bildern.", q["name"], len(items))
    return {"token": token, "name": q["name"],
            "items": [{"i": n, "rank": it["rank"], "score": it["coarse"],
                       "rel": os.path.relpath(it["path"], settings["photo_root"])} for n, it in enumerate(items)]}


def _best_threshold(pairs, beta, default, lo=0.15, hi=0.85):
    """pairs: [(score, ja?)] -> Schwelle, die Ja- und Nein-Bilder am besten trennt (F-beta)."""
    yes = [s for s, y in pairs if y]
    no = [s for s, y in pairs if not y]
    if not yes:
        return default
    if not no:
        return round(min(max(lo, min(default, min(yes) - 0.02)), hi), 2)
    cands = sorted({round(x, 3) for x in [default] + [(a + b) / 2 for a in yes + no for b in yes + no]})
    best, best_f = default, -1.0
    for t in cands:
        tp = sum(1 for s in yes if s >= t)
        fp = sum(1 for s in no if s >= t)
        fn = len(yes) - tp
        if tp == 0:
            continue
        p, r = tp / (tp + fp), tp / (tp + fn)
        f = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
        if f > best_f + 1e-9 or (abs(f - best_f) < 1e-9 and abs(t - default) < abs(best - default)):
            best, best_f = t, f
    return round(min(max(lo, best), hi), 2)


def assist_evaluate(draft, token, labels):
    """Bewertung der Probebilder -> (in Ordnung?, Meldung, Suche mit kalibrierten Schwellen)."""
    with PROBE_LOCK:
        probe = PROBES.get(token)
    if not probe:
        raise ValueError("Die Probesuche ist abgelaufen – bitte die Probebilder neu laden.")
    eng = ENGINE[0]
    settings = eng.settings if eng else load_config()[0]
    q = assist_search(draft, settings)
    items = probe["items"]
    rated = [(items[int(k)], bool(v)) for k, v in labels.items() if str(k).isdigit() and int(k) < len(items)]
    if len(rated) < min(8, len(items)):
        raise ValueError("Bitte bewerte mindestens 8 Bilder.")
    top = sorted(rated, key=lambda x: x[0]["rank"])[:8]
    hit_rate = sum(1 for _, y in top if y) / len(top)
    _, gap, beta = STRICTNESS[q["strict"]]
    q["coarse_threshold"] = _best_threshold([(it["coarse"], y) for it, y in rated], beta,
                                            q["coarse_threshold"], lo=max(0.15, q["coarse_threshold"] - 0.15))
    fin = [(it["final"], y) for it, y in rated if it["final"] is not None]
    if sum(1 for _, y in fin if y) >= 2:
        q["threshold"] = _best_threshold(fin, beta, q["threshold"], lo=0.2)
    ok = hit_rate >= 0.6
    if ok:
        msg = (f"{round(hit_rate * 100)} % der wahrscheinlichsten Probebilder passen – die Suche ist gut genug "
               f"zum Speichern.")
    else:
        msg = (f"Nur {round(hit_rate * 100)} % der wahrscheinlichsten Probebilder passen. So würde die Suche zu "
               f"viele falsche Bilder liefern. Schreib bei den Nein-Bildern kurz dazu, was stattdessen zu sehen "
               f"ist, oder ändere die Formulierungen – dann neu probieren.")
    return ok, msg, q, hit_rate


def assist_yaml_block(q, motiv, indent="  "):
    j = lambda x: json.dumps(str(x), ensure_ascii=False)  # noqa: E731  (JSON-Strings sind gültiges YAML)
    i2 = indent + "  "
    lines = [f"{indent}# {' '.join(str(motiv).split())[:100]} – angelegt mit dem Assistenten am "
             f"{datetime.now():%d.%m.%Y}",
             f"{indent}- name: {q['name']}", f"{i2}terms:"]
    lines += [f"{i2}  - {j(t)}" for t in q["terms"]]
    if q.get("own_negatives"):
        lines.append(f"{i2}negatives:")
        lines += [f"{i2}  - {j(n)}" for n in q["own_negatives"]]
    lines += [f"{i2}threshold: {q['threshold']:.2f}", f"{i2}coarse_threshold: {q['coarse_threshold']:.2f}",
              f"{i2}candidates: 2500", f"{i2}max_results: 1000"]
    return "\n".join(lines) + "\n"


def config_add_search(q, motiv):
    """Hängt die Suche in den Abschnitt 'searches:' der Konfiguration ein (Kommentare bleiben erhalten)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        text = f.read()
    _, searches = parse_config(yaml.safe_load(text) or {})
    if q["name"] in {s["name"] for s in searches}:
        raise ValueError(f"Es gibt schon eine Suche „{q['name']}“ – bitte einen anderen Namen wählen.")
    lines = text.splitlines()
    start = next((n for n, l in enumerate(lines) if re.match(r"^searches\s*:", l)), None)
    if start is None:
        lines += ["", "searches:"]
        start = len(lines) - 1
    elif re.match(r"^searches\s*:\s*\[\s*\]", lines[start]):
        lines[start] = "searches:"
    end = len(lines)
    for n in range(start + 1, len(lines)):
        if lines[n].strip() and not lines[n].startswith((" ", "\t", "#", "-")):
            end = n
            break
    indent = "  "
    for l in lines[start + 1:end]:
        m = re.match(r"^(\s*)- ", l)
        if m:
            indent = m.group(1)
            break
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    block = assist_yaml_block(q, motiv, indent).rstrip("\n").split("\n")
    new = lines[:end] + [""] + block + lines[end:]
    new_text = "\n".join(new) + "\n"
    ok, m = validate_config_text(new_text)
    if not ok:
        raise ValueError("Die neue Konfiguration wäre ungültig, nichts gespeichert: " + m)
    save_config_text(new_text)
    log.info("Assistent: neue Suche '%s' in die Konfiguration eingetragen.", q["name"])
    return assist_yaml_block(q, motiv, indent)


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


ASSIST_PAGE = r"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bildsuche – Neue Suche anlegen</title>
<style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#222;--mute:#777;--line:#e3e3e0;--acc:#2a6fdb;--ok:#1e8e3e;--bad:#c0392b;--tint:#eef3fb}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--card:#2c2c2e;--ink:#eee;--mute:#999;--line:#444;--tint:#23304a}}
*{box-sizing:border-box} body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--ink)}
main{max-width:860px;margin:0 auto;padding:20px 16px 80px}
a{color:var(--acc);text-decoration:none}
h1{font-size:24px;margin:8px 0 2px} .step{color:var(--mute);font-size:14px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:10px 0 18px} .bar>div{height:100%;background:var(--acc);transition:width .2s}
.card{background:var(--card);border-radius:12px;padding:18px 18px 16px;box-shadow:0 1px 2px rgba(0,0,0,.06)}
h2{font-size:20px;margin:0 0 10px}
.why{background:var(--tint);border-radius:10px;padding:12px 14px;font-size:14.5px;line-height:1.5;margin:0 0 14px}
.why>b:first-child{display:block;margin-bottom:4px}
.ex{font-size:14px;color:var(--mute);margin:6px 0 0;line-height:1.5}
textarea,input[type=text]{width:100%;font:inherit;font-size:16px;padding:10px 12px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
textarea{min-height:110px;resize:vertical}
.opt{display:flex;gap:12px;align-items:flex-start;padding:12px;border:1px solid var(--line);border-radius:10px;margin:8px 0;cursor:pointer}
.opt input{margin-top:4px;transform:scale(1.2)} .opt small{display:block;color:var(--mute);margin-top:3px;line-height:1.4}
.opt.on{border-color:var(--acc);background:var(--tint)}
.group{margin:14px 0 4px;font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px}
.chip{font:inherit;font-size:14px;padding:7px 12px;border-radius:16px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff}
.nav{display:flex;justify-content:space-between;gap:10px;margin-top:18px}
button.p,button.s{font:inherit;font-size:16px;padding:10px 18px;border-radius:8px;cursor:pointer;border:1px solid var(--acc)}
button.p{background:var(--acc);color:#fff} button.s{background:transparent;color:var(--acc)}
button:disabled{opacity:.45;cursor:default}
.warn{color:var(--bad);font-size:14px;margin-top:8px;white-space:pre-line} .msg{margin-top:12px;font-size:15px;white-space:pre-line;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:12px}
.pc{background:var(--card);border:3px solid var(--line);border-radius:10px;overflow:hidden}
.pc.yes{border-color:var(--ok)} .pc.no{border-color:var(--bad)}
.pc img{width:100%;height:140px;object-fit:cover;display:block;background:var(--line);cursor:zoom-in}
.pc .b{display:flex;gap:6px;padding:8px}
.pc .b button{flex:1;font:inherit;font-size:15px;padding:7px 0;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
.pc.yes .y{background:var(--ok);color:#fff;border-color:var(--ok)} .pc.no .n{background:var(--bad);color:#fff;border-color:var(--bad)}
.pc .r{font-size:12px;color:var(--mute);padding:0 8px 6px}
.pc input{display:none;margin:0 8px 8px;width:calc(100% - 16px);font-size:14px;padding:6px 8px}
.pc.no input{display:block}
pre{background:var(--tint);padding:12px;border-radius:8px;overflow:auto;font-size:13px}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.9);display:none;align-items:center;justify-content:center;z-index:9}
#zoom img{max-width:96vw;max-height:92vh;object-fit:contain} #zoom.on{display:flex}
</style></head><body><main>
<p><a href="/">← Übersicht</a></p>
<h1>Neue Suche anlegen</h1>
<div class="step" id="stepinfo"></div>
<div class="bar"><div id="prog"></div></div>
<div class="card" id="card"></div>
<div id="zoom" onclick="this.classList.remove('on')"><img alt=""></div>
</main>
<script>
const D = {motiv:'', kind:'', wer:[], wo:'egal', terms:[], termsTouched:false, negatives:[], strict:'', name:'', nameTouched:false};
let INFO = {ready:false, names:[], default_negatives:[]}, step = 0, PROBE = null, LABELS = {}, NOTES = {};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const lines = t => t.split('\n').map(x => x.trim()).filter(Boolean);
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;
function slug(t){ t = (t||'').toLowerCase().replace(/ä/g,'ae').replace(/ö/g,'oe').replace(/ü/g,'ue').replace(/ß/g,'ss');
  return t.replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'').slice(0,40).replace(/_+$/,'') || 'neue_suche'; }
function draft(){ return {motiv:D.motiv, kind:D.kind, terms:D.terms, negatives:D.negatives, strict:D.strict, name:D.name}; }

function suggestTerms(){
  const base = cap(D.motiv.trim().replace(/[.!?]+$/,''));
  const t = [base];
  if (D.wo === 'drinnen') t.push(base + ' in einem Zimmer');
  if (D.wo === 'draussen') t.push(base + ' im Freien');
  if (D.kind === 'objekt') t.push('Nahaufnahme: ' + base);
  return t;
}
function suggestNegatives(){
  // [Vorschlag, Kernbegriffe] – ein Vorschlag entfällt, wenn sein Kern im Gesuchten vorkommt
  const s = [];
  const kids = D.wer.includes('kinder'), adults = D.wer.includes('erwachsene') || D.wer.includes('familie');
  if (kids) s.push(['Kinder beim Essen am Tisch', ['essen','mahlz','frühs','mittag','abend']],
                   ['Kinder machen Hausaufgaben oder malen', ['hausa','malen','malt ','zeich','schul']],
                   ['Kinder basteln am Tisch', ['bastel']],
                   ['ein Kindergeburtstag mit Kuchen und Geschenken', ['geburt','kuche','torte','gesch','feier','party']],
                   ['Kinder schauen auf ein Tablet oder Handy', ['table','handy','ipad','fernse']],
                   ['Kinder spielen draußen im Garten', ['garte','drauß','spielp','schauk','rutsch']]);
  if (adults) s.push(['ein Familienfoto, alle schauen in die Kamera', ['gruppenf','portr','posie','kamera']],
                     ['Menschen beim Essen am Tisch', ['essen','mahlz','resta']],
                     ['eine Feier mit vielen Gästen', ['feier','party','fest ','hochz','gäste']]);
  if (D.wo === 'draussen') s.push(['ein Bergpanorama oder eine weite Landschaft', ['berg','lands','panor','gipfe','ausbl']],
                                  ['eine Wanderung auf einem Weg', ['wande','weg ','spazi']]);
  if (D.kind === 'objekt' || D.wer.includes('niemand')) s.push(['ein Foto eines Zimmers', ['zimme','wohnz','küche','raum']],
                                                               ['ein Foto von Essen', ['essen','kuche','torte','gerich','speis']],
                                                               ['eine Landschaft', ['lands','berg','natur','panor']]);
  s.push(['ein Screenshot oder Dokument', ['scree','dokum','text','brief']]);
  const me = (' ' + D.motiv + ' ' + D.terms.join(' ') + ' ').toLowerCase();
  const seen = new Set();
  return s.filter(([n, core]) => !core.some(c => me.includes(c)) && !seen.has(n) && seen.add(n)).map(([n]) => n);
}

const STEPS = [
 { title: 'Was möchtest du finden?',
   why: '<b>Warum diese Frage?</b>Die Bildsuche vergleicht deine Beschreibung mit jedem Foto im Archiv. Sie versteht Deutsch und ganze Sätze – am besten beschreibst du das Bild so, wie du es als Bildunterschrift in ein Fotoalbum schreiben würdest. Einzelne Wörter wie „Würfel“ funktionieren deutlich schlechter als ein kurzer Satz, der die ganze Situation beschreibt.',
   body: () => `<textarea id="f" placeholder="z. B. Kinder spielen ein Würfelspiel am Tisch">${esc(D.motiv)}</textarea>
     <p class="ex">Gute Beispiele: „Kinder bauen eine Sandburg am Strand“ · „unsere Geburtstagstorten mit Kerzen“ · „Oma liest den Kindern vor“.<br>
     Weniger gut: „Sand“ · „Torte“ · „schöne Momente“ (zu allgemein, das Bild lässt sich nicht vorstellen).</p>`,
   save: () => { const v = document.getElementById('f').value.trim(); if (v !== D.motiv) { D.motiv = v; D.termsTouched = false; if (!D.nameTouched) D.name = ''; } },
   check: () => D.motiv.length >= 4 || 'Bitte beschreibe kurz, was auf den Bildern zu sehen sein soll.' },

 { title: 'Was steht auf den Bildern im Mittelpunkt?',
   why: '<b>Warum diese Frage?</b>Das ist die wichtigste Entscheidung für eine gute Suche. Auf normalen Familienfotos sind Gegenstände oft nur klein zu sehen – ein Würfel ist vielleicht so groß wie ein Fingernagel. Die Bildsuche erkennt dann viel zuverlässiger die <i>Situation</i> („Kinder sitzen am Tisch und spielen“) als den kleinen Gegenstand. Geht es dagegen um etwas, das groß und deutlich im Bild ist (eine Torte, ein Auto, ein Bauwerk), beschreibt man besser den Gegenstand selbst.',
   body: () => opt('kind','szene','Eine Situation – Menschen tun etwas','z. B. Kinder spielen, die Familie wandert, jemand backt. Der Gegenstand darf klein oder nebensächlich sein.')
            + opt('kind','objekt','Ein Gegenstand oder Motiv, groß im Bild','z. B. eine Geburtstagstorte, ein LEGO-Modell, ein Sonnenuntergang, ein Auto.'),
   save: () => {}, check: () => !!D.kind || 'Bitte wähle eine der beiden Möglichkeiten.' },

 { title: 'Wer ist zu sehen – und wo?',
   why: '<b>Warum diese Frage?</b>Daraus macht der Assistent zusätzliche Formulierungen und schlägt passende Ausschlüsse vor (zum Beispiel „Kinder beim Essen“, wenn es um Kinder geht). Wenn du dir nicht sicher bist, wähle „egal“ – das schadet nicht.',
   body: () => '<div class="group">Wer ist meistens auf den Bildern? (mehrere möglich)</div><div class="chips">'
            + [['kinder','Kinder'],['erwachsene','Erwachsene'],['familie','die ganze Familie'],['niemand','niemand – nur das Motiv']]
              .map(([k,l]) => `<button class="chip${D.wer.includes(k)?' on':''}" onclick="togWer('${k}')">${l}</button>`).join('') + '</div>'
            + '<div class="group">Wo sind die Bilder meistens aufgenommen?</div>'
            + opt('wo','drinnen','Drinnen','in der Wohnung, im Haus, in einer Halle') + opt('wo','draussen','Draußen','im Garten, in der Natur, unterwegs') + opt('wo','egal','Egal / beides',''),
   save: () => {}, check: () => true },

 { title: 'Passen diese Formulierungen?',
   why: '<b>Warum diese Frage?</b>Die Suche arbeitet mit mehreren Formulierungen gleichzeitig und mittelt sie – das macht sie robuster. Jede Zeile sollte das <i>ganze Bild</i> beschreiben, wie eine Bildunterschrift. Du kannst die Vorschläge ändern, löschen oder eigene ergänzen; zwei bis vier Zeilen sind ideal. Eine englische Zeile („children playing a board game“) kann zusätzlich helfen, ist aber kein Muss.<br><br><b>Wichtig:</b> Schreib hier nichts, was <i>nicht</i> zu sehen sein soll („ohne Karten“, „keine Erwachsenen“) – Verneinungen versteht die Bildsuche nicht. Das kommt im nächsten Schritt dran.',
   enter: () => { if (!D.termsTouched || !D.terms.length) D.terms = suggestTerms(); },
   body: () => `<textarea id="f" rows="5">${esc(D.terms.join('\n'))}</textarea><p class="ex">Eine Formulierung pro Zeile.</p>`,
   save: () => { const t = lines(document.getElementById('f').value); if (t.join('\n') !== D.terms.join('\n')) D.termsTouched = true; D.terms = t; },
   check: () => { if (!D.terms.length) return 'Bitte mindestens eine Formulierung eintragen.';
     const neg = D.terms.filter(t => /(^|\s)(kein|keine|keinen|keiner|nicht)(\s|$)/i.test(t));
     if (neg.length) return 'Diese Zeilen enthalten eine Verneinung – die versteht die Bildsuche nicht:\n„' + neg.join('“, „') + '“\nBitte umformulieren und das Unerwünschte im nächsten Schritt eintragen.';
     const ohne = D.terms.filter(t => /(^|\s)ohne(\s|$)/i.test(t)), key = ohne.join('|');
     if (ohne.length && D.ohneOk !== key) { D.ohneOk = key; return 'Hinweis: „ohne …“ beachtet die Bildsuche kaum – „Kinder ohne Eltern“ findet fast dasselbe wie „Kinder“. Was nicht zu sehen sein soll, gehört besser in den nächsten Schritt. Wenn du es trotzdem so lassen willst, tippe noch einmal auf „Weiter“.'; }
     return true; } },

 { title: 'Was soll ausdrücklich nicht gefunden werden?',
   why: '<b>Warum diese Frage?</b>Die Suche entscheidet für jedes Bild: Passt es besser zu deiner Beschreibung oder besser zu einer dieser „Gegenbeschreibungen“? So lassen sich ähnliche, aber unerwünschte Bilder aussortieren. Am meisten bringen Situationen, die deiner Suche <i>ähnlich sehen</i> – bei „Kinder spielen am Tisch“ zum Beispiel „Kinder beim Essen am Tisch“.<br><br>Beschreibe positiv, was dann zu sehen ist („Kinder spielen ein Kartenspiel“), nicht „keine Karten“. Und übertreib es nicht: Eine Gegenbeschreibung, die fast dasselbe meint wie deine Suche, verdrängt auch echte Treffer.',
   body: () => '<div class="group">Vorschläge – antippen zum Übernehmen</div><div class="chips">'
            + suggestNegatives().map(n => `<button class="chip${D.negatives.includes(n)?' on':''}" data-n="${esc(n)}" onclick="togNeg(this)">${esc(n)}</button>`).join('')
            + `</div><div class="group">Deine Liste (eine Situation pro Zeile)</div><textarea id="f" rows="6" placeholder="z. B. Kinder spielen ein Kartenspiel">${esc(D.negatives.join('\n'))}</textarea>
            <p class="ex">Du darfst die Liste auch leer lassen${D.kind === 'objekt' ? ' – dann nimmt die Suche allgemeine Standard-Gegenbeschreibungen (Landschaft, Essen, Gebäude …)' : ''}.</p>`,
   save: () => { D.negatives = lines(document.getElementById('f').value); },
   check: () => { const neg = D.negatives.filter(t => /(^|\s)(kein|keine|keinen|keiner|nicht)(\s|$)/i.test(t));
     return !neg.length || 'Bitte beschreiben, was auf solchen Bildern zu sehen ist – statt „kein …“ oder „nicht …“:\n„' + neg.join('“, „') + '“'; } },

 { title: 'Wie streng soll die Suche sein?',
   why: '<b>Warum diese Frage?</b>Jede Suche muss zwischen zwei Fehlern abwägen: Bilder übersehen oder falsche Bilder zeigen. Beides ganz zu vermeiden geht nicht. Deine Wahl legt den Startwert fest – genau eingestellt wird er gleich anhand deiner Bewertung der Probebilder.',
   body: () => opt('strict','vollstaendig','Lieber vollständig','Mir soll kein schönes Foto entgehen. Ein paar falsche sortiere ich gern selbst aus.')
            + opt('strict','ausgewogen','Ausgewogen','Ein guter Mittelweg.')
            + opt('strict','treffsicher','Lieber treffsicher','Was die Suche zeigt, soll stimmen. Dafür darf auch mal ein Bild fehlen.'),
   save: () => {}, check: () => !!D.strict || 'Bitte wähle eine der drei Möglichkeiten.' },

 { title: 'Wie soll die Suche heißen?',
   why: '<b>Warum diese Frage?</b>Unter diesem Namen erscheint die Suche auf der Übersichtsseite, und so heißt später auch das vorgeschlagene Album in Synology Photos („Bildsuche: …“). Erlaubt sind Kleinbuchstaben, Ziffern und Unterstriche; der Name muss eindeutig sein.',
   enter: () => { if (!D.nameTouched || !D.name) D.name = slug(D.motiv); },
   body: () => `<input type="text" id="f" value="${esc(D.name)}"><p class="ex">Vorhandene Suchen: ${INFO.names.map(esc).join(', ') || '–'}</p>`,
   save: () => { const v = slug(document.getElementById('f').value); if (v !== D.name) D.nameTouched = true; D.name = v; },
   check: () => !INFO.names.includes(D.name) || 'Eine Suche mit diesem Namen gibt es schon – bitte einen anderen wählen.' },

 { title: 'Probebilder bewerten',
   why: '<b>Was passiert jetzt?</b>Die Bildsuche zeigt dir Bilder aus eurem Archiv, die sie mit deinen Angaben finden würde. Die ersten sind die wahrscheinlichsten, weiter hinten wird sie unsicherer – genau dort lernt sie am meisten. Tippe bei jedem Bild <b>✓ passt</b> oder <b>✗ passt nicht</b> an (mindestens 8, am besten alle). Bei „passt nicht“ hilft es sehr, wenn du kurz dazuschreibst, was stattdessen zu sehen ist – daraus wird eine Gegenbeschreibung. Antippen des Bildes vergrößert es.<br><br>Gespeichert wird die Suche nur, wenn die wahrscheinlichsten Bilder überwiegend passen. Sonst verbessern wir sie zusammen und probieren noch einmal.',
   enter: () => loadProbe(), body: () => '<div id="probe"><p class="msg">Probesuche läuft …</p></div><div class="msg" id="pmsg"></div>',
   save: () => {}, check: () => true, last: true },
];

function opt(field, val, label, sub){
  const on = D[field] === val;
  return `<label class="opt${on?' on':''}"><input type="radio" name="${field}" ${on?'checked':''} onchange="setF('${field}','${val}')"><span><b>${label}</b>${sub?`<small>${sub}</small>`:''}</span></label>`;
}
function setF(f, v){ D[f] = v; render(); }
function togWer(k){ if (k === 'niemand') D.wer = D.wer.includes(k) ? [] : ['niemand'];
  else { D.wer = D.wer.filter(x => x !== 'niemand'); D.wer = D.wer.includes(k) ? D.wer.filter(x => x !== k) : [...D.wer, k]; } render(); }
function togNeg(b){ const cur = document.getElementById('f'); D.negatives = lines(cur.value); const n = b.dataset.n;
  D.negatives = D.negatives.includes(n) ? D.negatives.filter(x => x !== n) : [...D.negatives, n]; render(); }

function render(){
  const S = STEPS[step];
  document.getElementById('stepinfo').textContent = `Schritt ${step + 1} von ${STEPS.length}`;
  document.getElementById('prog').style.width = (100 * (step + 1) / STEPS.length) + '%';
  document.getElementById('card').innerHTML = `<h2>${S.title}</h2><div class="why">${S.why}</div>${S.body()}<div class="warn" id="warn"></div>
    <div class="nav"><button class="s" onclick="back()" ${step ? '' : 'disabled'}>← Zurück</button>
    ${S.last ? '<button class="p" id="savebtn" onclick="werteAus()" disabled>Auswerten und speichern</button>' : '<button class="p" onclick="next()">Weiter →</button>'}</div>`;
  const f = document.getElementById('f'); if (f && !S.last) f.focus();
}
function go(n){ step = n; const S = STEPS[step]; if (S.enter) S.enter(); render(); window.scrollTo(0, 0); }
function next(){ const S = STEPS[step]; S.save(); const c = S.check(); if (c !== true) { document.getElementById('warn').textContent = c; return; } go(step + 1); }
function back(){ STEPS[step].save(); go(step - 1); }

async function api(path, body){
  const r = await fetch(path, {method: body ? 'POST' : 'GET', headers: {'Content-Type': 'application/json'}, body: body ? JSON.stringify(body) : undefined});
  const j = await r.json(); if (!r.ok) throw new Error(j.message || ('Fehler ' + r.status)); return j;
}
async function loadProbe(){
  PROBE = null; LABELS = {}; NOTES = {};
  try { PROBE = await api('/api/assistent/probe', {draft: draft()}); }
  catch (e) { document.getElementById('probe').innerHTML = `<p class="warn">${esc(e.message)}</p>`; return; }
  document.getElementById('probe').innerHTML = '<div class="grid">' + PROBE.items.map(it => `
    <div class="pc" id="pc${it.i}"><img loading="lazy" src="/api/assistent/bild?t=${PROBE.token}&i=${it.i}" alt="" onclick="zoom(${it.i})">
    <div class="b"><button class="y" onclick="rate(${it.i},true)">✓ passt</button><button class="n" onclick="rate(${it.i},false)">✗ passt nicht</button></div>
    <input type="text" placeholder="Was ist stattdessen zu sehen?" oninput="NOTES[${it.i}]=this.value">
    <div class="r">Platz ${it.rank}</div></div>`).join('') + '</div>';
  updSave();
}
function zoom(i){ const z = document.getElementById('zoom'); z.querySelector('img').src = `/api/assistent/bild?t=${PROBE.token}&i=${i}&gross=1`; z.classList.add('on'); }
function rate(i, y){ LABELS[i] = y; const c = document.getElementById('pc' + i); c.classList.toggle('yes', y); c.classList.toggle('no', !y);
  if (!y) setTimeout(() => c.querySelector('input').focus(), 50); updSave(); }
function updSave(){ const n = Object.keys(LABELS).length, b = document.getElementById('savebtn');
  if (b) { b.disabled = !PROBE || n < Math.min(8, PROBE.items.length); b.textContent = `Auswerten und speichern (${n}/${PROBE ? PROBE.items.length : 0} bewertet)`; } }
async function werteAus(){
  const m = document.getElementById('pmsg'); m.textContent = 'Werte aus …';
  try {
    const j = await api('/api/assistent/save', {draft: draft(), token: PROBE.token, labels: LABELS});
    if (j.saved) {
      document.getElementById('card').innerHTML = `<h2>✓ Gespeichert</h2><p class="msg">${esc(j.message)}</p>
        <details><summary>Eintrag in der Konfiguration ansehen</summary><pre>${esc(j.yaml)}</pre></details>
        <div class="nav"><a href="/"><button class="p">Zur Übersicht</button></a><button class="s" onclick="location.reload()">Noch eine Suche anlegen</button></div>`;
      return;
    }
    const notes = Object.entries(NOTES).filter(([i, t]) => LABELS[i] === false && t.trim()).map(([, t]) => t.trim());
    m.innerHTML = esc(j.message) + `<div class="nav"><button class="s" onclick="go(3)">Formulierungen ändern</button>
      <button class="p" onclick="retry()">${notes.length ? 'Hinweise übernehmen und neu probieren' : 'Neu probieren'}</button></div>`;
  } catch (e) { m.innerHTML = `<span class="warn">${esc(e.message)}</span>`; }
}
function retry(){
  const notes = Object.entries(NOTES).filter(([i, t]) => LABELS[i] === false && t.trim()).map(([, t]) => t.trim());
  notes.forEach(n => { if (!D.negatives.map(x => x.toLowerCase()).includes(n.toLowerCase())) D.negatives.push(n); });
  go(STEPS.length - 1);
}
(async () => {
  try { INFO = await api('/api/assistent/info'); } catch (e) {}
  if (!INFO.ready) document.getElementById('stepinfo').textContent = 'Hinweis: Die Bildsuche lädt gerade ihren Index – die Probebilder am Ende gehen erst danach.';
  // Aus der Themen-Seite: Vorschlag einer Gruppe übernehmen und bei den Formulierungen einsteigen
  const thema = new URLSearchParams(location.search).get('thema');
  if (thema) {
    try {
      const j = await (await fetch('/api/themen?n=' + encodeURIComponent(thema))).json();
      if (j.ready && j.search) {
        D.motiv = j.t; D.kind = 'objekt'; D.strict = 'ausgewogen';
        D.terms = j.search.pos.slice(); D.termsTouched = true;
        D.negatives = j.search.neg.filter(n => n.toLowerCase() !== 'ein foto');
        D.name = slug(j.t); D.nameTouched = true;
        document.getElementById('card').insertAdjacentHTML('beforebegin',
          '<div class="why" style="margin-bottom:14px"><b>Vorschlag aus der Themen-Gruppe „' + esc(j.crumbs.slice(1).map(c => c.t).join(' › ')) +
          '“ übernommen.</b> Die Formulierungen sind einzelne Begriffe – ein kurzer Satz wie eine Bildunterschrift trifft oft besser. ' +
          'Prüfe sie, die Ausschlüsse, Strenge und Namen; danach kommen die Probebilder. Mit „Zurück“ erreichst du auch die ersten Fragen.</div>');
        go(3);
        return;
      }
    } catch (e) {}
  }
  go(0);
})();
</script></body></html>"""


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


# ================================================================ Themen (semantische Gruppen)
# Alle Bilder werden anhand ihrer Gesamtbild-Vektoren (Stufe 1) hierarchisch gruppiert:
# oben ~12 grobe Themen, darunter jeweils bis zu 10 feinere, bis die Gruppen klein genug sind,
# um die Bilder einzeln zu zeigen. Die Bilder werden dabei so sortiert, dass jede Gruppe ein
# zusammenhängender Abschnitt [s, e) ist – "alle Bilder dieser Gruppe" ist dann nur ein Ausschnitt.
# Namen: Das CLIP-Modell vergleicht den Mittelpunkt jeder Gruppe mit einer deutschen Begriffsliste
# und nimmt die Begriffe, die diese Gruppe am besten von ihren Geschwistern UNTERSCHEIDEN.
THEMEN_TOP_K = 12          # grobe Themen auf der obersten Ebene
THEMEN_SUB_K = 10          # höchstens so viele Untergruppen je Gruppe
THEMEN_LEAF = 80           # ab dieser Größe wird nicht weiter geteilt
THEMEN_PAGE = 120          # Bilder pro Seite in der Bildansicht
THEMEN_FIT = 20000         # Stichprobe, auf der k-Means rechnet (Zuordnung danach für alle)
THEMEN_AUTO_NEW = 0.02     # automatisch neu berechnen ab 2 % neuer/fehlender Bilder
THEMEN_FORMAT = 4          # 4: + Suchvorschlag je Gruppe (2026-11-13b); 3: Begriffs-Hierarchie, Stufe 2+3, Querverweise

# Begriffe in drei Ebenen: "Oberbegriff | Bereich: Begriff; Begriff; …". Jeder Begriff steht genau einmal.
# Namen werden aus Oberbegriffen, Bereichen oder einzelnen Begriffen gewählt – je nachdem, was die
# Gruppe am besten zusammenfasst und von ihren Nachbarn abgrenzt.
THEMEN_TAXONOMIE = """
Menschen | Personen: Menschen; Porträt; Selfie; Paar; Gruppenfoto; Menschenmenge; Freunde; Touristen; Senioren; Jugendliche; Teenager
Menschen | Familie: Familie; Großeltern; Geschwister; Mutter mit Kind; Vater mit Kind
Menschen | Babys: Baby; Kleinkind; schlafendes Kind; Kinderwagen; Wickeltisch; Babybett; Stillen; Kinderhochstuhl; erste Schritte
Menschen | Kinder: Kinder; lachende Kinder; Kinder beim Spielen
Feste | Familienfeste: Hochzeit; Brautpaar; Taufe; Kommunion; Konfirmation; Jubiläum; Familienfeier
Feste | Geburtstage: Geburtstag; Geburtstagskuchen; Kindergeburtstag
Feste | Weihnachten: Weihnachten; Weihnachtsbaum; Geschenke auspacken; Adventskranz; Nikolaus; Weihnachtsessen; Weihnachtsmarkt
Feste | Bräuche im Jahreslauf: Ostern; Ostereier; Ostereiersuche; Fasnet; Kostüme; Halloween; Kürbis; Laterne; Martinsumzug; Erntedank; Silvester; Feuerwerk
Feste | Feiern & Vereine: Party; Gartenfest; Grillfest; Picknick; Volksfest; Festumzug; Verein; Vereinsfest; Ehrung; Urkunde
Kultur & Ausflüge | Musik & Bühne: Konzert; Bühne; Theateraufführung; Musiker; Musikinstrument; Akkordeon; Gitarre; Klavier; Blasmusik
Kultur & Ausflüge | Ausflugsziele: Jahrmarkt; Karussell; Zirkus; Freizeitpark; Achterbahn; Zoo; Tierpark; Aquarium; Museum; Ausstellung
Kultur & Ausflüge | Kunst: Gemälde; Kunstwerk; Skulptur; Graffiti; Denkmal; Statue
Spielen | Spielplatz: Spielplatz; Rutsche; Schaukel; Sandkasten; Trampolin; Planschbecken; Hängematte
Spielen | Spielzeug: Spielzeug; Spielfiguren; Puppe; Kuscheltier; Spielzeugauto; Modelleisenbahn; Modellbau
Spielen | LEGO: LEGO; LEGO-Modell; Bausteine
Spielen | Gesellschaftsspiele: Brettspiel; Würfelspiel; Kartenspiel; Puzzle
Spielen | Malen & Basteln: Malen; Basteln; Kinderzeichnung
Spielen | Laufrad & Roller: Laufrad; Roller; Inliner; Skateboard
Wasser | Strand: Sandstrand; Sandburg; Muscheln; Strandurlaub; Sonnenschirm; Liegestuhl; Palmen
Wasser | Meer & Küste: Meer; Wellen; Felsküste; Insel; Leuchtturm; Hafen
Wasser | Boote & Schiffe: Boot; Segelboot; Schiff; Fähre; Kanu
Wasser | Baden: Badesee; Schwimmbad; Hallenbad; Wasserrutsche; Pool; Schwimmen; Unterwasseraufnahme
Wasser | Flüsse & Seen: Fluss; Bach; See; Teich; Wasserfall; Spiegelung
Berge | Gebirge: Berge; Gipfel; Gipfelkreuz; Gletscher; Felswand; Panorama; Bergsee; Berghütte
Berge | Klettern: Klettern; Klettersteig; Bergsteigen
Berge | Wandern: Wandern; Wanderweg; Wanderurlaub; Waldweg
Natur | Wald: Wald; Bäume; Herbstlaub; Pilze
Natur | Landschaft: Landschaft; Hügel; Tal; Wiese; Feld; Getreidefeld; Weinberg; Wüste; Luftaufnahme
Natur | Blumen: Blumen; Blüten; Blumenwiese; Blumenstrauß
Natur | Garten: Garten; Gemüsebeet; Gartenarbeit; Rasenmähen; Apfelbaum; Beeren; Gartenhaus; Terrasse; Balkon
Natur | Himmel & Wetter: Himmel; Wolken; Sonnenuntergang; Sonnenaufgang; Regenbogen; Gewitter; Regen; Nebel; Sternenhimmel; Mond; Nachtaufnahme
Natur | Jahreszeiten: Sommer; Frühling; Herbst
Winter | Schnee: Schnee; verschneite Landschaft; Schneemann; Eis
Winter | Wintersport: Skifahren; Skipiste; Skilift; Langlauf; Snowboard; Schlittenfahren; Schlittschuhlaufen; Skiurlaub
Tiere | Haustiere: Hund; Katze; Kaninchen; Meerschweinchen
Tiere | Bauernhof: Bauernhof; Kühe; Pferde; Schafe; Ziegen; Hühner
Tiere | Wildtiere & Vögel: Wildtiere; Reh; Eichhörnchen; Vögel; Enten; Schwäne; Fische
Tiere | Insekten: Insekten; Schmetterling; Biene
Orte & Gebäude | Stadt: Stadt; Altstadt; Skyline; Hochhäuser; Straße; Gasse; Marktplatz; Brunnen; Brücke; Rathaus; Städtereise; Sehenswürdigkeit
Orte & Gebäude | Kirchen: Kirche; Kircheninneres; Dom; Kloster
Orte & Gebäude | Burgen & Schlösser: Burg; Schloss; Ruine; Turm
Orte & Gebäude | Dorf: Dorf
Orte & Gebäude | Unterkünfte: Hotel; Hotelzimmer; Ferienwohnung; Campingplatz; Zelt; Wohnmobil; Urlaub
Orte & Gebäude | Gastronomie: Restaurant; Café; Biergarten
Orte & Gebäude | Einkaufen: Wochenmarkt; Geschäft; Supermarkt; Einkaufen
Zuhause | Haus: Haus; Einfamilienhaus; Hausfassade; Dach; Haustür; Fenster; Garage
Zuhause | Räume: Wohnzimmer; Küche; Esszimmer; Schlafzimmer; Kinderzimmer; Badezimmer; Keller; Dachboden; Treppe; Innenraum
Zuhause | Einrichtung: Möbel; Sofa; Bett; Esstisch; Dekoration; Kerzen
Zuhause | Handwerk & Bauen: Baustelle; Renovierung; Handwerker; Werkzeug; Werkstatt; Holz; Brennholz; Elektroinstallation; Heizung; Solaranlage; Bagger
Zuhause | Feuer & Grill: Kaminofen; Feuer; Lagerfeuer; Grill
Essen & Trinken | Mahlzeiten: Essen; Mahlzeit; Frühstück; gedeckter Tisch; Buffet; Pizza; Nudeln; Salat; Kochen
Essen & Trinken | Backen & Kuchen: Kuchen; Torte; Plätzchen; Backen; Brot
Essen & Trinken | Obst & Gemüse: Obst; Gemüse
Essen & Trinken | Getränke & Eis: Getränke; Wein; Bier; Kaffee; Eiscreme
Fahrzeuge | Autos: Auto; Oldtimer; Sportwagen; Autorennen; Parkplatz; Autobahn
Fahrzeuge | Zweiräder: Motorrad; Fahrrad; Radtour; Mountainbike
Fahrzeuge | Bahn & Bus: Bus; Zug; Straßenbahn; Bahnhof
Fahrzeuge | Fliegen: Flugzeug; Blick aus dem Flugzeugfenster; Hubschrauber; Flughafen
Fahrzeuge | Einsatz- & Nutzfahrzeuge: Traktor; Feuerwehrauto; Krankenwagen; Feuerwehr; Polizei
Sport | Ballsport: Fußball; Fußballspiel; Tennis; Handball; Basketball; Volleyball; Tischtennis; Golf
Sport | Training & Wettkampf: Sport; Turnen; Leichtathletik; Joggen; Laufwettbewerb; Radrennen; Yoga; Fitnessstudio; Sporthalle; Stadion; Siegerehrung; Pokal
Sport | Reiten: Reiten
Schule & Arbeit | Schule: Schule; Klassenzimmer; Kindergarten; Hausaufgaben; Lesen; Buch; Einschulung; Schultüte; Abschlussfeier; Hörsaal; Tafel
Schule & Arbeit | Beruf: Büro; Arbeitsplatz; Schreibtisch; Vortrag; Präsentation; Besprechung; Fabrikhalle; Maschine
Schule & Arbeit | Technik: Computer; Laptop; Smartphone; Bildschirm; Elektronik; Kabel
Dokumente & Bildarten | Dokumente: Dokument; Text; Brief; Rechnung; Formular; Notizen; Zeitungsartikel; Landkarte; Plan; Schild; Plakat
Dokumente & Bildarten | Bildschirmfotos: Bildschirmfoto
Dokumente & Bildarten | Alte Fotos: abfotografiertes Foto; altes Foto; Schwarzweißfoto
Dokumente & Bildarten | Nahaufnahmen: Nahaufnahme; Makroaufnahme
Kleidung | Kleidung: Kleidung; Schuhe; Tracht; Uniform; Sonnenbrille; Mütze
"""


THEMEN = {"root": None, "tree": None, "mtime": 0}     # geladener Baum des aktuellen Bildpfads
THEMEN_LOCK = threading.Lock()
THEMEN_STATE = {"running": False, "message": "", "error": "", "step": 0, "steps": 4, "phase": "",
                "done": 0, "total": 0, "t_start": 0, "t_phase": 0, "holds_index": False}
THEMEN_PHASES = ["Begriffe vorbereiten", "Bildmerkmale zusammenführen", "Gruppen bilden", "Gruppen benennen",
                 "Querverweise suchen", "Vorschaubilder vorbereiten"]


def themen_phase(step, total=0):
    """Fortschritt: neuer Schritt (1..6) mit erwarteter Menge."""
    THEMEN_STATE.update(step=step, phase=THEMEN_PHASES[step - 1], done=0, total=total, t_phase=time.time(),
                        message=f"Schritt {step} von {len(THEMEN_PHASES)}: {THEMEN_PHASES[step - 1]} …")
    log.info("Themen: %s", THEMEN_STATE["message"])


def themen_tick(done):
    THEMEN_STATE["done"] = done


def themen_status():
    st = THEMEN_STATE
    out = {k: st[k] for k in ("running", "message", "error", "step", "phase", "done", "total")}
    out["steps"] = len(THEMEN_PHASES)
    now = time.time()
    out["elapsed"] = int(now - st["t_start"]) if st["running"] and st["t_start"] else 0
    eta = None
    if st["running"] and st["total"] and st["done"] > 0:
        rate = st["done"] / max(1e-3, now - st["t_phase"])
        eta = int((st["total"] - st["done"]) / rate)
    out["eta"] = eta
    return out
THEMEN_THUMB_SEM = threading.Semaphore(2)              # höchstens 2 Bilder gleichzeitig umrechnen


def themen_file(root):
    base = os.path.basename(index_file(root)).rsplit(".", 1)[0]
    return os.path.join(DATA_DIR, "themen", base + ".json.gz")


def themen_tree(root=None):
    """Baum des (aktuellen) Bildpfads, bei Bedarf von der Platte geladen. None = noch nicht berechnet."""
    import gzip
    root = root or CURRENT_ROOT[0]
    if not root:
        return None
    fn = themen_file(root)
    try:
        mt = os.path.getmtime(fn)
    except OSError:
        return None
    with THEMEN_LOCK:
        if THEMEN["root"] == root and THEMEN["mtime"] == mt and THEMEN["tree"] is not None:
            return THEMEN["tree"]
    try:
        with gzip.open(fn, "rt", encoding="utf-8") as f:
            tree = json.load(f)
        if tree.get("format") != THEMEN_FORMAT:
            return None
    except Exception as e:
        log.warning("Themen-Datei konnte nicht gelesen werden (%s) – wird neu berechnet.", e)
        return None
    with THEMEN_LOCK:
        THEMEN.update(root=root, tree=tree, mtime=mt)
    return tree


def _kmeans(X, k, rng, iters=25):
    """Sphärisches k-Means (Kosinus) mit k-means++-Start. X: (m, d) float32, Zeilen normiert."""
    m = len(X)
    k = max(1, min(k, m))
    C = np.empty((k, X.shape[1]), np.float32)
    C[0] = X[rng.integers(m)]
    best = X @ C[0]
    for j in range(1, k):                       # k-means++: weit entfernte Punkte bevorzugen
        d = np.clip(1.0 - best, 0, None) ** 2
        tot = float(d.sum())
        idx = int(rng.choice(m, p=d / tot)) if tot > 0 else int(rng.integers(m))
        C[j] = X[idx]
        best = np.maximum(best, X @ C[j])
    a = None
    for _ in range(iters):
        sims = X @ C.T
        na = sims.argmax(axis=1)
        if a is not None and (na != a).mean() < 0.002:
            a = na
            break
        a = na
        for j in range(k):
            sel = a == j
            if sel.any():
                c = X[sel].sum(axis=0)
                C[j] = c / (np.linalg.norm(c) or 1.0)
            else:                                # leere Gruppe: am schlechtesten passenden Punkt übernehmen
                C[j] = X[int(sims.max(axis=1).argmin())]
    return C


def _centroid(G, ids, chunk=20000):
    c = np.zeros(G.shape[1], np.float64)
    for a in range(0, len(ids), chunk):
        c += G[ids[a:a + chunk]].astype(np.float32).sum(axis=0)
    c = c.astype(np.float32)
    return c / (np.linalg.norm(c) or 1.0)


def _reps(G, ids, c, n=4, thr=0.95, look=200):
    """Typische Bilder: nah am Mittelpunkt, aber keine Serienbilder doppelt."""
    look_ids = ids[:look]                         # ids sind bereits nach Nähe zum Mittelpunkt sortiert
    picked = []
    for i in look_ids:
        g = G[i].astype(np.float32)
        if all(float(g @ G[j].astype(np.float32)) < thr for j in picked):
            picked.append(int(i))
        if len(picked) >= n:
            break
    return picked


def _vocab_matrix(base, vocab, offset=0):
    """Begriffe als Textvektoren (einmal berechnet, im Datenordner zwischengespeichert)."""
    key = hashlib.sha1(("|".join(vocab) + base.key).encode("utf-8")).hexdigest()[:12]
    fn = os.path.join(DATA_DIR, "themen", f"begriffe_{key}.npy")
    try:
        V = np.load(fn)
        if V.shape[0] == len(vocab):
            return V
    except Exception:
        pass
    texts = vocab + [f"ein Foto: {t}" for t in vocab]
    out, bs = [], 16
    themen_tick(offset)
    with base.torch.no_grad():               # stapelweise statt einzeln: ein Vielfaches schneller
        for a in range(0, len(texts), bs):
            f = base.model.encode_text(base.tokenizer(texts[a:a + bs]))
            f = f / f.norm(dim=-1, keepdim=True)
            out.append(f.cpu().numpy().astype(np.float32))
            themen_tick(offset + min(len(texts), a + bs) // 2)
    E = np.vstack(out)
    V = E[:len(vocab)] + E[len(vocab):]
    V = (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.save(fn, V)
    return V


def themen_taxonomy():
    """Begriffe in drei Ebenen: Oberbegriff › Bereich › Begriff. Liefert (Begriffe, Konzepte).
    Ein Konzept ist ein Name mit der Menge seiner Begriffe – daraus werden die Gruppennamen gewählt."""
    vocab, concepts, ober = [], [], {}
    for line in THEMEN_TAXONOMIE.strip().splitlines():
        head, terms = line.split(":", 1)
        o, m = [x.strip() for x in head.split("|")]
        idx = []
        for t in terms.split(";"):
            t = t.strip()
            if t:
                idx.append(len(vocab))
                vocab.append(t)
        concepts.append({"name": m, "kind": "bereich", "terms": frozenset(idx)})
        ober.setdefault(o, set()).update(idx)
    for o, idx in ober.items():
        if o not in {c["name"] for c in concepts}:
            concepts.append({"name": o, "kind": "ober", "terms": frozenset(idx)})
        else:                                     # gleichnamiger Bereich (z. B. "Kleidung"): nur einmal
            for c in concepts:
                if c["name"] == o:
                    c["terms"] = frozenset(idx)
    names = {c["name"] for c in concepts}
    for j, t in enumerate(vocab):
        if t not in names:
            concepts.append({"name": t, "kind": "begriff", "terms": frozenset([j])})
    return vocab, concepts


def _assign2(X, ids, C, chunk=20000):
    """Beste und zweitbeste Gruppe je Bild + Abstand (für Querverweise)."""
    best = np.empty(len(ids), np.int32)
    second = np.empty(len(ids), np.int32)
    margin = np.empty(len(ids), np.float32)
    for a in range(0, len(ids), chunk):
        S = X[ids[a:a + chunk]].astype(np.float32) @ C.T
        o = np.argsort(-S, axis=1)[:, :2] if S.shape[1] > 1 else np.zeros((len(S), 2), np.int64)
        r = np.arange(len(S))
        best[a:a + chunk] = o[:, 0]
        second[a:a + chunk] = o[:, 1]
        margin[a:a + chunk] = S[r, o[:, 0]] - S[r, o[:, 1]] if S.shape[1] > 1 else 1.0
    return best, second, margin


def _softmax_rows(L):
    L = L - L.max(axis=-1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(axis=-1, keepdims=True)


def themen_search_hits(settings, searches):
    """Bestätigte Treffer der aktiven Suchen (relative Pfade) – Wissen aus Stufe 3."""
    disp_root = settings["display_root"].rstrip("/")
    out = {}
    for q in searches:
        rels = set()
        try:
            with open(os.path.join(REPORT_DIR, q["name"], "results.csv"), encoding="utf-8-sig") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    if row.get("status") == "bestätigt" and row["pfad"].startswith(disp_root):
                        rels.add(row["pfad"][len(disp_root):].lstrip("/"))
        except (OSError, KeyError):
            continue
        if len(rels) >= 30:
            out[q["name"]] = rels
    return out


def themen_pretty(name):
    words = re.split(r"[_\s]+", name.strip())
    return " ".join(w[:1].upper() + w[1:] for w in words if w)


def themen_build(src, root):
    """Baut den Themenbaum aus allem, was der Index schon weiß:
       Gruppen   – Stufe 1 (Gesamtbild) + Stufe 2 (Kacheln, wo schon gerechnet)
       Namen     – Begriffe je Bild aus Stufe 1+2, bei geprüften Bildern zusätzlich das starke Modell
                   aus Stufe 3, dazu die bestätigten Treffer der Suchen
       Querverweise – Bilder, die fast genauso gut in eine Nachbargruppe passen."""
    import gzip
    t0 = time.time()
    st, base, strong = src["store"], src["base"], src["strong"]
    ids_g = src["ids"]                             # Store-Indizes der Bilder mit Gesamtbild-Vektor
    n = len(ids_g)
    rel_paths = src["rel"]
    rng = np.random.default_rng(42)
    vocab, concepts = themen_taxonomy()

    # ---- 1. Begriffe als Textvektoren (beide Modelle)
    themen_phase(1, len(vocab) * (2 if strong is not None else 1))
    V = _vocab_matrix(base, vocab)
    Vs = None
    if strong is not None and st.R:
        try:
            Vs = _vocab_matrix(strong, vocab, offset=len(vocab))
        except Exception as e:
            log.warning("Themen: Begriffe für das Prüfmodell nicht verfügbar (%s) – nur Stufe 1+2.", e)

    # ---- 2. Merkmal je Bild: Gesamtbild, ergänzt um den Mittelwert der Kacheln (wo vorhanden)
    themen_phase(2, n)
    X = np.empty((n, st.G.shape[1]), np.float16)
    with_t = 0
    for a in range(0, n, 20000):
        part = ids_g[a:a + 20000]
        x = st.G[part].astype(np.float32)
        ht = st.has_t[part]
        if ht.any():
            x[ht] += 0.5 * st.T[part[ht]].astype(np.float32).mean(axis=1)
            with_t += int(ht.sum())
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        X[a:a + 20000] = x
        themen_tick(min(n, a + 20000))
    r_set = {i for i in list(st.R.keys())} if Vs is not None else set()
    log.info("Themen: %d Bilder, davon %d mit Kacheln (Stufe 2) und %d geprüft (Stufe 3).",
             n, with_t, sum(1 for i in ids_g if int(i) in r_set) if r_set else 0)

    # ---- 3. Gruppen bilden (hierarchisches k-Means) + Kandidaten für Querverweise
    themen_phase(3, n)
    order = []
    nodes = [{"p": -1, "d": 0, "k": [], "s": 0, "e": 0, "r": []}]
    stack = [(0, np.arange(n))]
    alts = []                                       # (Bild, Nachbargruppe, Abstand)
    done = 0
    while stack:
        nid, ids = stack.pop()
        node = nodes[nid]
        c = _centroid(X, ids)
        node["c"] = c
        k = THEMEN_TOP_K if nid == 0 else min(THEMEN_SUB_K, int(np.ceil(len(ids) / (THEMEN_LEAF * 0.75))))
        split = None
        if len(ids) > THEMEN_LEAF and k >= 2:
            fit = ids if len(ids) <= THEMEN_FIT else rng.choice(ids, THEMEN_FIT, replace=False)
            C = _kmeans(X[fit].astype(np.float32), k, rng)
            a, sec, mar = _assign2(X, ids, C)
            sizes = np.bincount(a, minlength=len(C))
            if sizes.max() < 0.97 * len(ids):
                split = (a, sec, mar, sizes)
        if split is None:
            sims = np.concatenate([X[ids[x:x + 20000]].astype(np.float32) @ c for x in range(0, len(ids), 20000)])
            ids = ids[np.argsort(-sims)]
            node["s"] = len(order)
            order.extend(int(i) for i in ids)
            node["e"] = len(order)
            node["r"] = _reps(X, ids, c)
            done += len(ids)
            themen_tick(done)
            continue
        a, sec, mar, sizes = split
        kids, node_of = [], {}
        for j in np.argsort(-sizes):
            if sizes[j] == 0:
                continue
            nodes.append({"p": nid, "d": node["d"] + 1, "k": [], "s": 0, "e": 0, "r": []})
            node_of[int(j)] = len(nodes) - 1
            kids.append((len(nodes) - 1, ids[a == j]))
        node["k"] = [kid for kid, _ in kids]
        # Grenzgänger: die ~8 % der Bilder mit dem knappsten Abstand zur zweitbesten Gruppe (höchstens 0,02)
        delta = min(0.02, float(np.quantile(mar, 0.08)))
        for i, s2, m in zip(ids[mar <= delta], sec[mar <= delta], mar[mar <= delta]):
            if int(s2) in node_of:
                alts.append((int(i), node_of[int(s2)], float(m)))
        for kid, kid_ids in reversed(kids):
            stack.append((kid, kid_ids))
    for nd in reversed(nodes):
        if nd["k"]:
            nd["s"] = min(nodes[x]["s"] for x in nd["k"])
            nd["e"] = max(nodes[x]["e"] for x in nd["k"])
    for nd in reversed(nodes):
        if nd["k"]:
            r, lists = [], [nodes[x]["r"] for x in nd["k"]]
            for depth_i in range(4):
                for lst in lists:
                    if depth_i < len(lst) and len(r) < 4 and lst[depth_i] not in r:
                        r.append(lst[depth_i])
            nd["r"] = r
    order_a = np.asarray(order, np.int64)
    pos = np.empty(n, np.int64)
    pos[order_a] = np.arange(n)

    # ---- 4. Begriffsverteilung je Bild -> je Gruppe; Namen wählen
    themen_phase(4, n)
    leaf_ids = sorted((x for x, nd in enumerate(nodes) if not nd["k"]), key=lambda x: nodes[x]["s"])
    starts = np.array([nodes[x]["s"] for x in leaf_ids], np.int64)
    sums = np.zeros((len(leaf_ids), len(vocab)), np.float64)
    hits = src["hits"]
    hit_names = sorted(hits)
    rel_index = {r: j for j, r in enumerate(rel_paths)}
    hit_mask = np.zeros((len(hit_names), n), bool)
    for h, name in enumerate(hit_names):
        for r in hits[name]:
            j = rel_index.get(r)
            if j is not None:
                hit_mask[h, pos[j]] = True
    hit_sums = np.zeros((len(leaf_ids), len(hit_names)), np.float64)
    hit_terms = np.zeros((len(hit_names), len(vocab)), np.float64)
    scale = base.logit_scale
    for a in range(0, n, 4000):
        b = min(n, a + 4000)
        sidx = ids_g[order_a[a:b]]                 # Store-Indizes in Reihenfolge
        P = _softmax_rows(scale * (st.G[sidx].astype(np.float32) @ V.T))
        ht = st.has_t[sidx]
        if ht.any():                                # Kacheln: Motive am Rand bekommen ihr Gewicht
            Tt = st.T[sidx[ht]].astype(np.float32)  # (m, 4, d)
            Pt = _softmax_rows(scale * (Tt @ V.T)).mean(axis=1)
            P[ht] = 0.5 * P[ht] + 0.5 * Pt
        if Vs is not None:                          # geprüfte Bilder: starkes Modell zählt mehr
            rows = [x for x, i in enumerate(sidx) if int(i) in r_set]
            if rows:
                R = np.stack([st.R[int(sidx[x])] for x in rows]).astype(np.float32)   # (m, 5, d_r)
                Ps = _softmax_rows(strong.logit_scale * (R @ Vs.T))
                Ps = 0.5 * Ps[:, 0] + 0.5 * Ps[:, 1:].mean(axis=1)
                P[rows] = 0.4 * P[rows] + 0.6 * Ps
        lo = np.searchsorted(starts, a, side="right") - 1
        hi = np.searchsorted(starts, b, side="left")
        seg = np.clip(starts[lo:hi] - a, 0, None)
        sums[lo:lo + len(seg)] += np.add.reduceat(P, seg, axis=0)
        if hit_names:
            Hm = hit_mask[:, a:b]
            hit_sums[lo:lo + len(seg)] += np.add.reduceat(Hm.T.astype(np.float64), seg, axis=0)
            hit_terms += Hm.astype(np.float64) @ P
        themen_tick(b)
    D, H = {}, {}
    for li, x in enumerate(leaf_ids):
        D[x], H[x] = sums[li], hit_sums[li]
    for x in range(len(nodes) - 1, -1, -1):
        if nodes[x]["k"]:
            D[x] = sum(D[k] for k in nodes[x]["k"])
            H[x] = sum(H[k] for k in nodes[x]["k"])
    for x in D:
        size = max(1, nodes[x]["e"] - nodes[x]["s"])
        D[x] = D[x] / size
        H[x] = H[x] / size
    # Suchen als zusätzliche Namen: "Klettersteige" gehört zum Bereich, in dem ihre Treffer meist liegen
    for h, name in enumerate(hit_names):
        dist = hit_terms[h] / max(1.0, hit_mask[h].sum())
        best_c = max((c for c in concepts if c["kind"] == "bereich"),
                     key=lambda c: sum(dist[j] for j in c["terms"]))
        concepts.append({"name": themen_pretty(name), "kind": "suche", "h": h,
                         "terms": frozenset(best_c["terms"]) | {f"suche:{name}"}})
    themen_name_all(nodes, D, H, vocab, concepts, order, rel_paths)

    # ---- 5. Querverweise: Grenzgänger bis zur passendsten kleinsten Gruppe hinunterführen
    themen_phase(5, len(alts))
    per_img, strip = {}, {}
    for x, (i, alt, m) in enumerate(alts):
        top = alt
        v = X[i].astype(np.float32)
        while nodes[alt]["k"]:
            alt = max(nodes[alt]["k"], key=lambda kk: float(v @ nodes[kk]["c"]))
        lst = per_img.setdefault(int(pos[i]), [])
        if alt not in [a2 for a2, _ in lst]:
            lst.append((alt, m))
        for target in {top, alt}:
            strip.setdefault(target, []).append((m, int(pos[i])))
        if x % 2000 == 0:
            themen_tick(x)
    alt_map = {}
    for p_, lst in per_img.items():
        lst.sort(key=lambda t: t[1])
        alt_map[str(p_)] = [a2 for a2, _ in lst[:2]]
    for target, lst in strip.items():
        lst.sort()
        seen, keep = set(), []
        for m, p_ in lst:
            if p_ not in seen and not (nodes[target]["s"] <= p_ < nodes[target]["e"]):
                seen.add(p_)
                keep.append(p_)
            if len(keep) >= 24:
                break
        nodes[target]["x"] = keep
    for nd in nodes:
        nd.pop("c", None)
        nd["r"] = [int(pos[i]) for i in nd["r"]]
    depth = max(nd["d"] for nd in nodes)
    tree = {"format": THEMEN_FORMAT, "root": root, "built": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "built_ts": time.time(), "model": base.key, "n": n, "depth": depth,
            "paths": [rel_paths[i] for i in order], "nodes": nodes, "alt": alt_map,
            "basis": {"kacheln": with_t, "geprueft": sum(1 for i in ids_g if int(i) in r_set) if r_set else 0,
                      "suchen": [themen_pretty(h) for h in hit_names]}}
    fn = themen_file(root)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with gzip.open(fn + ".tmp", "wt", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(fn + ".tmp", fn)
    log.info("Themen berechnet: %d Bilder in %d Gruppen, %d Ebenen, %d Querverweise (%.0fs).",
             n, len(nodes) - 1, depth, len(alt_map), time.time() - t0)
    return tree


def themen_name_all(nodes, D, H, vocab, concepts, order, rel_paths):
    """Namen je Ebene. Für Geschwister gilt:
      - ABDECKUNG: Der Name soll möglichst alle Untergruppen der Gruppe erfassen. Gemessen wird der Anteil
        der Bilder, die zu den Begriffen des Namens passen – gemittelt über die Untergruppen, damit eine
        große Untergruppe nicht alles bestimmt. Dadurch wird der Name so allgemein wie nötig.
      - ABGRENZUNG: Abgezogen wird, wie gut derselbe Name auch das beste Geschwister beschreibt. Dadurch
        wird der Name so speziell wie möglich.
      - Namen von Geschwistern dürfen keine Begriffe teilen (keine Überschneidung), und kein Name
        wiederholt einen Namen weiter oben im Pfad.
    Reicht ein Begriff nicht, sind zwei verbunden möglich ("Berge & Seen")."""
    from collections import Counter
    term_concepts = [c for c in concepts if c["kind"] != "suche"]
    M = np.zeros((len(vocab), len(term_concepts)), np.float32)
    for ci, c in enumerate(term_concepts):
        for j in c["terms"]:
            M[j, ci] = 1.0
    search_c = [c for c in concepts if c["kind"] == "suche"]
    allc = term_concepts + search_c

    def cov_vec(x):
        """Abdeckung aller Namen für Gruppe x (über die Untergruppen gemittelt)."""
        parts = nodes[x]["k"] or [x]
        out = []
        for p_ in parts:
            v = D[p_] @ M
            if search_c:
                v = np.concatenate([v, np.array([H[p_][c["h"]] for c in search_c], np.float32)])
            out.append(v)
        return np.mean(out, axis=0)

    dpos = [os.path.dirname(rel_paths[i]) for i in order]
    for nd in nodes[1:]:                           # Ordner/Jahr-Hinweis (wie bisher)
        size = nd["e"] - nd["s"]
        dirs = Counter(dpos[nd["s"]:nd["e"]])
        d, c = dirs.most_common(1)[0]
        if d and c >= 0.7 * size:
            nd["folder"] = d.split("/")[-1]
            nd["w"] = "Ordner " + nd["folder"] if c == size else f"meist Ordner {nd['folder']} ({100 * c // size} %)"
            continue
        years = Counter(d.split("/")[0] for d in dpos[nd["s"]:nd["e"]])
        y, c = years.most_common(1)[0]
        if re.fullmatch(r"(19|20)\d\d", y) and c >= 0.7 * size:
            nd["w"] = f"aus {y}" if c == size else f"meist aus {y} ({100 * c // size} %)"

    nodes[0]["name"] = "Alle Bilder"
    nodes[0]["terms"] = []
    lam = 1.0
    for nid, nd in enumerate(nodes):
        kids = nd["k"]
        if not kids:
            continue
        used_above, x = set(), nid
        while x > 0:
            used_above.add(nodes[x]["name"])
            x = nodes[x]["p"]
        pterms = set(nd.get("terms") or [])         # Untergruppen nie allgemeiner benennen als die Gruppe selbst
        B = np.vstack([cov_vec(k) for k in kids])  # (Geschwister, Namen)
        if len(kids) > 1:
            srt = np.sort(B, axis=0)
            top1, top2 = srt[-1], srt[-2]
            other = np.where(B >= top1, top2, top1)    # bestes ANDERES Geschwister je Name
        else:
            other = np.zeros_like(B)
        S = B - lam * other
        cands = []                                    # (Wert, Geschwister, Name, Begriffsmenge)
        for r in range(len(kids)):
            order_c = [ci for ci in np.argsort(-S[r]) if allc[ci]["name"] not in used_above
                       and not (pterms and pterms < {t for t in allc[ci]["terms"] if isinstance(t, int)})][:10]
            for ci in order_c:
                cands.append((float(S[r, ci]), r, allc[ci]["name"], allc[ci]["terms"], float(B[r, ci])))
            for x1 in range(min(6, len(order_c))):     # Paare aus den besten Einzelnamen
                for x2 in range(x1 + 1, min(6, len(order_c))):
                    c1, c2 = allc[order_c[x1]], allc[order_c[x2]]
                    if c1["terms"] & c2["terms"]:
                        continue
                    cov = B[r, order_c[x1]] + B[r, order_c[x2]]
                    oth = max((B[o, order_c[x1]] + B[o, order_c[x2]] for o in range(len(kids)) if o != r), default=0.0)
                    val = float(cov - lam * oth - 0.06)        # kleiner Abzug für den längeren Namen
                    joint = " · " if "&" in c1["name"] + c2["name"] else " & "
                    cands.append((val, r, c1["name"] + joint + c2["name"], c1["terms"] | c2["terms"], float(cov)))
        cands.sort(key=lambda t: -t[0])
        names, taken = {}, []
        for val, r, name, terms, cov in cands:
            if r in names or any(terms & t for t in taken):
                continue
            names[r] = (name, terms, val, cov)
            taken.append(terms)
        for r, kid in enumerate(kids):
            knd = nodes[kid]
            if r in names and (names[r][2] > 0.04 or not knd.get("folder")):
                knd["name"], terms, val, cov = names[r]
                knd["score"] = round(val, 3)
            elif knd.get("folder"):                    # nichts Unterscheidendes: der Ordner sagt es besser
                knd["name"], terms = knd["folder"], frozenset()
                knd.pop("w", None)
            else:                                      # Notlösung: bester Name + häufigster eigener Begriff
                ci = int(np.argmax(S[r]))
                own = next((vocab[j] for j in np.argsort(-D[kid]) if j not in allc[ci]["terms"]), "")
                knd["name"], terms = f"{allc[ci]['name']} ({own})" if own else allc[ci]["name"], allc[ci]["terms"]
            knd["terms"] = sorted(t for t in terms if isinstance(t, int))
    # Beispiele (häufigste Einzelbegriffe, die nicht schon im Namen stecken) und Anteile für den Tooltip
    for x, nd in enumerate(nodes):
        d = D[x]
        top = np.argsort(-d)
        nd["ex"] = [vocab[j] for j in top[:6] if vocab[j] != nd["name"] and d[j] >= 0.01][:3]
        nd["top"] = [[vocab[j], round(float(d[j]) * 100)] for j in top[:6] if d[j] >= 0.01]
        nd.pop("terms", None)
    for x in range(1, len(nodes)):
        nodes[x]["pos"], nodes[x]["neg"] = themen_search_terms(x, nodes, D, vocab)


def themen_search_terms(x, nodes, D, vocab):
    """Vorschlag für eine Suche, die diese Gruppe im GANZEN Archiv findet.
    Positiv: Begriffe, die in der Gruppe häufig und im Archiv insgesamt selten sind (Anteil · log Lift).
    Negativ: Begriffe, die in den ähnlichsten Nachbargruppen häufig, in dieser Gruppe aber selten sind –
    genau die Bilder, die eine Suche sonst fälschlich mitnehmen würde."""
    eps = 1e-6
    d, g = D[x], D[0]
    score = d * np.log((d + eps) / (g + eps))
    pos = []
    for j in np.argsort(-score)[:8]:
        if d[j] < 0.03 or score[j] <= 0:
            continue
        if pos and d[j] < 0.3 * d[pos[0]]:
            continue
        pos.append(int(j))
        if len(pos) == 4:
            break
    if not pos:
        pos = [int(np.argmax(d))]
    # Nachbarschaft: Geschwister (nach Ähnlichkeit der Begriffsverteilung gewichtet), dann die Ebene darüber
    neigh, wsum = np.zeros_like(d), 0.0
    p_, weight = nodes[x]["p"], 1.0
    me = x
    while p_ >= 0 and weight > 0.2:
        for sib in nodes[p_]["k"]:
            if sib == me:
                continue
            sim = float(D[sib] @ d / (np.linalg.norm(D[sib]) * np.linalg.norm(d) + eps))
            neigh += weight * sim * D[sib]
            wsum += weight * sim
        me, p_, weight = p_, nodes[p_]["p"], weight * 0.5
    neg = []
    if wsum > 0:
        neigh /= wsum
        nscore = neigh - 2.0 * d
        for j in np.argsort(-nscore):
            if len(neg) == 5 or nscore[j] <= 0.005:
                break
            if int(j) not in pos:
                neg.append(int(j))
    return [vocab[j] for j in pos], [vocab[j] for j in neg]


def themen_yaml(nd, path):
    """Konfigurationsblock für searches.yaml (gleiches Format wie beim Assistenten)."""
    q = {"name": assist_slug(nd["name"]), "terms": nd.get("pos") or [nd["name"]],
         "own_negatives": (nd.get("neg") or []) + ["ein Foto"],
         "threshold": STRICTNESS["ausgewogen"][0],
         "coarse_threshold": round(STRICTNESS["ausgewogen"][0] - STRICTNESS["ausgewogen"][1], 2)}
    lines = assist_yaml_block(q, path).splitlines()
    lines[0] = f"  # Themen-Gruppe „{' '.join(path.split())[:100]}“ – Vorschlag vom {datetime.now():%d.%m.%Y}"
    return "\n".join(lines) + "\n"


def themen_job(src, root, reason):
    # Bewusst KEINE niedrigere Priorität: Die Bildsuche rechnet ständig, ein gebremster Thread
    # bekäme dann fast keine Rechenzeit (2026-11-12a: "Begriffe vorbereiten" hing so über 10 Minuten).
    THEMEN_STATE.update(running=True, error="", message="Berechnung startet …", t_start=time.time(), holds_index=True)
    log.info("Themen: Berechnung gestartet (%s, %d Bilder).", reason, len(src["rel"]))
    try:
        tree = themen_build(src, root)
        src.clear()                                # Index freigeben (der Neu-Scan wartet darauf)
        THEMEN_STATE["holds_index"] = False
        gc.collect()
        todo = []
        for nd in tree["nodes"]:
            if nd["d"] <= 2:
                todo.extend(nd["r"])
        todo = list(dict.fromkeys(todo))
        themen_phase(6, len(todo))
        for x, k in enumerate(todo):
            themen_tick(x + 1)
            with THEMEN_THUMB_SEM:
                thumb_for(os.path.join(root, tree["paths"][k]))
        THEMEN_STATE.update(message="", step=0, phase="", done=0, total=0)
        log.info("Themen: fertig (%.0f Minuten).", (time.time() - THEMEN_STATE["t_start"]) / 60)
    except Exception as e:
        THEMEN_STATE.update(error=f"Berechnung fehlgeschlagen: {e}", message="")
        log.error("Themen: Berechnung fehlgeschlagen", exc_info=True)
    finally:
        src.clear()
        THEMEN_STATE.update(running=False, holds_index=False)


def themen_maybe_start(eng, force=False):
    """Startet die Themen-Berechnung im Hintergrund, wenn noch keine da ist oder sich genug geändert hat.
    Voraussetzung: Stufe 1 ist (fast) vollständig. Stufe 2 und 3 werden genutzt, soweit schon gerechnet."""
    if eng is None or THEMEN_STATE["running"]:
        return False, "Die Berechnung läuft bereits."
    s = eng.store
    n, have = len(s.paths), int(s.has_g.sum())
    if not have or (not force and have < 0.98 * n):
        return False, (f"Stufe 1 ist erst zu {100 * have / max(1, n):.0f} % fertig – "
                       "die Themen werden danach automatisch berechnet.")
    root = eng.settings["photo_root"]
    ids = np.flatnonzero(s.has_g)
    rel = [os.path.relpath(s.paths[i], root) for i in ids]
    reason = "auf Wunsch" if force else None
    if not force:
        tree = themen_tree(root)
        if tree is None:
            reason = "noch keine Themen vorhanden"
        else:
            old = set(tree["paths"])
            new = sum(1 for p in rel if p not in old)
            gone = len(old) - (len(rel) - new)
            if new + gone >= max(500, THEMEN_AUTO_NEW * len(rel)):
                reason = f"{new} neue, {gone} entfernte Bilder"
        if reason is None:
            return False, "Die Themen sind aktuell."
    try:
        hits = themen_search_hits(eng.settings, eng.searches)
    except Exception as e:
        log.warning("Themen: Suchtreffer nicht lesbar (%s)", e)
        hits = {}
    # Keine Kopie der Vektoren (das wären bis zu 1 GB): Der Thread liest direkt aus dem Index.
    # Dafür wartet ein Neu-Scan, bis er fertig ist (THEMEN_STATE["holds_index"]).
    strong = eng.strong if (eng.strong is not None and eng.settings.get("rerank", True)) else None
    src = {"store": s, "ids": ids, "rel": rel, "base": eng.base, "strong": strong, "hits": hits}
    THEMEN_STATE.update(running=True, holds_index=True, message="Berechnung startet …", error="", step=0,
                        done=0, total=0, t_start=time.time())
    threading.Thread(target=themen_job, args=(src, root, reason), name="themen", daemon=True).start()
    return True, "Die Berechnung wurde gestartet. Das dauert einige Minuten."


def _themen_leaf_of(tree, k):
    """Kleinste Gruppe, in der das Bild an Position k steht (über die sortierten Abschnitte)."""
    import bisect
    li = tree.get("_leaves")
    if li is None:
        leaves = sorted((x for x, nd in enumerate(tree["nodes"]) if not nd["k"]), key=lambda x: tree["nodes"][x]["s"])
        li = tree["_leaves"] = (leaves, [tree["nodes"][x]["s"] for x in leaves])
    j = bisect.bisect_right(li[1], k) - 1
    return li[0][max(0, j)]


def _themen_path_name(nodes, x, levels=2):
    parts = []
    while x > 0 and len(parts) < levels:
        parts.append(nodes[x]["name"])
        x = nodes[x]["p"]
    return " › ".join(parts[::-1])


def themen_api(qs):
    """JSON für die Themen-Seite: Knoten mit Untergruppen oder (Blatt / "alle") eine Seite Bilder."""
    tree = themen_tree()
    status = themen_status()
    if tree is None:
        eng = ENGINE[0]
        hint = ""
        if eng is not None and not THEMEN_STATE["running"]:
            n, have = len(eng.store.paths), int(eng.store.has_g.sum())
            hint = (f"Stufe 1 ist zu {100 * have / max(1, n):.0f} % fertig." if have < 0.98 * n else "")
        return {"ready": False, "status": status, "hint": hint}
    nodes = tree["nodes"]
    try:
        nid = int((qs.get("n") or ["0"])[0])
        page = max(0, int((qs.get("p") or ["0"])[0]))
    except ValueError:
        nid, page = 0, 0
    if not 0 <= nid < len(nodes):
        nid = 0
    nd = nodes[nid]
    crumbs, x = [], nid
    while x >= 0:
        crumbs.append({"n": x, "t": nodes[x]["name"]})
        x = nodes[x]["p"]
    alt = tree.get("alt", {})

    def img(k, home=False):
        it = {"k": k, "rel": tree["paths"][k]}
        if home:                                   # Bild aus einer anderen Gruppe: Link zurück zur eigenen
            h = _themen_leaf_of(tree, k)
            it["home"] = {"n": h, "t": _themen_path_name(nodes, h)}
        it["alt"] = [{"n": a2, "t": _themen_path_name(nodes, a2)} for a2 in alt.get(str(k), [])]
        return it

    out = {"ready": True, "status": status, "built": tree["built"], "total": tree["n"],
           "groups": len(nodes) - 1, "depth": tree["depth"], "v": int(tree["built_ts"]),
           "basis": tree.get("basis", {}),
           "n": nid, "t": nd["name"], "ex": nd.get("ex", []), "size": nd["e"] - nd["s"], "crumbs": crumbs[::-1],
           "kids": [{"n": k, "t": nodes[k]["name"], "ex": nodes[k].get("ex", []), "size": nodes[k]["e"] - nodes[k]["s"],
                     "sub": len(nodes[k]["k"]), "r": nodes[k]["r"], "top": nodes[k].get("top", []),
                     "w": nodes[k].get("w", "")}
                    for k in nd["k"]],
           "top": nd.get("top", []), "w": nd.get("w", ""),
           "search": ({"pos": nd.get("pos", []), "neg": (nd.get("neg") or []) + ["ein Foto"],
                       "yaml": themen_yaml(nd, _themen_path_name(nodes, nid, levels=9))} if nid > 0 and "pos" in nd else None),
           "also": [img(k, home=True) for k in nd.get("x", [])]}
    if not nd["k"] or (qs.get("alle") or ["0"])[0] == "1":
        size = nd["e"] - nd["s"]
        pages = max(1, -(-size // THEMEN_PAGE))
        page = min(page, pages - 1)
        a = nd["s"] + page * THEMEN_PAGE
        b = min(nd["e"], a + THEMEN_PAGE)
        out.update(page=page, pages=pages, images=[img(k) for k in range(a, b)])
    return out


def themen_image(qs):
    """Vorschau (240 px) oder große Ansicht (1600 px) eines Bildes aus dem Themenbaum – nur über die
    Position im Baum, nie über einen Pfad aus der Anfrage."""
    tree = themen_tree()
    if tree is None:
        return None
    try:
        k = int((qs.get("k") or ["-1"])[0])
    except ValueError:
        return None
    if not 0 <= k < len(tree["paths"]):
        return None
    full = os.path.join(tree["root"], tree["paths"][k])
    if not os.path.isfile(full):
        return None
    with THEMEN_THUMB_SEM:
        if (qs.get("g") or ["0"])[0] == "1":
            return assist_thumb(full, big=True)
        return thumb_for(full)


THEMEN_HTML = r"""<!doctype html><html lang="de"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bildsuche – Themen</title>
<style>
:root{--bg:#f6f6f4;--fg:#222;--card:#fff;--line:#e5e5e5;--mut:#888;--acc:#2a6fdb;--tile:#ddd}
@media (prefers-color-scheme:dark){:root{--bg:#1c1c1e;--fg:#eee;--card:#2c2c2e;--line:#444;--mut:#999;--tile:#3a3a3c}}
body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
main{max-width:1200px;margin:0 auto;padding:20px 16px 40px}
a{color:var(--acc);text-decoration:none} a:hover{text-decoration:underline}
h1{margin:0 0 4px;font-size:26px} .meta{color:var(--mut);font-size:14px;line-height:1.6}
.top{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.crumbs{margin:16px 0 6px;font-size:15px;line-height:1.8}
.crumbs a,.crumbs span{white-space:nowrap} .crumbs .sep{color:var(--mut);margin:0 6px}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:4px 0 14px}
.head h2{margin:0;font-size:21px;flex:1 1 auto}
button{font:inherit;font-size:14px;padding:7px 12px;border-radius:8px;border:1px solid var(--line);
  background:var(--card);color:var(--fg);cursor:pointer}
button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
button:disabled{opacity:.5;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.card{position:relative;border-radius:12px;overflow:hidden;cursor:pointer;border:1px solid var(--line);
  background:var(--tile);transition:transform .12s, box-shadow .12s}
.card:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(0,0,0,.18)}
.card:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.mos{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:2px;aspect-ratio:1/1;background:var(--tile)}
.mos img,.mos i{width:100%;height:100%;object-fit:cover;display:block;background:var(--tile)}
.mos.one{grid-template-columns:1fr;grid-template-rows:1fr}
/* Beschriftung liegt auf den Bildern: schwarze Schrift, halbtransparente weiße Box – auf jedem Bild lesbar */
.cap{position:absolute;left:8px;right:8px;bottom:8px;padding:8px 10px;border-radius:8px;color:#111;
  background:rgba(255,255,255,.86);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px);
  box-shadow:0 1px 4px rgba(0,0,0,.15)}
.cap b{display:block;font-size:15px;line-height:1.3}
.cap em{display:block;font-style:normal;font-size:12.5px;color:#333;line-height:1.35;margin-top:1px}
.cap .wo{display:block;font-size:12px;color:#444;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cap small{display:block;color:#444;font-size:12px;margin-top:2px}
.mix{font-size:13px;color:var(--mut);margin:-8px 0 14px;line-height:1.6}
.imgs{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.imgs figure{position:relative;margin:0;border-radius:8px;overflow:hidden;border:1px solid var(--line);background:var(--tile)}
.imgs img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--tile)}
.imgs figcaption{position:absolute;left:4px;right:4px;bottom:4px;font-size:11px;color:#111;padding:3px 6px;border-radius:5px;
  background:rgba(255,255,255,.86);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none}
.imgs .tag{position:absolute;top:4px;left:4px;right:4px;width:max-content;max-width:calc(100% - 8px);box-sizing:border-box;
  font-size:10.5px;color:#111;background:rgba(255,255,255,.86);padding:2px 6px;border-radius:5px;pointer-events:none;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lbextra{display:none}
.srch{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin:-4px 0 16px;font-size:13.5px;line-height:1.6}
.srch summary{cursor:pointer;font-weight:600}
.srch .row{margin-top:4px}
.chip{display:inline-block;padding:1px 8px;margin:2px 3px 2px 0;border-radius:10px;font-size:12.5px}
.chip.p{background:#dff3e4;color:#14532d} .chip.n{background:#fbe1df;color:#7f1d1d}
.srch pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:8px;overflow:auto;font-size:12px;margin:8px 0 6px}
.also{margin-top:30px}
.also h3{font-size:16px;margin:0 0 2px}
@media (max-width:600px){.grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
  .cap{left:5px;right:5px;bottom:5px;padding:6px 7px} .cap b{font-size:13.5px} .cap em,.cap .wo{display:none}
  .imgs{grid-template-columns:repeat(3,minmax(0,1fr));gap:4px} .imgs figcaption{display:none}
  .head h2{font-size:18px;flex-basis:100%}}
.pager{display:flex;gap:10px;align-items:center;justify-content:center;margin:18px 0}
.box{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:18px;line-height:1.6}
.err{color:#c0392b}
#lbsel{display:none}
.prog{margin-top:10px;max-width:560px}
.prog .ph{font-size:15px;margin-bottom:6px}
.pbar{height:8px;background:var(--tile);border-radius:4px;overflow:hidden;margin-bottom:4px}
.pbar>div{height:100%;background:var(--acc);transition:width .6s}
.steps{margin:8px 0 4px;padding-left:20px;font-size:13px;color:var(--mut);line-height:1.7}
.steps li.ok{color:#2e8b57} .steps li.cur{color:var(--fg);font-weight:600}
</style></head><body><main>
<div class="top"><h1>Themen</h1><a href="/">← Übersicht</a></div>
<div class="meta" id="info">wird geladen …</div>
<div id="view"></div>
</main>
<script>
(function(){
  const view = document.getElementById('view'), info = document.getElementById('info');
  const fmt = n => Number(n).toLocaleString('de-DE');
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  let V = 0, poll = null;
  function state(){ const h = new URLSearchParams(location.hash.slice(1));
    return {n: +(h.get('n') || 0), alle: h.get('alle') === '1', p: +(h.get('p') || 0)}; }
  function go(n, alle, p){ location.hash = 'n=' + n + (alle ? '&alle=1' : '') + (p ? '&p=' + p : ''); }
  const thumb = k => '/api/themen/bild?k=' + k + '&v=' + V;
  const big = k => '/api/themen/bild?g=1&k=' + k + '&v=' + V;

  function statusLine(st){
    if (st.error) return '<span class="err">' + esc(st.error) + '</span>';
    return st.message ? esc(st.message) : '';
  }
  const dur = s => s >= 3600 ? Math.floor(s / 3600) + ' h ' + Math.round(s % 3600 / 60) + ' min'
                 : s >= 60 ? Math.round(s / 60) + ' min' : Math.max(1, s) + ' s';
  function progress(st, ready){
    if (st.error) return '<div class="err">' + esc(st.error) + '</div>';
    if (!st.running) return '';
    let h = '<div class="prog"><div class="ph">' + (st.step ? 'Schritt ' + st.step + ' von ' + st.steps + ': <b>' + esc(st.phase) + '</b>'
                                                      : esc(st.message || 'Berechnung startet …')) + '</div>';
    const pct = st.total ? Math.min(100, 100 * st.done / st.total) : 0;
    h += '<div class="pbar"><div style="width:' + pct.toFixed(1) + '%"></div></div><div class="meta">';
    const bits = [];
    if (st.total) bits.push(fmt(st.done) + ' von ' + fmt(st.total) + ' (' + Math.floor(pct) + ' %)');
    if (st.elapsed) bits.push('läuft seit ' + dur(st.elapsed));
    if (st.eta !== null && st.eta !== undefined && st.done < st.total) bits.push('noch ca. ' + dur(st.eta));
    h += bits.join(' · ') + '</div>';
    const steps = ['Begriffe vorbereiten', 'Bildmerkmale zusammenführen', 'Gruppen bilden', 'Gruppen benennen',
                   'Querverweise suchen', 'Vorschaubilder vorbereiten'];
    h += '<ol class="steps">' + steps.map((t, i) => '<li class="' + (i + 1 < st.step ? 'ok' : i + 1 === st.step ? 'cur' : '') + '">' +
      (i + 1 < st.step ? '✓ ' : '') + t + '</li>').join('') + '</ol>';
    if (ready) h += '<div class="meta">Die Ansicht unten ist schon nutzbar' + (st.step === 6 ? ' – Vorschaubilder erscheinen anfangs etwas langsamer.' : ' (bisherige Einteilung).') + '</div>';
    return h + '</div>';
  }
  function schedule(st){ clearTimeout(poll); if (st.running) poll = setTimeout(refreshStatus, 3000); }
  let lastReady = null;
  async function refreshStatus(){                         // nur den Fortschritt nachladen, nicht die Ansicht
    let j; try { j = await (await fetch('/api/themen/status')).json(); } catch (e) { poll = setTimeout(refreshStatus, 8000); return; }
    if (j.ready !== lastReady || (!j.running && !j.error)) { load(); return; }
    const box = document.getElementById('progbox'); if (box) box.innerHTML = progress(j, j.ready);
    schedule(j);
  }

  async function rebuild(btn){
    if (!confirm('Themen jetzt neu berechnen?\n\nDas dauert einige Minuten und läuft im Hintergrund. Bis dahin bleibt die bisherige Einteilung sichtbar.')) return;
    btn.disabled = true;
    try { const r = await fetch('/api/themen/neu', {method: 'POST'}); const j = await r.json();
      info.textContent = j.message; setTimeout(load, 3000);
    } catch (e) { info.textContent = 'Fehler: ' + e; btn.disabled = false; }
  }

  async function load(){
    const s = state();
    let j;
    try {
      const r = await fetch('/api/themen?n=' + s.n + '&p=' + s.p + (s.alle ? '&alle=1' : ''));
      j = await r.json();
    } catch (e) { info.textContent = 'Die Bildsuche ist gerade nicht erreichbar (' + e + ').'; return; }
    schedule(j.status);
    lastReady = j.ready;
    if (!j.ready){
      info.innerHTML = '';
      view.innerHTML = '<div class="box"><b>' + (j.status.running ? 'Die Themen werden berechnet.' : 'Die Themen sind noch nicht berechnet.') + '</b>' +
        (j.status.running ? '<div id="progbox">' + progress(j.status, false) + '</div><div class="meta">Diese Seite aktualisiert sich von selbst.</div>'
          : '<br>' + esc(j.hint || 'Sie werden automatisch berechnet, sobald Stufe 1 fertig ist.') + ' ' + statusLine(j.status) +
            '<p><button class="pri" id="rb">Jetzt berechnen</button></p>') + '</div>';
      const rb = document.getElementById('rb'); if (rb) rb.addEventListener('click', () => rebuild(rb));
      return;
    }
    V = j.v;
    info.innerHTML = fmt(j.total) + ' Bilder, automatisch nach Ähnlichkeit in ' + fmt(j.groups) + ' Gruppen auf ' +
      j.depth + ' Ebenen sortiert · berechnet am ' + esc(j.built) + basis(j.basis) +
      (j.status.running || j.status.error ? '' : ' · <a href="#" id="rb">neu berechnen</a>') +
      '<div id="progbox">' + progress(j.status, true) + '</div>';
    const rb = document.getElementById('rb');
    if (rb) rb.addEventListener('click', ev => { ev.preventDefault(); rebuild(rb); });

    let h = '<nav class="crumbs">' + j.crumbs.map((c, i) => i === j.crumbs.length - 1 && !s.alle
      ? '<span>' + esc(c.t) + '</span>' : '<a href="#n=' + c.n + '">' + esc(c.t) + '</a>').join('<span class="sep">›</span>') +
      (s.alle ? '<span class="sep">›</span><span>alle Bilder</span>' : '') + '</nav>';
    h += '<div class="head"><h2>' + esc(j.t) + ' <small class="meta">' + fmt(j.size) + ' Bilder</small></h2>';
    if (j.crumbs.length > 1) h += '<button id="up">↑ Eine Ebene höher</button>';
    if (j.kids.length) h += s.alle ? '<button id="sw">▦ Untergruppen zeigen</button>'
                                   : '<button id="sw">▤ Alle ' + fmt(j.size) + ' Bilder ansehen</button>';
    h += '</div>';
    if (j.top && j.top.length || j.w) h += '<div class="mix">' + (j.top && j.top.length ? 'Häufigste Motive: ' + esc(mix(j.top)) : '') +
      (j.w ? (j.top && j.top.length ? ' · ' : '') + esc(j.w) : '') + '</div>';
    if (j.search) {
      const chips = (a, c) => a.map(t => '<span class="chip ' + c + '">' + esc(t) + '</span>').join('');
      h += '<details class="srch"><summary>Als Suche verwenden</summary>' +
        '<div class="row"><b>Positiv</b> (das soll auf dem Bild sein): ' + chips(j.search.pos, 'p') + '</div>' +
        '<div class="row"><b>Negativ</b> (damit wird es leicht verwechselt): ' + chips(j.search.neg, 'n') + '</div>' +
        '<div class="meta">Abgeleitet aus dieser Gruppe: Positiv sind Begriffe, die hier häufig und im übrigen Archiv selten sind; ' +
        'negativ sind Begriffe, die in den ähnlichsten Nachbargruppen häufig sind, hier aber nicht. ' +
        'Als Eintrag für die Konfiguration (unter <code>searches:</code> einfügen):</div>' +
        '<pre id="yaml">' + esc(j.search.yaml) + '</pre><button class="pri" id="asst">Im Assistenten öffnen</button> ' +
        '<button id="cp">Kopieren</button> <span class="meta" id="cpmsg"></span>' +
        '<div class="meta">Der Assistent übernimmt die Begriffe, zeigt Probebilder aus deinem Archiv und stellt die Schwellen ' +
        'anhand deiner Bewertung ein – gespeichert wird erst, wenn die Probebilder passen.</div></details>';
    }
    if (j.images){
      h += pager(j) + '<div class="imgs">' + j.images.map(im => fig(im)).join('') + '</div>' + pager(j);
    } else {
      h += '<div class="grid">' + j.kids.map(k => {
        const r = k.r.slice(0, 4);
        const tiles = r.length >= 4 ? r.map(x => '<img loading="lazy" src="' + thumb(x) + '" alt="">').join('')
                    : r.length ? '<img loading="lazy" src="' + thumb(r[0]) + '" alt="">' : '<i></i>';
        return '<div class="card" tabindex="0" data-n="' + k.n + '" title="' + esc(mix(k.top)) + '"><div class="mos' + (r.length >= 4 ? '' : ' one') + '">' +
          tiles + '</div><div class="cap"><b>' + esc(k.t) + '</b>' +
          (k.ex && k.ex.length ? '<em>z. B. ' + esc(k.ex.join(' · ')) + '</em>' : '') +
          (k.w ? '<span class="wo">' + esc(k.w) + '</span>' : '') + '<small>' + fmt(k.size) + ' Bilder' +
          (k.sub ? ' · ' + k.sub + ' Untergruppen' : ' · Einzelbilder') + '</small></div></div>';
      }).join('') + '</div>';
    }
    if (j.also && j.also.length) {
      h += '<section class="also"><h3>Passt auch hierher</h3><div class="meta" style="margin-bottom:10px">' +
        'Bilder aus anderen Gruppen, die fast genauso gut zu „' + esc(j.t) + '“ passen. Sie bleiben in ihrer eigenen Gruppe; ' +
        'in der großen Ansicht führt ein Link dorthin.</div><div class="imgs">' + j.also.map(im => fig(im, true)).join('') + '</div></section>';
    }
    view.innerHTML = h;
    const up = document.getElementById('up');
    if (up) up.addEventListener('click', () => go(j.crumbs[j.crumbs.length - 2].n));
    const asst = document.getElementById('asst');
    if (asst) asst.addEventListener('click', () => { location.href = '/assistent?thema=' + j.n; });
    const cp = document.getElementById('cp');
    if (cp) cp.addEventListener('click', () => {          // ohne HTTPS gibt es navigator.clipboard nicht -> Textfeld
      const ta = document.createElement('textarea'); ta.value = j.search.yaml; document.body.appendChild(ta); ta.select();
      let ok = false; try { ok = document.execCommand('copy'); } catch (e) {}
      ta.remove(); document.getElementById('cpmsg').textContent = ok ? 'Kopiert.' : 'Bitte den Text oben markieren und kopieren.';
    });
    const sw = document.getElementById('sw');
    if (sw) sw.addEventListener('click', () => go(j.n, !s.alle));
    view.querySelectorAll('.card').forEach(c => {
      c.addEventListener('click', () => go(+c.dataset.n));
      c.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(+c.dataset.n); });
    });
    view.querySelectorAll('.pager button').forEach(b => b.addEventListener('click', () => go(j.n, s.alle, +b.dataset.p)));
    window.scrollTo(0, 0);
  }
  function fig(im, foreign){
    let ex = '';
    if (im.home) ex += 'Eigene Gruppe: <a href="#n=' + im.home.n + '">' + esc(im.home.t) + '</a>';
    if (im.alt && im.alt.length) ex += (ex ? '<br>' : '') + 'Passt auch zu: ' +
      im.alt.map(a => '<a href="#n=' + a.n + '">' + esc(a.t) + '</a>').join(' · ');
    return '<figure><a class="open" href="' + big(im.k) + '"><img loading="lazy" src="' + thumb(im.k) + '" alt=""></a>' +
      (foreign ? '<span class="tag">aus: ' + esc(im.home ? im.home.t : '') + '</span>'
               : (im.alt && im.alt.length ? '<span class="tag">↔ passt auch woanders</span>' : '')) +
      '<figcaption title="' + esc(im.rel) + '">' + esc(im.rel.split('/').slice(-2).join(' / ')) + '</figcaption>' +
      (ex ? '<div class="lbextra">' + ex + '</div>' : '') + '</figure>';
  }
  function basis(b){
    if (!b) return '';
    const p = [];
    if (b.kacheln) p.push(fmt(b.kacheln) + ' mit Kacheln');
    if (b.geprueft) p.push(fmt(b.geprueft) + ' geprüft');
    if (b.suchen && b.suchen.length) p.push('Suchen: ' + b.suchen.map(esc).join(', '));
    return p.length ? '<br>Grundlage: alle Gesamtbilder · ' + p.join(' · ') : '';
  }
  function mix(top){ return (top || []).map(x => x[0] + ' ' + x[1] + ' %').join(' · '); }
  function pager(j){
    if (j.pages <= 1) return '';
    return '<div class="pager"><button data-p="' + (j.page - 1) + '"' + (j.page ? '' : ' disabled') + '>‹ zurück</button>' +
      '<span class="meta">Seite ' + (j.page + 1) + ' von ' + j.pages + '</span>' +
      '<button data-p="' + (j.page + 1) + '"' + (j.page < j.pages - 1 ? '' : ' disabled') + '>weiter ›</button></div>';
  }
  document.addEventListener('keydown', ev => {          // Pfeil hoch / Backspace = herauszoomen
    const lb = document.getElementById('lb');
    if ((lb && !lb.hidden) || ev.target.tagName === 'INPUT') return;
    if (ev.key === 'Backspace' || (ev.key === 'ArrowUp' && ev.altKey)) { const u = document.getElementById('up'); if (u) { ev.preventDefault(); u.click(); } }
  });
  window.addEventListener('hashchange', load);
  load();
})();
</script>
"""


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
            if not getattr(self, "_cacheable", False):
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
                    f'alle 30 Sekunden neu.</p><p><a href="/themen">Bilder nach Themen erkunden</a> · '
                    f'<a href="/config">Konfiguration ansehen &amp; bearbeiten</a></p>'
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
            if path == "/assistent":
                return self._send_bytes(ASSIST_PAGE, "text/html; charset=utf-8")
            if path == "/themen":
                return self._send_bytes(THEMEN_HTML + LIGHTBOX + "\n</body></html>", "text/html; charset=utf-8")
            if path == "/api/themen":
                return self._json(200, themen_api(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)))
            if path == "/api/themen/status":
                st = themen_status()
                st["ready"] = themen_tree() is not None
                return self._json(200, st)
            if path == "/api/themen/bild":
                try:
                    fn = themen_image(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query))
                except Exception as e:
                    log.warning("Themen: Vorschau nicht möglich (%s)", e)
                    fn = None
                if not fn:
                    return self.send_error(404)
                with open(fn, "rb") as fh:
                    data = fh.read()
                self._cacheable = True             # Vorschaubilder darf der Browser eine Woche behalten
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "private, max-age=604800")
                self.end_headers()
                self._cacheable = False
                return self.wfile.write(data)
            if path == "/api/assistent/info":
                eng = ENGINE[0]
                try:
                    settings, searches = load_config()
                except Exception:
                    settings, searches = DEFAULTS, []
                return self._json(200, {"ready": bool(eng is not None and eng.store.has_g.any()),
                                        "names": [q["name"] for q in searches],
                                        "default_negatives": list(settings.get("default_negatives") or [])})
            if path == "/api/assistent/bild":
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                with PROBE_LOCK:
                    probe = PROBES.get((qs.get("t") or [""])[0])
                try:
                    it = probe["items"][int((qs.get("i") or ["-1"])[0])]
                    fn = assist_thumb(it["path"], big=(qs.get("gross") or ["0"])[0] == "1")
                    with open(fn, "rb") as fh:
                        return self._send_bytes(fh.read(), "image/jpeg")
                except Exception:
                    return self._json(404, {"message": "Bild nicht verfügbar."})
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
            if urllib.parse.urlparse(self.path).path == "/api/themen/neu":
                if ENGINE[0] is None:
                    return self._json(409, {"message": "Die Bildsuche ist noch nicht bereit – bitte gleich noch einmal."})
                started, msg = themen_maybe_start(ENGINE[0], force=True)
                if started:
                    log.info("Themen: Neuberechnung über die Webseite angestoßen.")
                return self._json(202 if started else 409, {"message": msg})
            if self.path.startswith("/api/assistent/"):
                try:
                    ln = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(ln).decode("utf-8")) if 0 < ln < 1_000_000 else {}
                except Exception:
                    return self._json(400, {"message": "Ungültige Anfrage."})
                draft = body.get("draft") or {}
                try:
                    if self.path.startswith("/api/assistent/probe"):
                        return self._json(200, assist_probe(draft))
                    if self.path.startswith("/api/assistent/save"):
                        ok, msg, q, rate = assist_evaluate(draft, body.get("token"), body.get("labels") or {})
                        if not ok:
                            return self._json(200, {"saved": False, "message": msg})
                        block = config_add_search(q, draft.get("motiv") or q["name"])
                        return self._json(200, {
                            "saved": True, "name": q["name"], "yaml": block,
                            "message": msg + f" Die Suche „{q['name']}“ ist gespeichert (Schwelle {q['threshold']:.2f}, "
                                             f"Grob-Schwelle {q['coarse_threshold']:.2f}). In etwa einer Minute erscheint "
                                             f"sie auf der Übersicht; die Treffer werden in den nächsten Stunden nach und "
                                             f"nach geprüft."})
                except (ValueError, RuntimeError) as e:
                    return self._json(400, {"message": str(e)})
                except Exception as e:
                    log.error("Assistent fehlgeschlagen", exc_info=True)
                    return self._json(500, {"message": f"Unerwarteter Fehler: {e}"})
                return self._json(404, {"message": "unbekannt"})
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
                    if LAST_INDEX[0]:
                        try:
                            write_index(*LAST_INDEX[0])      # Übersicht sofort ohne den gelöschten Report
                        except Exception as e:
                            log.warning("Übersicht konnte nicht sofort aktualisiert werden: %s", e)
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
        # Alten Index freigeben, BEVOR der neue gebaut wird. Sonst liegt er beim Neu-Scan kurz doppelt
        # im Arbeitsspeicher – das hat den Container regelmäßig über mem_limit getrieben (OOM).
        if THEMEN_STATE.get("holds_index"):
            log.info("Neu-Scan wartet, bis die Themen-Berechnung den Index freigibt …")
            while THEMEN_STATE.get("holds_index"):
                time.sleep(5)
        if ENGINE[0] is not None:
            ENGINE[0].pool.shutdown(wait=True)
        ENGINE[0] = None
        eng = store = None
        gc.collect()
        store = Store(found, db, base.dim, rerank_key)
        eng = Engine(base, db, store, settings, searches)
        ENGINE[0] = eng
        eng.strong = strong if (strong and strong.key == rerank_key) else None
        eng.progress(force=True)
        next_scan = time.time() + float(settings["rescan_interval_hours"]) * 3600
        if store.has_g.any():
            eng.report(final=False)          # Übersicht + Reports sofort nach dem Start aktualisieren
        themen_checked = False               # Themen prüfen, sobald die Arbeit ruht (oder Stufe 1 fertig ist)
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
            if not themen_checked and (kind is None or not eng.todo_g):
                themen_checked = True        # einmal pro Scan: fehlende oder veraltete Themen nachrechnen
                try:
                    started, msg = themen_maybe_start(eng)
                    if not started and "Stufe 1" in msg:
                        log.info("Themen: %s", msg)
                except Exception:
                    log.error("Themen: Prüfung fehlgeschlagen", exc_info=True)
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
