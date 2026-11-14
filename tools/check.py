#!/usr/bin/env python3
"""Schnelle Prüfung vor jedem Commit – läuft ohne NAS, ohne Modelle, ohne PyTorch.

Prüft: Python-Syntax, Laden der Konfig-Vorlage, und das JavaScript, das die Seiten erzeugen
(per `node --check`, falls Node.js installiert ist). Aufruf: python3 tools/check.py
"""
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app", "bildsuche.py")
ok = True


def step(name, fn):
    global ok
    try:
        fn()
        print(f"  ok   {name}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  FEHLER {name}: {e}")


tmp = tempfile.mkdtemp(prefix="bildsuche-check-")
os.environ.update(DATA=os.path.join(tmp, "data"), REPORTS=os.path.join(tmp, "reports"),
                  CONFIG=os.path.join(ROOT, "config", "searches.example.yaml"), WEB_PORT="0")
os.makedirs(os.environ["REPORTS"], exist_ok=True)

print("Bildsuche – Prüfung")
step("Python-Syntax", lambda: py_compile.compile(APP, doraise=True))

sys.path.insert(0, os.path.dirname(APP))
import bildsuche as b  # noqa: E402

print(f"  Version {b.VERSION}")


def cfg():
    settings, searches = b.load_config()
    assert searches, "keine Suchen in der Vorlage"
    for k in ("photo_root", "photos_url", "photos_share_role"):
        assert k in settings, f"{k} fehlt"


step("Konfig-Vorlage laden", cfg)


def scripts():
    pages = {"album_box": b.album_box("test"), "LIGHTBOX": b.LIGHTBOX, "CONFIG_PAGE": b.CONFIG_PAGE,
             "protokoll": b.protokoll_page("/protokoll"), "ASSIST_PAGE": b.ASSIST_PAGE,
             "THEMEN_HTML": b.THEMEN_HTML}
    b.CURRENT_ROOT[0] = "/volume1/photo"
    b.write_index((10, 10, 1, 5, 2), [("test", 2, 0)], False)
    with open(os.path.join(os.environ["REPORTS"], "index.html"), encoding="utf-8") as f:
        pages["index"] = f.read()
    node = shutil.which("node")
    for name, html in pages.items():
        for i, js in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
            if not node:
                continue
            p = os.path.join(tmp, f"{name}-{i}.js")
            with open(p, "w", encoding="utf-8") as f:
                f.write(js)
            r = subprocess.run([node, "--check", p], capture_output=True, text=True)
            if r.returncode:
                lines = r.stderr.strip().splitlines()
                msg = next((x for x in lines if "Error" in x), lines[0] if lines else "?")
                raise RuntimeError(f"{name}: {msg}")
    if not node:
        print("  (Node.js nicht gefunden – JavaScript-Prüfung übersprungen)")


step("Generiertes JavaScript", scripts)


def private():
    bad = re.compile(r"192\.168\.(?!1\.10\b)\d+\.\d+|the_gridle|vialko", re.I)
    for d, _, files in os.walk(ROOT):
        if ".git" in d or "tools" in d:
            continue
        for fn in files:
            if fn in ("CLAUDE.local.md", "LICENSE") or not fn.endswith((".py", ".md", ".yaml", ".example", "Dockerfile")):
                continue
            with open(os.path.join(d, fn), encoding="utf-8", errors="ignore") as f:
                m = bad.search(f.read())
            if m:
                raise RuntimeError(f"persönlicher Wert '{m.group(0)}' in {os.path.relpath(os.path.join(d, fn), ROOT)}")


step("Keine persönlichen Werte im Repository", private)
shutil.rmtree(tmp, ignore_errors=True)
print("Alles in Ordnung." if ok else "Es gibt Fehler.")
sys.exit(0 if ok else 1)
