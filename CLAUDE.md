# CLAUDE.md – synology-bildsuche

Projektwissen für Claude Code. Persönliche Werte (NAS-Adresse, Benutzer, aktueller Betriebsstand) stehen in
`CLAUDE.local.md` (nicht im Repository). Entscheidungen: @docs/ENTSCHEIDUNGEN.md · Offene Punkte: @docs/OFFENE-PUNKTE.md

## Worum es geht

Docker-Container für eine Synology-NAS (Intel J4125 **ohne AVX**, Container Manager, **kein SSH**), der ein großes
Fotoarchiv (~200.000 Bilder, HEIC/JPG/RAW) mit deutschen Suchbegriffen durchsucht, Reports als Webseite (Port 8091)
ausliefert und Treffer in Synology-Photos-Alben übernimmt. Alles lokal, Archiv nur lesend eingebunden.
Benutzerdoku: `README.md`, `docs/INSTALLATION.md`, `docs/SPEZIFIKATION.md`, `docs/FEHLERSUCHE.md`.

## Arbeitsweise mit dem Nutzer

- Sprache: **Deutsch**, klar und freundlich; eine Rückfrage auf einmal. Der Nutzer ist erfahrener IT-Architekt,
  mag Begründungen und Denkanstöße, aber keine langen Romane.
- Der Nutzer arbeitet am Mac; die NAS bedient er über die DSM-Oberfläche (Container Manager, File Station,
  Aufgabenplaner für root-Skripte). Befehle für die NAS als Aufgabenplaner-Skript formulieren, nicht als SSH.
- Erst fragen, bevor Funktionen gebaut werden, die er nicht verlangt hat. Abgelehnte Ideen nicht wieder anbieten
  (siehe Entscheidungen).
- Keine Passwörter erfragen oder verarbeiten; Zugangsdaten trägt der Nutzer selbst in `data/photos.env` ein.

## Aufbau

Eine Datei: `app/bildsuche.py` (~2.100 Zeilen). Wichtige Teile in Reihenfolge:

| Bereich | Inhalt |
|---|---|
| Kopf | `VERSION`, Pfade (`/data`, `/config`, `/reports`), Logging (stdout + `data/logs/bildsuche.log`), Excepthooks |
| `DEFAULTS`, `parse_config`, `validate_config_text`, `save_config_text` | Konfiguration (`searches.yaml`), Prüfung, Sicherung |
| `Model`, `pos_prob`, `load_image`, `quadrants`, `padded` | open_clip-Modelle, Scoring (Softmax gegen Negatives), Bildladen (HEIC/RAW) |
| `index_file`, `DB`, `list_indexes`, `migrate_legacy_index` | SQLite-Index **pro Bildpfad**, relative Pfade, float16 |
| `scan`, `Store` | Dateiscan, RAM-Arrays G (Gesamtbild), T (Kacheln), R (Prüfung) |
| `Engine` | Arbeitsschleife: Stufe 1 → Prüfung → Kacheln, adaptive Kandidatenmenge (`pool_size`), Reports |
| `write_report`, `write_index`, `album_box`, `LIGHTBOX` | HTML/CSV der Reports, Übersichtsseite, Album-UI, Bildansicht |
| `PhotosClient`, `album_job` | inoffizielle Synology-Photos-API (Login `session=webui`), Album anlegen/ergänzen/teilen/löschen |
| `assist_*`, `config_add_search`, `ASSIST_PAGE` | Assistent „Neue Suche anlegen“: Probesuche, Kalibrierung, Eintrag in die Konfiguration |
| `read_log_lines`, `protokoll_page` | Protokoll-Seite |
| `start_web` | `ThreadingHTTPServer`: statische Reports + `/bild`, `/config`, `/protokoll`, `/api/*` |
| `main` | Start, Modell-Download/Offline-Modus, Hauptschleife, Scan alle 6 h |

Weitere Dateien: `compose.yaml` (**nur ASCII**), `Dockerfile`, `config/searches.example.yaml`, `data/photos.env.example`.

## Konventionen

- **Version** bei jeder Änderung hochzählen: `VERSION = "JJJJ-MM-TT<buchstabe>"`; Eintrag in `CHANGELOG.md`;
  betroffene Doku (`docs/SPEZIFIKATION.md`, `docs/FEHLERSUCHE.md`, `README.md`) mitpflegen.
- Alle Texte, Meldungen, UI und Protokoll auf **Deutsch**.
- HTML entsteht in Python-**f-Strings**: geschweifte Klammern in CSS/JS verdoppeln (`{{ }}`); `\n` in JS-Strings
  innerhalb von f-Strings als `\\n` schreiben (sonst echter Zeilenumbruch → JS-Syntaxfehler, ist zweimal passiert).
  Größere JS-Blöcke als **raw string** ohne f-String (siehe `LIGHTBOX`, `ASSIST_PAGE`).
- Funktionen, die aus `onclick="…"` aufgerufen werden, nie wie Methoden von `document` benennen
  (`evaluate`, `open`, `close`, `write` …) – im Inline-Handler gewinnt `document.evaluate` usw. (ist passiert).
- Keine neuen Abhängigkeiten ohne Not (der Build auf der NAS dauert, CPU ohne AVX).
- Der Index enthält nur suchunabhängige Vektoren; Suchänderungen dürfen nie eine Neuberechnung erzwingen.
- Nichts darf Bilddateien verändern oder löschen; das Archiv ist `:ro` eingebunden. Album-Funktionen ändern nur Alben.
- Keine personenbezogenen Werte (IP, Hostnamen, Benutzernamen) in Code, Doku oder Commits – Platzhalter verwenden
  (`192.168.1.10`, `diskstation.local`).

## Prüfen vor jedem Commit

```
python3 tools/check.py        # Syntax, Konfig-Vorlage, generiertes JavaScript (node --check, falls vorhanden)
```

Größere Änderungen an der Weboberfläche zusätzlich im Browser testen (z. B. Playwright gegen eine Testseite).
Echte Tests gegen die NAS macht der Nutzer; Ergebnisse schickt er als Protokoll (`/protokoll.txt`).

## Ausrollen auf die NAS

1. `app/bildsuche.py` in die Freigabe `docker` kopieren: `docker/bildsuche/app/bildsuche.py`
   (am Mac per SMB gemountet, typischerweise `/Volumes/docker/bildsuche/app/`).
2. Container Manager → Container `bildsuche` → **Neu starten** (macht der Nutzer).
3. Auf der Protokoll-Seite (`http://<NAS>:8091/protokoll`) muss die neue `VERSION` erscheinen.
- Änderungen an `compose.yaml`: Projekt stoppen und neu erstellen.
- Die Konfiguration der NAS (`Bildsuche/config/searches.yaml`) enthält die echten Werte und wird nie aus dem
  Repository überschrieben.
