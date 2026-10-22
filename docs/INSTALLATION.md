# Installation auf einer Synology

Getestet auf DS425+ mit DSM 7.3 und Container Manager.
Dauer: ca. 30 Minuten Handarbeit, danach rechnet die NAS selbstständig
(Gesamtbild ca. 1 Tag pro 200.000 Bilder, Kacheln und Prüfung 1–3 Wochen; Zwischenergebnisse sind sofort nutzbar).

## Zielstruktur auf der NAS

```
/volume1/docker/bildsuche/        (Freigabe "docker")
├── compose.yaml                  aus dem Repository
├── Dockerfile                    aus dem Repository
├── app/bildsuche.py              aus dem Repository
└── data/                         Index, Modelle, Protokoll, photos.env (entsteht/wird befüllt)
/volume1/Bildsuche/               (neue Freigabe "Bildsuche", per SMB sichtbar)
├── config/searches.yaml          Kopie von config/searches.example.yaml
└── reports/                      hier entstehen die Reports
```

## 1. Voraussetzungen

- **Container Manager** aus dem Paket-Zentrum
- Bilder unter `/volume1/photo/...` (für Alben: **Synology Photos** mit aktiviertem Freigegebenen Bereich)
- Internetzugang beim ersten Start (Download der KI-Modelle, ca. 2 GB)
- mindestens 8 GB freier RAM für den Container (`mem_limit: 7g` in `compose.yaml`)

## 2. Dateien auf die NAS bringen

1. Systemsteuerung → Freigegebener Ordner → **Erstellen**: `Bildsuche` (auf volume1).
2. Repository herunterladen (GitHub → *Code* → *Download ZIP*) und per Finder/Explorer kopieren:
   - `compose.yaml`, `Dockerfile`, `app/` → `docker/bildsuche/`
   - leeren Ordner `docker/bildsuche/data/` anlegen
   - `config/searches.example.yaml` → `Bildsuche/config/searches.yaml`
   - leeren Ordner `Bildsuche/reports/` anlegen
3. Alle Ordner müssen **vorher existieren**: Docker legt auf der Synology keine fehlenden Ordner an.

> `compose.yaml` nur mit einem normalen Texteditor am Rechner bearbeiten, nicht mit dem Synology-Texteditor
> (der hat schon ungültige Zeichen hineingeschrieben). Andere Volumes oder Ordner: linke Seite der `volumes`-Zeilen anpassen.

## 3. Bildpfad und Suchen einstellen

In `Bildsuche/config/searches.yaml`:

- `photo_root`: der zu durchsuchende Ordner, z. B. `/volume1/photo/Bildarchiv`
  (er muss unterhalb einer in `compose.yaml` eingebundenen Freigabe liegen, standardmäßig `/volume1/photo`)
- `web_base`: Adresse der Webseite, z. B. `http://diskstation.local:8091`
- die Suchen unter `searches:` (lässt sich später bequem im Browser ändern)

## 4. Projekt anlegen

1. Container Manager → **Projekt** → **Erstellen**
2. Projektname `bildsuche`, Pfad `/docker/bildsuche`
3. Quelle **„Vorhandene compose.yaml verwenden“** → Weiter → Web-Portal überspringen → **Fertig**
4. Der Build lädt PyTorch (CPU) und dauert 5–15 Minuten.

Im Protokoll (Container → bildsuche → Protokoll) erscheinen `Bildsuche Version …`, `Webseite läuft auf Port 8091`,
der Modell-Download und dann `Fortschritt | …`.

Webseite: **`http://<NAS>:8091/`** – auf dem iPad/iPhone auch als Web-App auf dem Home-Bildschirm nutzbar.

## 5. Album-Funktion (optional)

### 5a. Hilfsbenutzer anlegen

Systemsteuerung → Benutzer und Gruppe → Erstellen, z. B. `bildsuche`:
- Gruppe: nur `users`
- Freigabe `photo`: **Schreibgeschützt**, alle anderen: kein Zugriff
- Anwendungen: nur **Synology Photos**

So liegt nie das Passwort eines Hauptkontos im Container.

### 5b. Zugriff in Synology Photos freigeben

Als Administrator in Photos → Freigegebener Bereich → den Bildordner (z. B. `Bildarchiv`) → Berechtigungen:
den Hilfsbenutzer mit **Anzeigen** eintragen – **auch für alle Unterordner**. Die Freigabe des obersten Ordners allein
gilt in Photos nicht für die Unterordner.

Berechtigungen immer **in Photos** setzen, nicht per Skript: Photos führt eine eigene Rechte-Datenbank
(Details: [FEHLERSUCHE.md](FEHLERSUCHE.md)).

### 5c. Zugangsdaten hinterlegen

`data/photos.env.example` als `docker/bildsuche/data/photos.env` ablegen und ausfüllen:

```
PHOTOS_USER=bildsuche
PHOTOS_PASS=…
```

### 5d. Einstellungen in `searches.yaml`

| Einstellung | Bedeutung |
|---|---|
| `photos_url` | Adresse von DSM, **als IP** (z. B. `http://192.168.1.10:5000`) – `.local`-Namen löst der Container meist nicht auf |
| `photos_share_with` | DSM-Benutzer, mit denen neue Alben geteilt werden, z. B. `["anna", "ben"]` |
| `photos_share_role` | `download` = Kopiervorlage (empfohlen) · `manage` = Album bearbeitbar · `view` = nur ansehen |
| `photos_album_prefix` | Vorbelegung des Albumnamens, Standard `"Bildsuche: "` |

### 5e. Testen

Container neu starten → Report-Seite → **Probelauf (5 Bilder)** → Meldung
„Album … mit 5 Bildern angelegt, geteilt mit … (Rolle: …)“.

## 6. Empfohlen

- **Papierkorb** für die Freigabe `photo` aktivieren (Systemsteuerung → Freigegebener Ordner → photo → Bearbeiten).
  In Photos versehentlich gelöschte Fotos landen dann in `#recycle`.
- Regelmäßiges **Hyper Backup** der Bilder.
- Die Webseite nicht ins Internet freigeben (kein Passwortschutz).

## 7. Updates und Pflege

| Was | Wie |
|---|---|
| Programm-Update | `app/bildsuche.py` ersetzen → Container neu starten → Version im Protokoll prüfen |
| `compose.yaml` geändert | Projekt stoppen → erneut erstellen |
| Suchen ändern | Übersichtsseite → „Konfiguration ansehen & bearbeiten“ (wirkt nach ca. 30 s, mit Prüfung und Sicherung) |
| Protokoll ansehen | Übersichtsseite → „Protokoll“ |
| Index neu aufbauen | Container stoppen, Datei in `data/indexes/` löschen, starten |
| Last reduzieren | in `compose.yaml` `THREADS=2` |

## Umzug auf eine andere NAS

Um nicht neu rechnen zu müssen, mitnehmen:
- `data/indexes/` – gilt nur, wenn die Bilder wieder unter genau demselben Pfad liegen
- `data/models/` – spart den Modell-Download
- `Bildsuche/config/searches.yaml` – eigene Suchen
