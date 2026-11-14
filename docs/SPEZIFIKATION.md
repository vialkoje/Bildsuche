# Spezifikation

Stand: Version 2026-11-14b

## Zweck

Die Bildsuche durchsucht ein großes Fotoarchiv auf einer Synology-NAS nach frei formulierten Motiven, etwa
LEGO-Modellen, Spielszenen oder Klettersteigen. Jedes Bild wird **einmal** von KI-Modellen „angeschaut“ und als
Vektor gespeichert. Danach kostet ein neuer oder geänderter Suchbegriff nur Sekunden. Die Ergebnisse erscheinen als
Webseite im Heimnetz und lassen sich in Alben von Synology Photos übernehmen.

Entwickelt und betrieben auf einer DS425+ (Intel J4125, 4 Kerne ohne AVX, 18 GB RAM, DSM 7.3) mit rund
200.000 Bildern (HEIC, JPG, wenige RAW).

## Architektur

```mermaid
flowchart LR
  subgraph NAS["Synology-NAS"]
    A["Fotoarchiv<br/>/volume1/photo/…<br/>(nur lesend)"]
    subgraph C["Container bildsuche"]
      S["Scan und Index<br/>SQLite pro Bildpfad"]
      P["KI-Pipeline<br/>1 Gesamtbild · 2 Kacheln · 3 Prüfung"]
      W["Webserver :8091<br/>Reports · Themen · Bildansicht · Config · Protokoll · Alben"]
      S --> P --> W
    end
    F["Freigabe Bildsuche<br/>config/searches.yaml<br/>reports/"]
    D["docker/bildsuche/data<br/>Indizes · Themen · Vorschaubilder · Modelle · Protokoll · photos.env"]
    PH["Synology Photos<br/>Freigegebener Bereich · Alben"]
    A --> S
    P <--> D
    W <--> F
    W -->|"Album-API"| PH
    A -.->|"indiziert"| PH
  end
  B["Browser im Heimnetz<br/>Mac · iPad · iPhone"] <--> W
```

Alles läuft in **einem** Container. Er liest das Archiv nur, hält Index und Modelle in eigenen Ordnern und spricht
Synology Photos nur an, wenn ein Album angelegt oder gelöscht wird.

| Komponente | Technik |
|---|---|
| Laufzeit | Python 3.11, PyTorch (CPU), open_clip, Pillow, pillow-heif, rawpy, NumPy, PyYAML |
| Index | SQLite, Embeddings als float16, im Betrieb komplett im RAM (auch die Prüf-Vektoren als float16) |
| Webserver | Python `http.server` (ThreadingHTTPServer), statische Reports + kleine JSON-API |
| Album-Anbindung | inoffizielle Synology-Photos-Web-API (`SYNO.Foto.*`, `SYNO.FotoTeam.*`) |

## Suchverfahren

| Stufe | Modell | Berechnet | Bilder | Tempo (DS425+) |
|---|---|---|---|---|
| 1 Gesamtbild | `xlm-roberta-base-ViT-B-32` (LAION-5B), mehrsprachig | ein Vektor pro Bild | alle | ca. 1 Tag / 200.000 |
| 2 Kacheln | dasselbe Modell | 4 überlappende Viertel (je 60 %) | alle, wahrscheinlichste zuerst | ca. 0,3 Bilder/s |
| 3 Prüfung | `ViT-B-16-SigLIP2-384` (WebLI) | Ganzbild (aufgefüllt) + 4 Viertel, 384 px | nur Kandidaten | ca. 20 s pro Bild |

**Reihenfolge:** Stufe 1 → Prüfung der Kandidaten → Kacheln. Neue Kandidaten aus den Kacheln werden sofort geprüft;
die Prüfung hat Vorrang.

**Scoring:** Softmax über Suchbegriff (Mittel aller Formulierungen) gegen die Gegenkategorien, skaliert mit dem
`logit_scale` des Modells. Grob-Score = Maximum aus Gesamtbild und bester Kachel; Prüf-Score = Maximum über die fünf Ausschnitte.

| Status | Bedeutung |
|---|---|
| Bestätigt | geprüft, Prüf-Score ≥ `threshold` |
| Vorläufig | noch nicht geprüft, Grob-Score ≥ `coarse_threshold` |
| Verworfen | geprüft, Prüf-Score < `threshold` |

**Gegenkategorien:** Hat eine Suche eigene `negatives`, **ersetzen** diese die `default_negatives`.

**Adaptive Kandidatenmenge:** Geprüft werden mindestens die besten `candidates` Bilder. Solange von den letzten 500
geprüften mindestens `expand_if_rate` (2 %) bestätigt werden, wächst die Menge um `expand_step` (1000), höchstens bis
`candidates_max`. So werden auch Treffer gefunden, die das schnelle Modell unterschätzt.

**Ähnliche Bilder:** Serienbilder und Kopien (Kosinus-Ähnlichkeit ≥ `dedupe_similarity`) werden im Report zu einem
Motiv „+n ähnliche“ gebündelt; die CSV enthält alle mit Verweis in `aehnlich_wie_rang`.

**Änderung einer Suche:** Der Index enthält nur suchunabhängige Bild-Vektoren. Nach dem Speichern werden alle Scores in
Sekunden neu berechnet und der Report vollständig neu aufgebaut; bereits geprüfte Bilder behalten ihren Prüf-Vektor
(suchübergreifend). Nur neu hinzugekommene Kandidaten müssen geprüft werden.

## Index

- **Ein Index pro Bildpfad:** `data/indexes/<pfad-slug>_<sha1-8>.sqlite`. Ein neuer `photo_root` legt einen weiteren
  Index an; beim Zurückwechseln wird der vorhandene weiterbenutzt.
- **Inhalt:** relative Pfade, Größe und Änderungszeit, Embeddings (Gesamtbild, Kacheln, Prüfung), Lesefehler.
  Suchbegriffe stehen nicht im Index.
- **Scan:** beim Start, alle `rescan_interval_hours` (6) und bei Pfadwechsel. Neue, geänderte und verschwundene Bilder
  werden erkannt und mit Namen protokolliert.
- **Ausgelassen:** `exclude_dirs` (u. a. `@eaDir`, `#recycle`, `#snapshot`), Dateien unter `min_file_kb`,
  RAW-Dateien mit gleichnamigem JPG/HEIC daneben.
- **Schutz:** Fehlen bei einem Scan mehr als max(1000, 30 %) der bekannten Bilder, wird der Index nicht bereinigt
  (falsche Konfiguration, nicht eingebundenes Laufwerk).
- **Reihenfolge:** zufällig, damit Zwischen-Reports Stichproben aus allen Jahren sind.
- **Offline:** Nach dem ersten erfolgreichen Modell-Download startet der Container ohne Internetzugriff.
- **Speicher beim Neu-Scan:** Der alte Index wird freigegeben, **bevor** der neue geladen wird (sonst lag er kurz
  doppelt im RAM und der Container lief regelmäßig in `mem_limit`). Hält gerade die Themen-Berechnung den Index,
  wartet der Neu-Scan darauf. Die Prüf-Scores werden stückweise (20.000 Bilder) berechnet.

## Konfiguration (`searches.yaml`)

Änderungen wirken nach ca. 30 Sekunden, ohne Neustart. Fehlt ein Wert, gilt der eingebaute Standard.

### Einstellungen (`settings`)

| Parameter | Standard | Bedeutung |
|---|---|---|
| `photo_root` | `/volume1/photo/Bildarchiv` | zu durchsuchender Ordner (NAS-Pfad) |
| `rescan_interval_hours` | 6 | Abstand der Scans |
| `report_minutes` | 20 | Zwischen-Report während der Berechnung |
| `batch_size` | 16 | Bilder pro Rechenschritt |
| `exclude_dirs` | `@eaDir`, `#recycle`, … | übersprungene Ordner |
| `min_file_kb` | 20 | kleinere Dateien ignorieren |
| `skip_raw_with_sibling` | true | RAW überspringen, wenn JPG/HEIC daneben liegt |
| `random_order` | true | zufällige Reihenfolge |
| `dedupe_similarity` | 0,95 | Bündelung ähnlicher Bilder, 0 = aus |
| `tiles` / `rerank` | true / true | Stufe 2 / 3 an oder aus |
| `rerank_model` / `rerank_pretrained` | `ViT-B-16-SigLIP2-384` / `webli` | Modell der Prüfung |
| `preview_max` | 300 | vorläufige Treffer als Bild im Report |
| `expand_if_rate` / `expand_step` / `candidates_max` | 0,02 / 1000 / 30000 | adaptive Kandidatenmenge |
| `web_base` | `http://diskstation.local:8091` | Adresse der Webseite (Links in der CSV) |
| `photos_url` | `http://192.168.1.10:5000` | DSM-Adresse für die Album-API (IP) |
| `photos_folder_prefix` | `auto` | Ordner in Photos; `auto` = `photo_root` ohne `photos_share_path` |
| `photos_share_path` | `/volume1/photo` | Freigabe hinter dem Freigegebenen Bereich |
| `photos_share_with` | `[]` | Benutzer, mit denen Alben geteilt werden |
| `photos_share_role` | `manage` (Beispielkonfiguration: `download`) | Rolle beim Teilen |
| `photos_album_prefix` | `"Bildsuche: "` | Vorbelegung des Albumnamens |
| `default_negatives` | allgemeine Motive | Gegenkategorien ohne eigene Liste |

### Suchen (`searches`)

| Feld | Bedeutung |
|---|---|
| `name` | Name, zugleich Ordner des Reports |
| `terms` | mehrere Formulierungen, werden gemittelt |
| `negatives` | optional, ersetzt `default_negatives` |
| `threshold` | Schwelle für „Bestätigt“ (Prüf-Score) |
| `coarse_threshold` | Schwelle für „Vorläufig“ (Grob-Score) |
| `candidates` | Mindestzahl geprüfter Kandidaten |
| `max_results` | bestätigte Motive als Bild im Report (CSV: alle) |
| `candidates_max` | optional, Obergrenze nur für diese Suche |

## Webseite

Port 8091, ohne Passwort. Abschalten: `WEB_PORT=0` in `compose.yaml`.

| Seite / Endpunkt | Inhalt |
|---|---|
| `/` | Übersicht: Kacheln für Assistent und Themen, Suchen als Karten (bestätigt/vorläufig), Fortschritt der drei Stufen, Indizes und ältere Reports (aufklappbar, mit Löschknopf); auf dem Handy einspaltig, hell/dunkel nach Systemeinstellung |
| `/<suche>/report.html` | Treffer als Vorschaubilder, Album-Kasten, Auswahl per Klick / Umschalt-Klick |
| `/<suche>/results.csv` | alle Treffer (Semikolon, deutsches Komma) |
| `/bild?s=&p=` | Originalbild, HEIC/RAW als JPEG; nur Bilder aus dem Report dieser Suche |
| `/config`, `/api/config` | Editor für `searches.yaml` mit Prüfung und Sicherung (letzte 10) |
| `/protokoll`, `/protokoll.txt` | Programmprotokoll, neueste oben, Filter, Download |
| `/api/album`, `/api/album/status` | Album-Aufträge |
| `/api/report/delete` | Löschen eines älteren Reports (nicht für aktive Suchen) |
| `/assistent`, `/api/assistent/*` | Assistent für neue Suchen (Info, Probesuche, Probebilder, Auswerten und Speichern); `/assistent?thema=<n>` übernimmt den Suchvorschlag einer Themen-Gruppe |
| `/themen` | Bilder nach Themen erkunden (siehe unten) |
| `/api/themen?n=&p=&alle=` | Gruppe `n` mit Untergruppen bzw. Seite `p` ihrer Bilder (120 pro Seite) |
| `/api/themen/status`, `/api/themen/neu` | Stand der Berechnung; Neuberechnung anstoßen (POST) |
| `/api/themen/bild?k=&g=` | Vorschau (240 px) bzw. große Ansicht (`g=1`); nur über die Position im Themenbaum, nie über einen Pfad |

**Bildansicht:** „Bild öffnen“ zeigt das Bild groß über dem Report (auch als iPad-Web-App), Blättern per Wischen,
Pfeilen oder Tastatur, „Auswählen“ fürs Album, Schließen per ✕, Esc oder Zurück-Geste.

**Reports** entstehen beim Start, alle `report_minutes` und nach jeder Konfigurationsänderung; sie werden atomar
geschrieben. Reports von Suchen, die nicht mehr existieren, erscheinen auf der Übersicht unter „Ältere Reports“.

**Protokoll:** zusätzlich zur Containerausgabe in `data/logs/bildsuche.log` (rotierend, 6 × 2 MB), inklusive Abstürzen
und Webserver-Fehlern. Nur im Container Manager: Build-Ausgaben, Download-Fortschritt, harte Abbrüche von außen.

## Assistent „Neue Suche anlegen“

Ein Knopf auf der Übersicht öffnet einen Fragebogen, mit dem auch Laien eine gute Suche anlegen – ohne Sprachmodell,
komplett lokal. Jede Frage erklärt, warum sie gestellt wird und wie eine gute Antwort aussieht.

| Schritt | Frage | Wirkung |
|---|---|---|
| 1 | Was möchtest du finden? (ein Satz wie eine Bildunterschrift) | Grundlage der Formulierungen |
| 2 | Situation oder Gegenstand im Mittelpunkt? | Szene: eigene Gegenbeschreibungen statt der Standardliste; Objekt: zusätzliche Nahaufnahme-Formulierung |
| 3 | Wer ist zu sehen, wo? | Formulierungsvarianten, Vorschläge für Ausschlüsse |
| 4 | Formulierungen prüfen (2–4 Zeilen) | `terms`; Verneinungen („kein“, „ohne“) werden abgefangen |
| 5 | Was soll nicht gefunden werden? (Vorschläge zum Antippen) | `negatives` (+ „ein Foto“) |
| 6 | Wie streng? (vollständig / ausgewogen / treffsicher) | Start-Schwelle 0,30 / 0,40 / 0,50 |
| 7 | Name | eindeutiger Suchname (Kleinbuchstaben, Ziffern, `_`) |
| 8 | Probebilder bewerten | Kalibrierung und Freigabe |

**Probesuche:** bis zu 16 Bilder aus dem Archiv auf gestaffelten Rängen (1 … ca. 400), Serienbilder übersprungen.
Grob-Score aus dem vorhandenen Index (Sekunden); für bereits geprüfte Bilder zusätzlich der Prüf-Score des starken Modells.
Bei „passt nicht“ kann die Nutzerin notieren, was stattdessen zu sehen ist – das wird beim nächsten Versuch Gegenbeschreibung.

**Freigabe:** Gespeichert wird nur, wenn von den 8 wahrscheinlichsten bewerteten Bildern mindestens 60 % passen.
**Kalibrierung:** Grob- und Prüf-Schwelle werden so gewählt, dass sie Ja- und Nein-Bilder am besten trennen
(F-Maß; Gewichtung nach Strenge). Ohne genügend geprüfte Bilder gilt der Startwert der Strenge.
**Speichern:** Der Abschnitt wird mit Kommentar am Ende von `searches:` eingefügt (Kommentare bleiben erhalten),
vorher geprüft und gesichert; wirksam nach ca. 30 Sekunden. Die Prüfung der neuen Kandidaten läuft danach im Hintergrund.

## Themen „Bilder nach Themen erkunden“

Ohne Suchbegriff durch das ganze Archiv stöbern: Die Bilder werden anhand ihrer Vektoren automatisch in Gruppen
und Untergruppen eingeteilt und mit deutschen Begriffen benannt. Erreichbar über die Übersicht (Knopf „▦ Bilder nach
Themen erkunden“) oder `/themen`.

**Gruppen:** hierarchisches k-Means auf den Gesamtbild-Vektoren (Stufe 1), ergänzt um die Kacheln (Stufe 2), soweit
schon gerechnet. Oben `THEMEN_TOP_K` = 12 grobe Themen, darunter je bis zu `THEMEN_SUB_K` = 10 Untergruppen, bis eine
Gruppe höchstens `THEMEN_LEAF` = 80 Bilder hat. k-Means rechnet auf einer Stichprobe von 20.000 Bildern, die
Zuordnung danach gilt für alle. Die Bilder sind so sortiert, dass jede Gruppe ein zusammenhängender Abschnitt ist.

**Namen:** aus einer festen Begriffsliste in drei Ebenen (`THEMEN_TAXONOMIE`, „Oberbegriff | Bereich: Begriff; …“,
z. B. „Berge | Klettern: Klettern; Klettersteig; Bergsteigen“). Gewählt wird der Begriff, Bereich oder Oberbegriff,
der die Untergruppen am besten **abdeckt** und sich am besten von den Geschwistern **abgrenzt**; Geschwister teilen
keine Begriffe, ein Name wiederholt keinen Namen weiter oben im Pfad, bei Bedarf zwei Begriffe („Berge & Seen“).
Bei geprüften Bildern fließt das starke Modell (Stufe 3) ein, dazu die bestätigten Treffer der Suchen
(ab 30 Treffern je Suche).

**Querverweise:** Bilder, die fast genauso gut in eine Nachbargruppe passen, erscheinen dort unter „Passt auch
hierher“; in der Bildansicht führt „↔ passt auch woanders“ zur anderen Gruppe.

**Suchvorschlag je Gruppe:** Positiv sind Begriffe, die in der Gruppe häufig und im Archiv selten sind; negativ
Begriffe der ähnlichsten Nachbargruppen. „Als Suche verwenden“ zeigt den fertigen Block für `searches.yaml` zum
Kopieren; „Im Assistenten öffnen“ übernimmt Formulierungen, Ausschlüsse und Namen in den Assistenten (Schritt 4)
– so werden Schwellen mit Probebildern kalibriert.

**Berechnung:** im Hintergrund in sechs Schritten (Begriffe vorbereiten, Bildmerkmale zusammenführen, Gruppen bilden,
Gruppen benennen, Querverweise suchen, Vorschaubilder vorbereiten), Fortschritt auf der Seite. Automatisch einmal pro
Scan, sobald Stufe 1 zu mindestens 98 % fertig ist und noch kein Baum existiert oder sich mindestens 2 %
der Bilder geändert haben; sonst per „neu berechnen“. Die bisherige Einteilung bleibt währenddessen nutzbar.
Der Thread läuft bewusst **ohne** gesenkte Priorität (sonst bekommt er neben der Bildsuche kaum Rechenzeit).
Ergebnis: `data/themen/<index>.json.gz` (ein Baum pro Bildpfad, wie die Indizes); Vorschaubilder in `data/thumbs/`,
die oberen zwei Ebenen werden vorab erzeugt, der Browser darf sie eine Woche zwischenspeichern.

## Synology-Photos-Alben

Bedienung im Album-Kasten jeder Report-Seite:

- **Alle bestätigten hinzufügen** oder **Auswahl hinzufügen**; „ähnliche Bilder mitnehmen“ ist optional
- **Probelauf (5 Bilder)** legt „Bildsuche-Test: …“ an
- **Album löschen** (mit Rückfrage) – nur Alben, die die Bildsuche selbst angelegt hat; Fotos bleiben erhalten

Ablauf:

1. Pfade aus `results.csv` (bei Auswahl nur Pfade, die im Report stehen)
2. Anmeldung über `SYNO.API.Auth` mit `session=webui` (andere Session-Namen liefern Fehler 402)
3. Ordner im Freigegebenen Bereich Ebene für Ebene suchen (`SYNO.FotoTeam.Browse.Folder`), Bilder auflisten (`…Browse.Item`)
4. Zuordnung: exakter Dateiname → ohne Groß-/Kleinschreibung → Namensstamm (RAW+JPG, Live-Fotos)
5. Existiert das Album, werden Bilder **hinzugefügt**, nie entfernt; sonst wird es angelegt
6. Teilen mit `photos_share_with` und `photos_share_role`; lehnt Photos die Rolle ab, schrittweise schwächer
   (manage → upload → download → view)
7. Meldung mit Anzahl und bis zu drei nicht gefundenen Dateien (vollständige Liste im Protokoll)

**Rollen:** Ein Album gehört immer dem Benutzer, der es anlegt, und lässt sich nicht übergeben. Löschen kann nur der
Besitzer.

| Rolle | Geteilter Benutzer kann |
|---|---|
| `view` | ansehen, Diashow – nicht auswählen |
| `download` | auswählen, herunterladen, in eigenes Album übernehmen (Kopiervorlage) |
| `manage` | Bilder im Album hinzufügen und entfernen, aber das Album nicht löschen |

**Vorsicht:** Das Menü „Löschen“ für die Foto-*Datei* hängt an den Ordnerrechten des Benutzers in Photos, nicht an der
Album-Rolle, und erscheint daher auch in geteilten Alben. Schutz: Papierkorb der Freigabe `photo` aktivieren.

## Berechtigungen und Sicherheit

- Eigener DSM-Benutzer für die Alben (nur `users`, Freigabe `photo` schreibgeschützt, nur Synology Photos);
  Zugangsdaten nur in `data/photos.env`.
- **Ordnerrechte in Photos setzen:** Synology Photos führt eine eigene Rechte-Datenbank und schreibt die ACLs selbst.
  Der Hilfsbenutzer braucht „Anzeigen“ für den Bildordner **und jeden Unterordner**. Änderungen per `synoacltool`
  erreichen nur das Dateisystem, nicht Photos; `synoacltool -enforce-inherit` kann bestehende Listen ersetzen.
- Archiv und Programmcode sind im Container nur lesend eingebunden.
- `/bild` liefert nur Bilder aus Reports aus, kein Durchstöbern von Ordnern.
- Die Webseite hat keinen Passwortschutz und gehört nur ins Heimnetz.

## Betrieb

| Thema | Umsetzung |
|---|---|
| Priorität | `cpu_shares: 256` (der Synology-Kernel unterstützt keine harte CPU-Grenze per `cpus:`) |
| Threads | `THREADS=4`, weniger Last mit `THREADS=2` |
| Speicher | `mem_limit: 7g` (beide Modelle + Index im RAM; Themen-Berechnung kurzzeitig zusätzlich) |
| Neustart | `restart: unless-stopped`; Code wird aus `app/` eingebunden, Update = Datei ersetzen + Neustart |
| Datensicherung | Papierkorb für `photo`, Hyper Backup; verschwundene Bilder stehen mit Namen im Protokoll |

## Bekannte Grenzen

- Die Prüfung mit SigLIP 2 ist auf einer CPU ohne AVX teuer (ca. 20 s pro Bild); große Kandidatenmengen halten die
  Kacheln tagelang auf.
- Erkennung ist statistisch: Schwellen und Gegenkategorien sind je Suche einzustellen; ähnliche Motive
  (Berg mit Mensch statt Klettersteig) werden mitunter bestätigt.
- Die Album-Funktion nutzt eine inoffizielle API und kann nach Updates von Synology Photos brechen –
  nach Updates zuerst den Probelauf testen.
- Bilder außerhalb des Freigegebenen Bereichs lassen sich durchsuchen, aber nicht in Alben übernehmen.
- Themen-Namen kommen aus einer festen Begriffsliste; was dort fehlt, kann nicht als Name erscheinen.
  Die Einteilung ist statistisch und ändert sich bei jeder Neuberechnung leicht.
