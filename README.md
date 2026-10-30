# synology-bildsuche

**Ein Fotoarchiv auf der Synology mit deutschen Suchbegriffen durchsuchen, komplett lokal.**

Die Bildsuche läuft als Docker-Container auf einer Synology-NAS. Sie schaut sich jedes Foto einmal mit
KI-Bildmodellen an und speichert das Ergebnis als Vektor. Danach findet sie zu frei formulierten
Suchbegriffen wie *„ein fertig gebautes LEGO-Modell auf einem Tisch“* oder *„Kinder spielen mit LEGO auf dem Boden“*
die passenden Bilder. Eine neue Suche kostet keine Neuberechnung, nur Sekunden.

Die Treffer erscheinen als Webseite im Heimnetz. Von dort lassen sie sich per Knopfdruck in ein geteiltes
Album in **Synology Photos** übernehmen. Keine Cloud, keine Bilder verlassen die NAS, das Archiv ist nur lesend eingebunden.

## Funktionen

- **Drei Stufen für hohe Trefferqualität:** schnelles mehrsprachiges CLIP-Modell für alle Bilder,
  Kacheln für Motive am Bildrand, starkes Modell (SigLIP 2, 384 px) zur Prüfung der Kandidaten
- **Deutsche Suchbegriffe** mit Gegenkategorien, Schwellen pro Suche und automatisch wachsender Kandidatenmenge
- **Webseite** (Port 8091): Übersicht mit Fortschritt, Reports mit Vorschaubildern, große Bildansicht mit Blättern,
  CSV-Export, Konfigurationseditor mit Prüfung, Protokoll
- **Synology-Photos-Alben:** alle bestätigten Treffer oder eine Auswahl per Klick ins Album, Teilen mit Benutzern,
  Album löschen, Probelauf
- **Assistent für neue Suchen:** Schritt für Schritt mit Erklärungen, Probebildern aus dem eigenen Archiv und automatisch eingestellten Schwellen – ohne die Konfigurationsdatei anzufassen
- **Ein Index pro Bildpfad:** Wechsel des Ordners legt einen weiteren Index an, Zurückwechseln nutzt den alten weiter
- **Ressourcenschonend:** läuft mit niedriger Priorität rund um die Uhr, erkennt neue Bilder automatisch
- Liest JPG, PNG, **HEIC** und **RAW** (über die eingebettete Vorschau); bündelt Serienbilder

## Schnellstart

```
docker/bildsuche/          <- compose.yaml, Dockerfile, app/, data/
Bildsuche/                 <- eigene Freigabe: config/searches.yaml, reports/
```

1. Freigabe `Bildsuche` anlegen und die Ordner wie oben auf die NAS kopieren
   (`config/searches.example.yaml` → `Bildsuche/config/searches.yaml`).
2. In `searches.yaml` den Bildpfad (`photo_root`) und die Suchen eintragen.
3. Container Manager → Projekt → Erstellen → Pfad `/docker/bildsuche`, „Vorhandene compose.yaml verwenden“.
4. Webseite öffnen: `http://<NAS>:8091/`

Ausführlich, inklusive Album-Funktion und Berechtigungen: **[docs/INSTALLATION.md](docs/INSTALLATION.md)**

## Dokumentation

| Dokument | Inhalt |
|---|---|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Schritt für Schritt auf einer neuen Synology, Updates, Umzug |
| [docs/SPEZIFIKATION.md](docs/SPEZIFIKATION.md) | Architektur, Suchverfahren, Index, Konfigurationsreferenz, Webseite, Alben, Sicherheit, Betrieb |
| [docs/FEHLERSUCHE.md](docs/FEHLERSUCHE.md) | Bekannte Stolpersteine mit Lösung (Container Manager, Photos-Rechte, iPad) |
| [CHANGELOG.md](CHANGELOG.md) | Versionsgeschichte |
| [docs/ENTSCHEIDUNGEN.md](docs/ENTSCHEIDUNGEN.md) | getroffene Architektur- und Produktentscheidungen |
| [docs/OFFENE-PUNKTE.md](docs/OFFENE-PUNKTE.md) | offene Punkte und Ideen |
| [CLAUDE.md](CLAUDE.md) | Projektwissen für die Weiterentwicklung mit Claude Code |

## Voraussetzungen

- Synology mit **Container Manager** (getestet: DS425+, Intel J4125 ohne AVX, 18 GB RAM, DSM 7.3)
- mindestens 8 GB freier Arbeitsspeicher für den Container
- für Alben: **Synology Photos** mit aktiviertem Freigegebenen Bereich
- Internet nur beim ersten Start (Download der Modelle, ca. 2 GB)

## Rechenzeit (Richtwerte DS425+)

| Stufe | Tempo | 200.000 Bilder |
|---|---|---|
| 1 Gesamtbild | ca. 2–3 Bilder/s | ca. 1 Tag |
| 2 Kacheln | ca. 0,3 Bilder/s | ca. 1 Woche |
| 3 Prüfung | ca. 20 s pro Kandidat | wenige Tage, je nach Kandidaten |

Zwischenergebnisse sind sofort nutzbar; die Reports werden alle 20 Minuten aktualisiert.

## Datenschutz und Sicherheit

- Alles läuft lokal; nach dem ersten Modell-Download arbeitet der Container offline.
- Das Fotoarchiv ist nur lesend eingebunden.
- Die Webseite hat **kein Passwort**. Sie gehört nur ins Heimnetz, nicht ins Internet.
- Für die Album-Funktion wird ein eigener DSM-Benutzer mit Leserechten empfohlen; dessen Zugangsdaten
  stehen nur in `data/photos.env` (nicht im Repository).

## Entwicklung

Vor jedem Commit: `python3 tools/check.py` (Syntax, Konfig-Vorlage, generiertes JavaScript, keine persönlichen Werte).
Ausrollen: `app/bildsuche.py` auf die NAS kopieren und den Container neu starten (Details in [CLAUDE.md](CLAUDE.md)).

## Lizenz

[MIT](LICENSE). Verwendete Modelle: [OpenCLIP](https://github.com/mlfoundations/open_clip) `xlm-roberta-base-ViT-B-32`
(LAION-5B) und `ViT-B-16-SigLIP2-384` (WebLI). Es gelten deren eigene Lizenzen.
