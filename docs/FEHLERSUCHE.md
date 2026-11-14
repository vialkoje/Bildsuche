# Fehlersuche

Stolpersteine, die beim Aufbau tatsächlich aufgetreten sind, mit Ursache und Lösung.

## Container Manager

| Symptom | Ursache | Lösung |
|---|---|---|
| `NanoCPUs can not be set, as your kernel does not support CPU CFS scheduler` | Der Synology-Kernel unterstützt `cpus:` nicht | `cpu_shares` statt `cpus` verwenden (so in `compose.yaml`) |
| `invalid trailing UTF-8 octet` beim Erstellen | `compose.yaml` mit dem Synology-Texteditor bearbeitet | Datei nur am Rechner bearbeiten, nur ASCII-Zeichen verwenden |
| „Container nicht vorhanden“ beim Stoppen, Projekt hängt | Projekt wurde neu erstellt, während der alte Container lief | Container Manager neu starten, Projekt stoppen, erneut erstellen |
| Änderung an `compose.yaml` wirkt nicht | Neustart reicht dafür nicht | Projekt stoppen → erstellen |
| Container startet alle paar Stunden neu, im Protokoll „Zu wenig Arbeitsspeicher“ | beim Neu-Scan lag der alte Index kurz doppelt im RAM (OOM bei `mem_limit`) | behoben in 2026-11-13c (alter Index wird vorher freigegeben); sonst `mem_limit` erhöhen |
| Neue Programmversion läuft nicht | alte `bildsuche.py` noch in `app/` oder Datei heißt z. B. `bildsuche (1).py` | Datei exakt als `app/bildsuche.py` ersetzen, Version im Protokoll prüfen |

## Webseite

| Symptom | Ursache | Lösung |
|---|---|---|
| 404 direkt nach dem Start | noch kein Report geschrieben | wenige Minuten warten (die Übersicht entsteht beim Start) |
| Album-Knöpfe funktionieren nicht | Report als Datei (Finder/SMB) statt über die Webseite geöffnet | über `http://<NAS>:8091/` öffnen |
| iPad: nach „Bild öffnen“ kein Zurück | Seite als Web-App auf dem Home-Bildschirm, alte Version öffnete einen neuen Tab | ab 2026-10-21e erscheint das Bild über dem Report; sonst Web-App schließen und neu öffnen |
| Themen: „Die Themen sind noch nicht berechnet“ | Stufe 1 ist noch nicht zu 98 % fertig | abwarten – die Berechnung startet danach von selbst; oder „Jetzt berechnen“ |
| Themen: Berechnung hängt bei „Begriffe vorbereiten“ | ältere Version rechnete mit gesenkter Priorität und bekam kaum CPU | ab 2026-11-12a behoben |
| Themen: Vorschaubilder erscheinen langsam | bei der ersten Anzeige tieferer Ebenen werden sie erst erzeugt (höchstens 2 gleichzeitig) | kein Fehler; danach aus dem Zwischenspeicher |
| „Vorläufig“ ist überall 0 | alle Bilder über der Grob-Schwelle sind bereits geprüft | kein Fehler |

## Synology Photos / Alben

| Symptom | Ursache | Lösung |
|---|---|---|
| `SYNO.API.Auth.login: Fehlercode 402` | Session-Name wird von DSM abgelehnt | ab 2026-10-17 meldet sich das Programm mit `session=webui` an |
| „Ordner in Photos nicht sichtbar“, Hilfsbenutzer sieht in Photos nur den obersten Ordner leer | Photos-Berechtigung fehlt für die Unterordner | in Photos als Administrator **jeden Unterordner** für den Hilfsbenutzer mit „Anzeigen“ freigeben |
| File Station zeigt die Bilder, Photos nicht | Rechte per Skript (`chmod`, `synoacltool`) geändert – Photos führt eine eigene Rechte-Datenbank | Berechtigung in Photos setzen; Photos-Paket ggf. neu starten |
| Eigene Ordnerliste von `Bildarchiv` plötzlich weg | `synoacltool -enforce-inherit` auf dem Ordner ausgeführt | Liste mit `synoacltool -copy <unterordner> <ordner>` zurückkopieren; den Befehl nicht verwenden |
| `chmod`/`chgrp` ohne Wirkung | Freigaben unter DSM nutzen ACLs (erkennbar am `+` in `ls -l`) | Rechte über DSM bzw. Photos setzen |
| Geteiltes Album: nur Diashow, kein Auswählen | Rolle `view` | `photos_share_role: download` oder `manage`, danach einmal „Alle bestätigten hinzufügen“ |
| Album lässt sich als geteilter Benutzer nicht löschen | nur der Besitzer darf Alben löschen | Knopf „Album löschen“ auf der Report-Seite |
| „Album löschen“ findet das Album nicht | Album gehört einem anderen Benutzer (z. B. mit anderen Zugangsdaten angelegt) | als dessen Besitzer in Photos löschen |
| Fotos verschwunden nach Arbeiten im Album | „Löschen“ (Papierkorb-Symbol) entfernt die Datei, nicht nur den Album-Eintrag | Papierkorb der Freigabe aktivieren; Wiederherstellung aus `#recycle` oder Hyper Backup – die Namen stehen im Protokoll („Nicht mehr vorhanden: …“) |

## Rechenzeit

| Symptom | Ursache | Lösung |
|---|---|---|
| Kacheln stehen still | die Prüfung hat Vorrang, solange Kandidaten warten | abwarten oder `expand_if_rate` erhöhen (z. B. 0,05) |
| Prüfung wird nie fertig | die Kandidatenmenge wächst, solange am Listenende ≥ 2 % Treffer auftauchen | `expand_if_rate` erhöhen oder `candidates_max` pro Suche setzen |
| Sehr viele Treffer bei einer Suche | Schwelle zu niedrig oder ähnliche Motive werden bestätigt | Ränge am Listenende sichten; `threshold` anheben oder typische Fehltreffer als `negatives` aufnehmen |
