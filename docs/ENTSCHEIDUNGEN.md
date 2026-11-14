# Entscheidungen

Getroffene Architektur- und Produktentscheidungen mit Begründung, damit sie nicht erneut diskutiert oder
versehentlich rückgängig gemacht werden. Neueste oben.

| Nr. | Entscheidung | Begründung / Kontext |
|---|---|---|
| E18 | Themen-Berechnung läuft **ohne** gesenkte Thread-Priorität | Die Bildsuche rechnet ständig; ein gebremster Thread bekam kaum CPU („Begriffe vorbereiten“ hing über 10 Minuten, 2026-11-12a) |
| E17 | **Kein Zugriff aus dem Internet**; Zugriff von unterwegs nur per VPN (Tailscale oder DSM-VPN) | Familienfotoarchiv; selbstgebauter Webserver ohne Login. Konzept für DSM-Login (Login gegen `SYNO.API.Auth`, Allowlist, 2FA-Feld, eigene Sperre gegen Passwort-Raten, Reverse Proxy mit HTTPS) liegt vor, wird nur auf ausdrücklichen Wunsch gebaut |
| E16 | Bildansicht als Overlay im Report statt neuem Tab | iPad-Web-App (Home-Bildschirm) hat keine Tabs/Zurück |
| E15 | Protokoll zusätzlich als Datei, Anzeige im Browser; **kein** Zugriff auf Docker-Socket | Docker-Socket im Container wäre ein Sicherheitsrisiko |
| E14 | Geteilte Alben bevorzugt mit Rolle `download` (Kopiervorlage), konfigurierbar | Ein Album gehört immer dem anlegenden Benutzer, Besitz ist nicht übertragbar, löschen kann nur der Besitzer; `manage` ist ein verwirrender Zwischenweg |
| E13 | Album löschen über die Bildsuche-Webseite (Container handelt als Besitzer) | geteilte Benutzer können Alben nicht löschen |
| E12 | Alben werden nur **ergänzt**, nie geleert; Probelauf ersetzt nur das Testalbum | keine Überraschungen für manuell gepflegte Alben |
| E11 | „Ähnliche Bilder“ nur auf Wunsch mitnehmen (Häkchen, Standard aus) | eine Auswahl soll genau das gezeigte Bild betreffen |
| E10 | Eigener DSM-Benutzer (`bildsuche`) für die Photos-API, **nicht** das Hauptkonto | Passwort des Hauptkontos nie im Container; der Nutzer besteht darauf |
| E9 | Photos-Berechtigungen in **Synology Photos** setzen (jeder Unterordner), nicht per `chmod`/`synoacltool` | Photos führt eigene Rechte-Datenbank; `synoacltool -enforce-inherit` hat einmal eine ACL ersetzt |
| E8 | Photos-Login mit `session=webui` | andere Session-Namen → Fehler 402, auch für Admins |
| E7 | Webseite im Heimnetz **ohne Passwort** | ausdrücklicher Wunsch des Nutzers (siehe E17 für außen) |
| E6 | Ein Index pro Bildpfad, Zurückwechseln nutzt den alten weiter; Schutz vor Massenlöschung | Pfadwechsel darf keine Tage Rechenzeit vernichten |
| E5 | Adaptive Kandidatenmenge statt fester Grenzen | feste Grenzen haben Treffer abgeschnitten; unter der Grob-Schwelle finden sich weiter 3–5 % Treffer |
| E4 | Dreistufig: CLIP-Gesamtbild → Kacheln → SigLIP-2-Prüfung; Prüfung hat Vorrang | hohe Qualität gefordert, eine Woche Rechenzeit akzeptiert, Zwischenergebnisse nutzbar |
| E3 | Drosselung über `cpu_shares` + `THREADS`, nicht `cpus:` | Synology-Kernel ohne CFS-Quota (NanoCPUs-Fehler) |
| E2 | Code aus `app/` einbinden statt ins Image backen | Update = Datei ersetzen + Neustart, kein Rebuild |
| E1 | Alles auf der NAS, keine Cloud | Datenschutz; ausdrücklicher Wunsch |

## Ausdrücklich abgelehnt (nicht wieder anbieten)

- **Tags in Bilddateien oder Photos schreiben** (IPTC/XMP/Photos-Tags) – „das Tagging will ich so nicht“
- **Mac-Hilfsprogramm mit `bildsuche://`-Links** für die Vorschau
- **Report als einzelne HTML-Datei mit eingebetteten Bildern** (base64)
- „Ordner öffnen“-Links im Report
- Öffentlicher Zugriff mit Login (vorerst; siehe E17)
