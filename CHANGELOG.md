# Changelog

Versionen sind nach Datum benannt (`JJJJ-MM-TT` + Buchstabe). Die laufende Version steht im Protokoll beim Start
und auf jeder Report-Seite.

| Version | Änderungen |
|---|---|
| 2026-10-21e | Bildansicht über dem Report statt neuem Tab (Blättern, Auswählen, Zurück-Geste; funktioniert in der iPad-Web-App) |
| 2026-10-21d | Fix: Absturz der Protokollierung bei kaputten Anfragen (z. B. HTTPS auf dem HTTP-Port); 400/404 werden nicht mehr gemeldet |
| 2026-10-21c | Webserver-Fehler und Fehler in Hintergrund-Threads landen im Protokoll |
| 2026-10-21b | Protokoll-Seite `/protokoll` mit Filtern und Download, Protokolldatei in `data/logs/` |
| 2026-10-21a | „Ältere Reports“ auf der Übersicht mit Link und Löschknopf |
| 2026-10-20c | Namen verschwundener und geänderter Bilder im Protokoll |
| 2026-10-20b | Knopf „Album löschen“ |
| 2026-10-20a | einstellbare Rolle beim Teilen (`photos_share_role`) mit Rückfall auf schwächere Rollen; bestehende Alben werden neu geteilt |
| 2026-10-17d | „ähnliche Bilder mitnehmen“ optional; robuster Abgleich mit Photos (Groß-/Kleinschreibung, RAW+JPG, Live-Fotos); nicht gefundene Dateien werden benannt; Obergrenze `candidates_max` exakt |
| 2026-10-17c | ein Index pro Bildpfad, Konfigurationseditor im Browser, Schutz vor Massenlöschung im Index |
| 2026-10-17 | Album-Funktion für Synology Photos (alle/Auswahl, Teilen, Probelauf), Anmeldung mit `session=webui` |
| frühere Versionen | Webseite auf Port 8091 mit Bildansicht (HEIC/RAW-Umwandlung); dreistufiges Verfahren mit SigLIP-2-Prüfung, adaptive Kandidatenmenge, Bündelung ähnlicher Bilder; erste Version mit CLIP-Index, Reports als HTML/CSV und Drosselung für die Synology |
