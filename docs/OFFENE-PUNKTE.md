# Offene Punkte und Ideen

## Technisch

- [ ] Automatische Tests ausbauen (`tools/check.py` ist der Anfang): Scoring, `pool_size`, Konfig-Prüfung, Album-Abgleich gegen eine Photos-Attrappe
- [ ] Optional: GitHub Actions, die `tools/check.py` bei jedem Push ausführen
- [ ] Optional: `app/bildsuche.py` in Module aufteilen, sobald es weiter wächst (Konfiguration, Index, Engine, Web, Photos)
- [ ] Kandidatenmenge: bei weiter sinkender Trefferquote prüfen, ob `expand_if_rate` 0,02 sinnvoll ist (Rechenzeit vs. späte Treffer)

## Fachlich / Betrieb

- [ ] Suchen nachjustieren, sobald alle Stufen einmal durch sind (z. B. Klettersteig-Suche: viele Treffer, Schwelle oder gezielte Gegenkategorien)
- [ ] Nach Updates von Synology Photos: Probelauf testen (inoffizielle API)
- [ ] Assistent im Alltag erproben (Freigabe-Grenze 60 %, Anzahl Probebilder, Formulierungsvorschläge)

## Ideen (nur auf Wunsch)

- Anmeldung mit DSM-Benutzern für Zugriff von außen (Konzept in E17)
- „Auswahl aus Album entfernen“ über die Webseite
