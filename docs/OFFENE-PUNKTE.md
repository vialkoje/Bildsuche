# Offene Punkte und Ideen

## Technisch

- [ ] Vor dem ersten Ausrollen aus dem Repository prüfen, dass `link_root`, `web_base`, `photos_url` und
  `photos_share_with` in `searches.yaml` der NAS stehen – auf der NAS standen sie bis 2026-11-13c in den `DEFAULTS`
  des Codes, im Repository sind dort nur Platzhalter

- [ ] Automatische Tests ausbauen (`tools/check.py` ist der Anfang): Scoring, `pool_size`, Konfig-Prüfung, Album-Abgleich gegen eine Photos-Attrappe
- [ ] Optional: GitHub Actions, die `tools/check.py` bei jedem Push ausführen
- [ ] Optional: `app/bildsuche.py` in Module aufteilen, sobald es weiter wächst (Konfiguration, Index, Engine, Web, Photos)
- [ ] Kandidatenmenge: bei weiter sinkender Trefferquote prüfen, ob `expand_if_rate` 0,02 sinnvoll ist (Rechenzeit vs. späte Treffer)

## Fachlich / Betrieb

- [ ] Suchen nachjustieren, sobald alle Stufen einmal durch sind (z. B. Klettersteig-Suche: viele Treffer, Schwelle oder gezielte Gegenkategorien)
- [ ] Nach Updates von Synology Photos: Probelauf testen (inoffizielle API)
- [ ] Themen im Alltag erproben: Namen, Gruppengrößen (`THEMEN_TOP_K`, `THEMEN_LEAF`), Lücken in der Begriffsliste
- [ ] Assistent im Alltag erproben (Freigabe-Grenze 60 %, Anzahl Probebilder, Formulierungsvorschläge)

## Ideen (nur auf Wunsch)

- Anmeldung mit DSM-Benutzern für Zugriff von außen (Konzept in E17)
- „Auswahl aus Album entfernen“ über die Webseite
