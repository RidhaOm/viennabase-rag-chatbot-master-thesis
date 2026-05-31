# Evaluation Guide

Schritt-für-Schritt-Anleitung zur Durchführung der drei Evaluationsläufe.

---

## 1. Ordnerstruktur

```
<projektordner>/
├── chatbot.py
├── api.py
├── ...
└── eval/
    ├── eval_runner.py
    ├── EVAL_GUIDE.md
    ├── data/
    │   └── questions_clean.csv
    ├── results/                        ← wird automatisch erstellt
    │   ├── results_baseline.csv
    │   ├── results_no_threshold.csv
    │   ├── results_no_history.csv
    │   ├── run_meta_baseline.json
    │   ├── run_meta_no_threshold.json
    │   └── run_meta_no_history.json
    ├── labels/                         ← annotierte Labels
    ├── analysis/                       ← Metriken und Statistik
    └── figures/                        ← generierte Plots
```

Der `results/`-Ordner wird beim ersten Lauf automatisch angelegt.

---

## 2. Die drei Konfigurationen

| Config          | `USE_EVIDENCE_THRESHOLD` | `USE_HISTORY_REFORMULATION` |
|-----------------|:------------------------:|:---------------------------:|
| baseline        | **True**                 | **True**                    |
| no_threshold    | **False**                | True                        |
| no_history      | True                     | **False**                   |

Die Toggles werden manuell vor jedem Run gesetzt — der `--config`-Parameter dient nur als Label für die Ausgabedateien.

---

## 3. Quick-Test (empfohlen vor den echten Runs)

Bevor die vollständigen Runs gestartet werden, empfiehlt sich ein kurzer Test mit 3 Fragen. Dieser dauert ca. 15 Sekunden.

**Schritte:**

1. Toggles prüfen (baseline-Zustand):
   - `chatbot.py`, Zeile ~60: `USE_EVIDENCE_THRESHOLD = True`
   - `api.py`, Zeile ~35: `USE_HISTORY_REFORMULATION = True`

2. FastAPI starten (Terminal 1):
   ```bash
   uvicorn api:app --reload
   ```
   Warten bis `Application startup complete` erscheint (der Vektorstore wird beim Start neu aufgebaut, das dauert ca. 30 Sekunden).

3. Test-Run (Terminal 2, aus dem Projekt-Root):
   ```bash
   python eval/eval_runner.py --config baseline --limit 3
   ```

4. `eval/results/results_baseline.csv` öffnen und prüfen:
   - 3 Zeilen mit den ersten 3 Fragen
   - `ok = True`
   - `answer` enthält deutschen Text
   - `sources_returned` ist ein JSON-Array mit `source`/`label`-Objekten
   - `latency_s` liegt typischerweise zwischen 1 und 5 Sekunden
   - Bei unbeantwortbaren Fragen: `no_context = True`, `answer` enthält die Fallback-Nachricht

5. Test-Dateien löschen:
   ```bash
   rm eval/results/results_baseline.csv
   rm eval/results/run_meta_baseline.json
   ```

---

## 4. Die drei Runs

### Run 1: baseline

Toggles (sollten nach dem Quick-Test bereits gesetzt sein):
- `chatbot.py`: `USE_EVIDENCE_THRESHOLD = True`
- `api.py`: `USE_HISTORY_REFORMULATION = True`

FastAPI neu starten:
```bash
uvicorn api:app --reload
```

Runner starten:
```bash
python eval/eval_runner.py --config baseline
```

Dauer: ca. 8–15 Minuten. Ausgabe: `eval/results/results_baseline.csv`.

---

### Run 2: no_threshold

Nur eine Zeile ändern:
- `chatbot.py`: `USE_EVIDENCE_THRESHOLD = False`
- `api.py`: bleibt unverändert (`USE_HISTORY_REFORMULATION = True`)

FastAPI neu starten, dann:
```bash
python eval/eval_runner.py --config no_threshold
```

---

### Run 3: no_history

Toggles:
- `chatbot.py`: `USE_EVIDENCE_THRESHOLD = True`
- `api.py`: `USE_HISTORY_REFORMULATION = False`

FastAPI neu starten, dann:
```bash
python eval/eval_runner.py --config no_history
```

Dauer kürzer als die anderen Runs — kein Reformulations-LLM-Call. Ca. 5–8 Minuten.

---

## 5. Nach den Runs

Toggles zurückstellen (Produktionszustand):
- `chatbot.py`: `USE_EVIDENCE_THRESHOLD = True`
- `api.py`: `USE_HISTORY_REFORMULATION = True`

Nach drei erfolgreichen Runs liegen folgende Dateien vor:
- `eval/results/results_baseline.csv`
- `eval/results/results_no_threshold.csv`
- `eval/results/results_no_history.csv`
- je eine `run_meta_*.json` pro Run

Sanity-Check: In allen drei `run_meta_*.json` sollte `"n_ok": 79, "n_error": 0` stehen. Falls nicht, die `error`-Spalte in der entsprechenden Ergebnis-CSV prüfen.

---

## 6. Metriken und Plots berechnen

Metriken (nach der Annotation):
```bash
python eval/scripts/metrics.py \
    --labels-dir eval/labels \
    --results-dir eval/results \
    --out eval/analysis/metrics_summary.csv
```

Statistiktests:
```bash
python eval/scripts/statistics_tests.py \
    --labels-dir eval/labels \
    --out-dir eval/analysis
```

Plots:
```bash
python eval/scripts/plots.py \
    --labels-dir eval/labels \
    --results-dir eval/results \
    --analysis-dir eval/analysis \
    --out-dir eval/figures
```

---

## Troubleshooting

| Problem | Ursache / Lösung |
|---|---|
| `ConnectionError` bei allen Calls | FastAPI läuft nicht — `uvicorn api:app --reload` starten |
| `http_400: Eingabe enthält unzulässige Muster` | Input-Guardrail hat eine Frage abgelehnt (z. B. wegen `\|` oder `;`) |
| Sehr langsame Runs (>10 s/Frage) | OpenAI Rate-Limiting — `--sleep 1.0` verwenden |
| Abgebrochener Run | Die CSV wird zeilenweise geschrieben; bereits abgeschlossene Fragen sind erhalten. Run nach dem Löschen der Datei neu starten. |
| Zweiter Run zeigt identische Antworten | FastAPI nach dem Toggle-Wechsel nicht neu gestartet |
| `No module named 'eval'` | Script wird nicht aus dem Projekt-Root aufgerufen |
