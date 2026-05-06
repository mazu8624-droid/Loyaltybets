"""
Backtesting automático
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Guarda cada pick que hace la herramienta.
Al día siguiente verifica el resultado real contra API-Football.
Calcula accuracy por deporte y mercado.
"""

import json
import os
import requests
from datetime import datetime, timedelta, timezone

PICKS_FILE = "picks_history.json"
CHILE_TZ   = timezone(timedelta(hours=-4))
APIF_BASE  = "https://v3.football.api-sports.io"


def load_history():
    if not os.path.exists(PICKS_FILE):
        return []
    try:
        with open(PICKS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(history):
    try:
        with open(PICKS_FILE, "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def save_pick(match, sport, pick_label, pick_odds, pick_prob, edge,
              fixture_id=None, home=None, away=None):
    """Guarda un pick nuevo en el historial."""
    history = load_history()
    now_cl  = datetime.now(CHILE_TZ)
    history.append({
        "date":        now_cl.strftime("%Y-%m-%d"),
        "time":        now_cl.strftime("%H:%M"),
        "match":       match,
        "sport":       sport,
        "pick_label":  pick_label,
        "pick_odds":   round(pick_odds, 2),
        "pick_prob":   round(pick_prob * 100, 1),
        "edge":        round(edge, 1),
        "fixture_id":  fixture_id,
        "home":        home,
        "away":        away,
        "result":      None,
        "correct":     None,
        "checked":     False,
        "profit_loss": None,
    })
    save_history(history)


def check_result_apif(fixture_id, api_key):
    """
    Verifica el resultado de un partido de fútbol en API-Football.
    Retorna {'home_score': int, 'away_score': int, 'status': str} o None.
    """
    try:
        r = requests.get(
            f"{APIF_BASE}/fixtures",
            params={"id": fixture_id},
            headers={"x-apisports-key": api_key},
            timeout=8,
        )
        if r.status_code != 200:
            return None
        data = r.json().get("response", [])
        if not data:
            return None
        fix    = data[0]
        status = fix.get("fixture", {}).get("status", {}).get("short", "")
        goals  = fix.get("goals", {})
        if status in ("FT", "AET", "PEN") and goals.get("home") is not None:
            return {
                "home_score": goals["home"],
                "away_score": goals["away"],
                "status":     status,
            }
    except Exception:
        pass
    return None


def pick_correct(pick_label, result, home, away):
    """
    Determina si el pick fue correcto dado el resultado.
    Retorna True/False/None (si no se puede determinar).
    """
    if not result:
        return None
    hs = result["home_score"]
    as_ = result["away_score"]

    label = pick_label.lower()

    # Ganador
    if "local" in label or (home and home.lower() in label):
        return hs > as_
    if "visitante" in label or (away and away.lower() in label):
        return as_ > hs
    if "empate" in label:
        return hs == as_

    # Más/Menos
    total = hs + as_
    if "más de" in label:
        try:
            line = float(label.split("más de")[1].strip().split()[0])
            return total > line
        except Exception:
            return None
    if "menos de" in label:
        try:
            line = float(label.split("menos de")[1].strip().split()[0])
            return total < line
        except Exception:
            return None

    return None


def verify_pending_picks(apif_key):
    """
    Verifica todos los picks pendientes (checked=False) que sean de ayer o antes.
    Solo para picks de fútbol con fixture_id disponible.
    """
    history  = load_history()
    now_cl   = datetime.now(CHILE_TZ)
    today    = now_cl.strftime("%Y-%m-%d")
    updated  = 0

    for pick in history:
        if pick.get("checked"):
            continue
        if pick.get("date") == today:
            continue  # Partido de hoy, aún no terminó
        if not pick.get("fixture_id"):
            pick["checked"] = True  # Sin ID no podemos verificar
            pick["result"]  = "Sin datos"
            continue

        result = check_result_apif(pick["fixture_id"], apif_key)
        if not result:
            continue  # Aún no hay resultado, lo dejamos pendiente

        correct = pick_correct(
            pick["pick_label"], result,
            pick.get("home"), pick.get("away")
        )

        pick["result"]    = f"{result['home_score']}-{result['away_score']}"
        pick["correct"]   = correct
        pick["checked"]   = True
        pick["profit_loss"] = round(pick["pick_odds"] - 1, 2) if correct else -1.0
        updated += 1

    if updated:
        save_history(history)

    return updated


def get_stats():
    """
    Calcula estadísticas de rendimiento del historial.
    Retorna dict con accuracy por deporte y global.
    """
    history = load_history()
    checked = [p for p in history if p.get("checked") and p.get("correct") is not None]

    if not checked:
        return {"total": 0, "correct": 0, "accuracy": 0,
                "profit": 0, "by_sport": {}, "by_market": {}}

    total   = len(checked)
    correct = sum(1 for p in checked if p["correct"])
    profit  = sum(p.get("profit_loss", 0) or 0 for p in checked)

    # Por deporte
    by_sport = {}
    for p in checked:
        sport = p.get("sport", "Otro")
        if sport not in by_sport:
            by_sport[sport] = {"total": 0, "correct": 0}
        by_sport[sport]["total"]   += 1
        by_sport[sport]["correct"] += 1 if p["correct"] else 0

    # Por mercado
    by_market = {"Ganador": {"total":0,"correct":0}, "Más/Menos": {"total":0,"correct":0}}
    for p in checked:
        label = p.get("pick_label","").lower()
        if "más de" in label or "menos de" in label:
            mkt = "Más/Menos"
        else:
            mkt = "Ganador"
        by_market[mkt]["total"]   += 1
        by_market[mkt]["correct"] += 1 if p["correct"] else 0

    return {
        "total":    total,
        "correct":  correct,
        "accuracy": round(correct / total * 100, 1),
        "profit":   round(profit, 2),
        "by_sport": by_sport,
        "by_market":by_market,
        "history":  sorted(history, key=lambda x: x["date"], reverse=True)[:30],
    }
