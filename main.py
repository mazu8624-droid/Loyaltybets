"""
BetAnalyzer Pro — Versión completa
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keys necesarias (ambas gratuitas en api-sports.io):
  • The Odds API key
  • API-Sports key  →  sirve para fútbol, basketball Y béisbol

Capas de señal por deporte:
  ⚽ Fútbol:
    40% API-Football  (stats, lesiones, H2H, predicción)
    30% Forebet + Sofascore
    20% TheSportsDB histórico
    10% Movimiento de cuotas

  🏀 Basketball:
    40% API-Basketball (puntos/partido, win%, forma, H2H)
    30% Sofascore
    20% TheSportsDB histórico
    10% Movimiento de cuotas

  ⚾ Béisbol:
    40% API-Baseball  (carreras, ERA, forma, H2H)
    30% MLB Stats API (stats oficiales temporada)
    20% Sofascore
    10% Movimiento de cuotas

  🎾 Tenis:
    60% Sofascore + TheSportsDB
    30% Histórico TheSportsDB
    10% Movimiento de cuotas

Modelo estadístico:
  ⚽ Poisson bivariado (Dixon-Coles simplificado)
  🏀 Pythagorean basketball (exponente 13.91, Morey)
  ⚾ Pythagorean béisbol (exponente 1.83, James)
  🎾 Win% ajustado

Backtesting automático: guarda picks → verifica al día siguiente
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import requests
from datetime import datetime, timedelta, timezone
import time
import re
import math
from statistics import NormalDist
from bs4 import BeautifulSoup
import backtesting as bt

# ══════════════════════════════════════════════════════════════════
#  1. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
APIF_BASE     = "https://v3.football.api-sports.io"       # Fútbol
APIB_BASE     = "https://v1.basketball.api-sports.io"     # Basketball
APIBB_BASE    = "https://v1.baseball.api-sports.io"       # Béisbol
TSDB_BASE     = "https://www.thesportsdb.com/api/v1/json/3"
MLB_BASE      = "https://statsapi.mlb.com/api/v1"
SOFA_BASE     = "https://api.sofascore.com/api/v1"
FOREBET_URL   = "https://www.forebet.com/en/football-tips-and-predictions-for-today"

CHILE_TZ      = timezone(timedelta(hours=-4))
SEASON_FOOT   = datetime.now().year
SEASON_BBALL  = "2024-2025"
SEASON_BASE   = datetime.now().year
NBA_LEAGUE_ID = 12
MLB_LEAGUE_ID = 1

MM_PROB_THRESHOLD = 0.60

BROWSER_HDR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

SPORTS = {
    "⚽ Fútbol": {
        "odds_keys": [
            "soccer_epl","soccer_spain_la_liga","soccer_italy_serie_a",
            "soccer_germany_bundesliga","soccer_france_ligue_one",
            "soccer_conmebol_copa_libertadores","soccer_chile_campeonato",
            "soccer_argentina_primera_division","soccer_mexico_ligamx",
            "soccer_brazil_campeonato","soccer_usa_mls",
            "soccer_uefa_champs_league","soccer_uefa_europa_league",
        ],
        "has_draw": True,  "model": "poisson",
        "icon": "⚽",      "unit": "goles",
        "sofa_sport": "football",
    },
    "🏀 Basketball": {
        "odds_keys": ["basketball_nba","basketball_ncaab","basketball_euroleague"],
        "has_draw": False, "model": "pythagorean_bball",
        "icon": "🏀",      "unit": "puntos",
        "sofa_sport": "basketball",
    },
    "⚾ Béisbol": {
        "odds_keys": ["baseball_mlb"],
        "has_draw": False, "model": "pythagorean_base",
        "icon": "⚾",      "unit": "carreras",
        "sofa_sport": "baseball",
    },
    "🎾 Tenis": {
        "odds_keys": [
            "tennis_atp_french_open","tennis_wta_french_open",
            "tennis_atp_wimbledon","tennis_wta_wimbledon",
            "tennis_atp_us_open","tennis_wta_us_open",
            "tennis_atp_australian_open","tennis_wta_australian_open",
        ],
        "has_draw": False, "model": "general",
        "icon": "🎾",      "unit": "games",
        "sofa_sport": "tennis",
    },
}

# ══════════════════════════════════════════════════════════════════
#  2. UTILIDADES
# ══════════════════════════════════════════════════════════════════

def ip(odds):
    return (1.0 / odds) if odds > 1.0 else 0.0

def edge(my_p, odds):
    return (my_p - ip(odds)) * 100.0 if odds > 1 else 0.0

def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

def fuzzy_match(a, b, threshold=5):
    an, bn = norm_name(a), norm_name(b)
    return an[:threshold] in bn or bn[:threshold] in an or an in bn or bn in an


# ══════════════════════════════════════════════════════════════════
#  3. THE ODDS API
# ══════════════════════════════════════════════════════════════════

def get_todays_games(api_key, sport_keys):
    now_cl    = datetime.now(CHILE_TZ)
    day_start = now_cl.replace(hour=0,  minute=0,  second=0,  microsecond=0)
    day_end   = now_cl.replace(hour=23, minute=59, second=59, microsecond=0)
    games     = []
    for key in sport_keys:
        try:
            r = requests.get(f"{ODDS_API_BASE}/sports/{key}/odds/",
                params={"apiKey":api_key,"regions":"eu,us",
                        "markets":"h2h,totals","oddsFormat":"decimal",
                        "dateFormat":"iso"}, timeout=12)
            if r.status_code == 401:
                st.error("❌ Odds API Key inválida.")
                return []
            if r.status_code not in (200,422): continue
            for g in r.json():
                gt    = datetime.fromisoformat(g["commence_time"].replace("Z","+00:00"))
                gt_cl = gt.astimezone(CHILE_TZ)
                if day_start <= gt_cl <= day_end:
                    g["_time_cl"]   = gt_cl
                    g["_sport_key"] = key
                    games.append(g)
            time.sleep(0.2)
        except Exception:
            continue
    return sorted(games, key=lambda x: x["_time_cl"])


def extract_h2h_odds(game):
    bests = {"home":0.0,"away":0.0,"draw":0.0}
    for bk in game.get("bookmakers",[]):
        for mkt in bk.get("markets",[]):
            if mkt["key"] != "h2h": continue
            for oc in mkt["outcomes"]:
                p = float(oc["price"])
                if oc["name"] == game["home_team"]:   bests["home"] = max(bests["home"],p)
                elif oc["name"] == game["away_team"]: bests["away"] = max(bests["away"],p)
                elif oc["name"] == "Draw":            bests["draw"] = max(bests["draw"],p)
    return bests


def extract_all_totals(game):
    lines = {}
    for bk in game.get("bookmakers",[]):
        for mkt in bk.get("markets",[]):
            if mkt["key"] != "totals": continue
            line=over_p=under_p=None
            for oc in mkt["outcomes"]:
                try:    line = float(oc.get("description") or 0)
                except: line = 0.0
                p = float(oc["price"])
                if oc["name"] == "Over":    over_p  = p
                elif oc["name"] == "Under": under_p = p
            if line and over_p:
                if line not in lines or over_p > lines[line]["over_odds"]:
                    lines[line] = {"line":line,"over_odds":over_p,"under_odds":under_p or 0.0}
    return sorted(lines.values(), key=lambda x: x["line"])


def get_opening_odds(api_key, sport_key, event_id):
    try:
        r = requests.get(
            f"{ODDS_API_BASE}/historical/sports/{sport_key}/events/{event_id}/odds",
            params={"apiKey":api_key,"regions":"eu","markets":"h2h","oddsFormat":"decimal"},
            timeout=8)
        if r.status_code != 200: return {}
        data = r.json()
        first = data[0] if isinstance(data,list) and data else {}
        for bk in first.get("bookmakers",[])[:1]:
            for mkt in bk.get("markets",[]):
                if mkt["key"] != "h2h": continue
                result = {}
                for oc in mkt["outcomes"]:
                    name = oc.get("name","").lower()
                    if "home" in name: result["home_open"] = float(oc["price"])
                    elif "away" in name: result["away_open"] = float(oc["price"])
                return result
    except Exception:
        pass
    return {}


def odds_movement_signal(opening, current):
    if not opening or not opening.get("home_open"): return {"home":0.0,"away":0.0}
    ho = opening.get("home_open",current["home"])
    ao = opening.get("away_open",current["away"])
    hm = (ho - current["home"]) / ho if ho > 0 else 0
    am = (ao - current["away"]) / ao if ao > 0 else 0
    return {
        "home": max(min(hm*0.25, 0.05), -0.05),
        "away": max(min(am*0.25, 0.05), -0.05),
    }


# ══════════════════════════════════════════════════════════════════
#  4. API-SPORTS — Helper común
# ══════════════════════════════════════════════════════════════════

def apis_get(base_url, endpoint, params, key):
    """Helper genérico para cualquier API-Sports (fútbol, basket, béisbol)."""
    try:
        r = requests.get(f"{base_url}/{endpoint}", params=params,
                         headers={"x-apisports-key": key}, timeout=10)
        if r.status_code == 200:
            return r.json().get("response", [])
    except Exception:
        pass
    return []


# ══════════════════════════════════════════════════════════════════
#  5. API-FOOTBALL — Señal fútbol
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800)
def apif_fixtures_today(date_str, key):
    return apis_get(APIF_BASE, "fixtures", {"date": date_str}, key)


def apif_find_fixture(home, away, fixtures):
    for fix in fixtures:
        fh = fix.get("teams",{}).get("home",{}).get("name","")
        fa = fix.get("teams",{}).get("away",{}).get("name","")
        if fuzzy_match(home, fh) and fuzzy_match(away, fa):
            return fix
    return None


def apif_team_form(team_id, key, last=5):
    data = apis_get(APIF_BASE, "fixtures",
                    {"team":team_id,"last":last,"status":"FT"}, key)
    if not data: return None
    wins=gs=gc=n=0
    for fix in data:
        goals = fix.get("goals",{}); hs=goals.get("home",0) or 0; as_=goals.get("away",0) or 0
        is_home = fix.get("teams",{}).get("home",{}).get("id") == team_id
        sc,cc = (hs,as_) if is_home else (as_,hs)
        gs+=sc; gc+=cc; n+=1
        if sc>cc: wins+=1
    if n==0: return None
    return {"win_pct":wins/n,"goals_scored_pg":gs/n,"goals_conceded_pg":gc/n,
            "games":n,"src":"API-Football"}


def apif_prediction(fixture_id, key):
    data = apis_get(APIF_BASE, "predictions", {"fixture":fixture_id}, key)
    if not data: return None
    p    = data[0].get("predictions",{})
    perc = p.get("percent",{})
    def pct(v): return float((v or "0%").replace("%",""))/100
    return {
        "home_pct": pct(perc.get("home")),
        "draw_pct": pct(perc.get("draw")),
        "away_pct": pct(perc.get("away")),
        "goals_home": p.get("goals",{}).get("home"),
        "goals_away": p.get("goals",{}).get("away"),
    }


def apif_h2h_stats(team1_id, team2_id, home_name, key):
    data = apis_get(APIF_BASE,"fixtures/headtohead",
                    {"h2h":f"{team1_id}-{team2_id}","last":8}, key)
    if not data: return None
    wins=draws=losses=n=0
    for fix in data:
        status = fix.get("fixture",{}).get("status",{}).get("short","")
        if status not in ("FT","AET","PEN"): continue
        goals = fix.get("goals",{}); hs=goals.get("home",0) or 0; as_=goals.get("away",0) or 0
        fh    = fix.get("teams",{}).get("home",{}).get("name","")
        is_home = fuzzy_match(home_name, fh)
        sc,cc = (hs,as_) if is_home else (as_,hs)
        n+=1
        if sc>cc: wins+=1
        elif sc==cc: draws+=1
        else: losses+=1
    if n==0: return None
    return {"win_pct":wins/n,"draw_pct":draws/n,"n":n}


def apif_injuries(fixture_id, key):
    data = apis_get(APIF_BASE, "injuries", {"fixture":fixture_id}, key)
    return data or []


def build_football_signal(home, away, fixtures, key):
    fix = apif_find_fixture(home, away, fixtures)
    if not fix: return {}
    fix_id  = fix.get("fixture",{}).get("id")
    home_id = fix.get("teams",{}).get("home",{}).get("id")
    away_id = fix.get("teams",{}).get("away",{}).get("id")
    sig = {"fixture_id": fix_id}

    pred = apif_prediction(fix_id, key)
    if pred: sig["pred"] = pred
    time.sleep(0.3)

    if home_id:
        f = apif_team_form(home_id, key)
        if f: sig["home_form"] = f
        time.sleep(0.3)
    if away_id:
        f = apif_team_form(away_id, key)
        if f: sig["away_form"] = f
        time.sleep(0.3)

    if home_id and away_id:
        h2h = apif_h2h_stats(home_id, away_id, home, key)
        if h2h: sig["h2h"] = h2h
        time.sleep(0.3)

    injuries = apif_injuries(fix_id, key)
    sig["home_injuries"] = sum(1 for i in injuries if i.get("team",{}).get("id")==home_id)
    sig["away_injuries"] = sum(1 for i in injuries if i.get("team",{}).get("id")==away_id)

    return sig


# ══════════════════════════════════════════════════════════════════
#  6. API-BASKETBALL — Señal basketball
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def apib_all_teams(key):
    """Todos los equipos NBA desde API-Basketball."""
    return apis_get(APIB_BASE, "teams", {"league":NBA_LEAGUE_ID,"season":SEASON_BBALL}, key)


def apib_find_team(name, teams):
    for t in teams:
        tn = t.get("name","")
        if fuzzy_match(name, tn): return t
    return None


def apib_team_stats(team_id, key):
    """
    Estadísticas de temporada del equipo:
    puntos anotados/recibidos por partido, win%.
    """
    data = apis_get(APIB_BASE, "teams/statistics",
                    {"league":NBA_LEAGUE_ID,"season":SEASON_BBALL,"team":team_id}, key)
    if not data: return None
    d = data[0]
    games   = d.get("games",{})
    points  = d.get("points",{})
    wins    = games.get("wins",{}).get("all",{})
    losses  = games.get("losses",{}).get("all",{})
    total_w = wins.get("total",0) or 0
    total_l = losses.get("total",0) or 0
    total_g = total_w + total_l
    pts_for  = points.get("for",{}).get("average",{}).get("all", None)
    pts_ag   = points.get("against",{}).get("average",{}).get("all", None)
    if not pts_for or not pts_ag or total_g == 0: return None
    return {
        "win_pct":    total_w / total_g,
        "pts_for_pg": float(pts_for),
        "pts_ag_pg":  float(pts_ag),
        "games":      total_g,
        "src":        "API-Basketball",
    }


def apib_team_form(team_id, key, last=5):
    """Últimos N partidos del equipo NBA."""
    data = apis_get(APIB_BASE, "games",
                    {"league":NBA_LEAGUE_ID,"season":SEASON_BBALL,
                     "team":team_id,"last":last}, key)
    if not data: return None
    wins=pts_for=pts_ag=n=0
    for g in data:
        status = g.get("status",{}).get("long","")
        if "Finished" not in status: continue
        scores = g.get("scores",{})
        ht_id  = g.get("teams",{}).get("home",{}).get("id")
        is_home = ht_id == team_id
        home_pts = scores.get("home",{}).get("total") or 0
        away_pts = scores.get("away",{}).get("total") or 0
        pf, pa  = (home_pts,away_pts) if is_home else (away_pts,home_pts)
        pts_for+=pf; pts_ag+=pa; n+=1
        if pf>pa: wins+=1
    if n==0: return None
    return {"win_pct":wins/n,"pts_for_pg":pts_for/n,"pts_ag_pg":pts_ag/n,
            "games":n,"src":"API-Basketball (forma reciente)"}


def apib_h2h(team1_id, team2_id, key, last=8):
    """H2H entre dos equipos NBA."""
    data = apis_get(APIB_BASE, "games/h2h",
                    {"h2h":f"{team1_id}-{team2_id}","last":last}, key)
    if not data: return None
    wins=n=0
    for g in data:
        status = g.get("status",{}).get("long","")
        if "Finished" not in status: continue
        scores  = g.get("scores",{})
        ht_id   = g.get("teams",{}).get("home",{}).get("id")
        is_home = ht_id == team1_id
        hp = scores.get("home",{}).get("total") or 0
        ap = scores.get("away",{}).get("total") or 0
        pf,pa = (hp,ap) if is_home else (ap,hp)
        n+=1
        if pf>pa: wins+=1
    if n==0: return None
    return {"win_pct":wins/n,"n":n}


def build_basketball_signal(home, away, key):
    """
    Construye señal completa de API-Basketball para un partido NBA.
    """
    all_teams = apib_all_teams(key)
    home_t    = apib_find_team(home, all_teams)
    away_t    = apib_find_team(away, all_teams)
    if not home_t or not away_t: return {}

    home_id = home_t.get("id"); away_id = away_t.get("id")
    sig = {}

    # Stats de temporada
    hs = apib_team_stats(home_id, key)
    if hs: sig["home_stats"] = hs
    time.sleep(0.3)

    as_ = apib_team_stats(away_id, key)
    if as_: sig["away_stats"] = as_
    time.sleep(0.3)

    # Forma reciente
    hf = apib_team_form(home_id, key)
    if hf: sig["home_form"] = hf
    time.sleep(0.3)

    af = apib_team_form(away_id, key)
    if af: sig["away_form"] = af
    time.sleep(0.3)

    # H2H
    h2h = apib_h2h(home_id, away_id, key)
    if h2h: sig["h2h"] = h2h

    return sig


# ══════════════════════════════════════════════════════════════════
#  7. API-BASEBALL — Señal béisbol
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def apibb_all_teams(key):
    """Todos los equipos MLB desde API-Baseball."""
    return apis_get(APIBB_BASE, "teams",
                    {"league":MLB_LEAGUE_ID,"season":SEASON_BASE}, key)


def apibb_find_team(name, teams):
    for t in teams:
        tn = t.get("name","")
        if fuzzy_match(name, tn): return t
    return None


def apibb_team_stats(team_id, key):
    """
    Estadísticas de temporada MLB desde API-Baseball:
    carreras anotadas/permitidas por partido, win%.
    """
    data = apis_get(APIBB_BASE, "teams/statistics",
                    {"league":MLB_LEAGUE_ID,"season":SEASON_BASE,"team":team_id}, key)
    if not data: return None
    d = data[0]
    games  = d.get("games",{})
    runs   = d.get("runs",{})
    wins   = games.get("wins",{}).get("all",{})
    losses = games.get("losses",{}).get("all",{})
    total_w = wins.get("total",0) or 0
    total_l = losses.get("total",0) or 0
    total_g = total_w + total_l
    runs_for = runs.get("for",{}).get("total") or None
    runs_ag  = runs.get("against",{}).get("total") or None
    if not runs_for or total_g == 0: return None
    return {
        "win_pct":         total_w / total_g if total_g > 0 else 0.5,
        "runs_scored_pg":  runs_for / total_g,
        "runs_allowed_pg": runs_ag / total_g if runs_ag else 4.5,
        "games":           total_g,
        "src":             "API-Baseball",
    }


def apibb_team_form(team_id, key, last=5):
    """Últimos N partidos del equipo MLB."""
    data = apis_get(APIBB_BASE, "games",
                    {"league":MLB_LEAGUE_ID,"season":SEASON_BASE,
                     "team":team_id,"last":last}, key)
    if not data: return None
    wins=rs=ra=n=0
    for g in data:
        status = g.get("status",{}).get("long","")
        if "Finished" not in status: continue
        scores = g.get("scores",{})
        ht_id  = g.get("teams",{}).get("home",{}).get("id")
        is_home = ht_id == team_id
        hp = scores.get("home",{}).get("total") or 0
        ap = scores.get("away",{}).get("total") or 0
        pf,pa = (hp,ap) if is_home else (ap,hp)
        rs+=pf; ra+=pa; n+=1
        if pf>pa: wins+=1
    if n==0: return None
    return {"win_pct":wins/n,"runs_scored_pg":rs/n,"runs_allowed_pg":ra/n,
            "games":n,"src":"API-Baseball (forma reciente)"}


def apibb_h2h(team1_id, team2_id, key, last=8):
    data = apis_get(APIBB_BASE, "games/h2h",
                    {"h2h":f"{team1_id}-{team2_id}","last":last}, key)
    if not data: return None
    wins=n=0
    for g in data:
        status = g.get("status",{}).get("long","")
        if "Finished" not in status: continue
        scores  = g.get("scores",{})
        ht_id   = g.get("teams",{}).get("home",{}).get("id")
        is_home = ht_id == team1_id
        hp = scores.get("home",{}).get("total") or 0
        ap = scores.get("away",{}).get("total") or 0
        pf,pa = (hp,ap) if is_home else (ap,hp)
        n+=1
        if pf>pa: wins+=1
    if n==0: return None
    return {"win_pct":wins/n,"n":n}


def build_baseball_signal(home, away, key):
    """Señal completa de API-Baseball + MLB Stats API para un partido."""
    all_teams = apibb_all_teams(key)
    home_t    = apibb_find_team(home, all_teams)
    away_t    = apibb_find_team(away, all_teams)
    if not home_t or not away_t: return {}

    home_id = home_t.get("id"); away_id = away_t.get("id")
    sig = {}

    hs = apibb_team_stats(home_id, key)
    if hs: sig["home_stats"] = hs
    time.sleep(0.3)

    as_ = apibb_team_stats(away_id, key)
    if as_: sig["away_stats"] = as_
    time.sleep(0.3)

    hf = apibb_team_form(home_id, key)
    if hf: sig["home_form"] = hf
    time.sleep(0.3)

    af = apibb_team_form(away_id, key)
    if af: sig["away_form"] = af
    time.sleep(0.3)

    h2h = apibb_h2h(home_id, away_id, key)
    if h2h: sig["h2h"] = h2h

    return sig


# ══════════════════════════════════════════════════════════════════
#  8. MLB STATS API — Refuerzo béisbol (oficial, sin límite)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def mlb_all_teams():
    try:
        r = requests.get(f"{MLB_BASE}/teams",
                         params={"sportId":1,"season":SEASON_BASE}, timeout=10)
        if r.status_code == 200: return r.json().get("teams",[])
    except Exception: pass
    return []


def mlb_find_id(name, teams):
    nl = name.lower(); words=[w for w in nl.split() if len(w)>2]
    for t in teams:
        if t.get("name","").lower() == nl: return t["id"]
    for t in teams:
        if any(w in t.get("name","").lower() for w in words): return t["id"]
    return None


def mlb_team_stats(team_id):
    stats = {}
    try:
        for group in ["hitting","pitching"]:
            r = requests.get(f"{MLB_BASE}/teams/{team_id}/stats",
                             params={"stats":"season","group":group,"season":SEASON_BASE},
                             timeout=8)
            if r.status_code != 200: continue
            splits = r.json().get("stats",[{}])[0].get("splits",[])
            if not splits: continue
            s = splits[0].get("stat",{}); gp=float(s.get("gamesPlayed") or 1)
            if group=="hitting":
                stats["runs_scored_pg"] = float(s.get("runs",0))/gp
                stats["games"] = int(gp)
            else: stats["runs_allowed_pg"] = float(s.get("runs",0))/gp
            time.sleep(0.2)
        r2 = requests.get(f"{MLB_BASE}/standings",
                          params={"leagueId":"103,104","season":SEASON_BASE,
                                  "standingsTypes":"regularSeason"}, timeout=8)
        if r2.status_code == 200:
            for rec in r2.json().get("records",[]):
                for tr in rec.get("teamRecords",[]):
                    if tr.get("team",{}).get("id")==team_id:
                        stats["win_pct"] = float(tr.get("winningPercentage",0.5))
        stats["src"] = "MLB Stats API"
    except Exception: pass
    return stats


# ══════════════════════════════════════════════════════════════════
#  9. FOREBET + SOFASCORE
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_forebet():
    predictions = {}
    try:
        r = requests.get(FOREBET_URL, headers=BROWSER_HDR, timeout=12)
        if r.status_code != 200: return predictions
        soup = BeautifulSoup(r.text,"html.parser")
        for row in soup.select(".rcnt,.tr_0,.tr_1"):
            try:
                teams = row.select(".tright.mutual_link a,.tnm a")
                if len(teams)<2: continue
                home=teams[0].text.strip().lower(); away=teams[1].text.strip().lower()
                probs=row.select(".fprc span")
                if len(probs)>=3:
                    hw=float(probs[0].text.strip().replace("%",""))/100
                    dw=float(probs[1].text.strip().replace("%",""))/100
                    aw=float(probs[2].text.strip().replace("%",""))/100
                    predictions[f"{home}|{away}"]={"home":hw,"draw":dw,"away":aw}
            except Exception: continue
    except Exception: pass
    return predictions


def find_forebet(home, away, predictions):
    for key,val in predictions.items():
        parts=key.split("|")
        if len(parts)!=2: continue
        if fuzzy_match(home,parts[0]) and fuzzy_match(away,parts[1]): return val
    return None


@st.cache_data(ttl=3600)
def fetch_sofascore(sport, date_str):
    try:
        r = requests.get(f"{SOFA_BASE}/sport/{sport}/scheduled-events/{date_str}",
                         headers={**BROWSER_HDR,"Referer":"https://www.sofascore.com/"},
                         timeout=10)
        if r.status_code==200: return r.json().get("events",[])
    except Exception: pass
    return []


def sofa_team_form(team_id):
    try:
        r = requests.get(f"{SOFA_BASE}/team/{team_id}/events/last/0",
                         headers={**BROWSER_HDR,"Referer":"https://www.sofascore.com/"},
                         timeout=8)
        if r.status_code!=200: return None
        events=r.json().get("events",[])[:5]; wins=0
        for ev in events:
            wc=ev.get("winnerCode"); hid=ev.get("homeTeam",{}).get("id")
            if wc==1 and hid==team_id: wins+=1
            elif wc==2 and hid!=team_id: wins+=1
        return wins/len(events) if events else None
    except Exception: return None


def get_sofa_signal(home, away, events):
    for ev in events:
        fh=ev.get("homeTeam",{}).get("name",""); fa=ev.get("awayTeam",{}).get("name","")
        if fuzzy_match(home,fh) and fuzzy_match(away,fa):
            hid=ev.get("homeTeam",{}).get("id"); aid=ev.get("awayTeam",{}).get("id")
            hf=sofa_team_form(hid) if hid else None
            time.sleep(0.3)
            af=sofa_team_form(aid) if aid else None
            if hf is not None and af is not None: return {"home_form":hf,"away_form":af}
    return None


# ══════════════════════════════════════════════════════════════════
#  10. MODELOS ESTADÍSTICOS
# ══════════════════════════════════════════════════════════════════

def _pmf(lam, k):
    if lam<=0: return 1.0 if k==0 else 0.0
    return math.exp(-lam)*(lam**k)/math.factorial(k)


def poisson_h2h_probs(lam_h, lam_a, max_g=8):
    hw=draw=aw=0.0
    for h in range(max_g+1):
        for a in range(max_g+1):
            p=_pmf(lam_h,h)*_pmf(lam_a,a)
            if h>a: hw+=p
            elif h==a: draw+=p
            else: aw+=p
    return hw,draw,aw


def poisson_line_prob(lam_h, lam_a, line):
    lam_t=lam_h+lam_a; floor=int(line)
    menos=sum(_pmf(lam_t,k) for k in range(floor+1))
    return 1.0-menos, menos


def pythagorean(pts_for, pts_ag, exp):
    """Fórmula Pythagorean genérica."""
    if pts_for+pts_ag==0: return 0.5
    return (pts_for**exp)/((pts_for**exp)+(pts_ag**exp))


# ══════════════════════════════════════════════════════════════════
#  11. COMBINACIÓN DE SEÑALES POR DEPORTE
# ══════════════════════════════════════════════════════════════════

def combine_football(home, away, apif_sig, forebet, sofa_sig, hist_home, hist_away, odds_move):
    """
    Fútbol — Poisson + API-Football + Forebet + Sofascore
    """
    AVG=1.30
    hf = apif_sig.get("home_form") or hist_home
    af = apif_sig.get("away_form") or hist_away

    lam_h = max(min((hf.get("goals_scored_pg",AVG)*af.get("goals_conceded_pg",AVG))/AVG*1.08,4.5),0.3)
    lam_a = max(min((af.get("goals_scored_pg",AVG)*hf.get("goals_conceded_pg",AVG))/AVG,4.5),0.3)

    h2h = apif_sig.get("h2h")
    if h2h:
        adj=(h2h["win_pct"]-0.5)*0.10; lam_h*=(1+adj); lam_a*=(1-adj)

    hi=apif_sig.get("home_injuries",0); ai=apif_sig.get("away_injuries",0)
    if hi>0: lam_h*=max(1-hi*0.03,0.75)
    if ai>0: lam_a*=max(1-ai*0.03,0.75)

    lam_h=max(min(lam_h,4.5),0.3); lam_a=max(min(lam_a,4.5),0.3)
    hw,draw,aw=poisson_h2h_probs(lam_h,lam_a)
    base={"home":hw,"draw":draw,"away":aw}

    pred=apif_sig.get("pred")
    l1={"home":pred["home_pct"],"draw":pred["draw_pct"],"away":pred["away_pct"]} if pred else base

    l2_h=l2_d=l2_a=None
    if forebet: l2_h,l2_d,l2_a=forebet.get("home"),forebet.get("draw",0),forebet.get("away")
    if sofa_sig:
        sf_tot=sofa_sig["home_form"]+sofa_sig["away_form"]
        sf_h=sofa_sig["home_form"]/sf_tot if sf_tot>0 else 0.5
        sf_a=1.0-sf_h
        l2_h=(l2_h+sf_h)/2 if l2_h else sf_h
        l2_a=(l2_a+sf_a)/2 if l2_a else sf_a

    has_l1=pred is not None; has_l2=l2_h is not None
    om=odds_move or {"home":0.0,"away":0.0}

    if has_l1 and has_l2:   w1,w2,w3,w4=0.20,0.40,0.30,0.10
    elif has_l1:             w1,w2,w3,w4=0.25,0.55,0.00,0.10  # sin Forebet/Sofa -> más peso API-F
    elif has_l2:             w1,w2,w3,w4=0.35,0.00,0.55,0.10
    else:                    w1,w2,w3,w4=0.90,0.00,0.00,0.10

    def w(b,l1v,l2v,om_adj):
        return b*w1+(l1v or b)*w2+(l2v or b)*w3+max(min(b+om_adj,0.95),0.05)*w4

    ch=w(base["home"],l1["home"],l2_h,om["home"])
    ca=w(base["away"],l1["away"],l2_a,om["away"])
    cd=w(base["draw"],l1["draw"],l2_d,0)
    tot=ch+ca+cd or 1.0

    return {
        "home":ch/tot,"away":ca/tot,"draw":cd/tot,
        "lam_h":lam_h,"lam_a":lam_a,
        "has_l1":has_l1,"has_l2":has_l2,
        "has_l3":bool(om["home"]!=0 or om["away"]!=0),
        "src":"API-Football + Forebet + Sofascore",
    }


def combine_basketball(home, away, apib_sig, sofa_sig, hist_home, hist_away, odds_move):
    """
    Basketball — Pythagorean (exp 13.91) + API-Basketball + Sofascore
    """
    # Mejor fuente disponible para stats
    hs = apib_sig.get("home_form") or apib_sig.get("home_stats") or hist_home
    as_ = apib_sig.get("away_form") or apib_sig.get("away_stats") or hist_away

    pts_h_for = hs.get("pts_for_pg", hs.get("goals_scored_pg", 112.0))
    pts_h_ag  = hs.get("pts_ag_pg",  hs.get("goals_conceded_pg", 112.0))
    pts_a_for = as_.get("pts_for_pg", as_.get("goals_scored_pg", 110.0))
    pts_a_ag  = as_.get("pts_ag_pg",  as_.get("goals_conceded_pg", 110.0))

    # Pythagorean basketball (exponente Morey: 13.91)
    hw = max(min(pythagorean(pts_h_for, pts_h_ag, 13.91)+0.03, 0.85), 0.15)
    aw = 1.0-hw

    # Ajuste H2H
    h2h=apib_sig.get("h2h")
    if h2h:
        adj=(h2h["win_pct"]-0.5)*0.08; hw=max(min(hw+adj,0.85),0.15); aw=1.0-hw

    # Señal Sofascore
    l2_h=l2_a=None
    if sofa_sig:
        sf_tot=sofa_sig["home_form"]+sofa_sig["away_form"]
        l2_h=sofa_sig["home_form"]/sf_tot if sf_tot>0 else 0.5
        l2_a=1.0-l2_h

    has_apib=bool(apib_sig.get("home_stats") or apib_sig.get("home_form"))
    has_l2=l2_h is not None
    om=odds_move or {"home":0.0,"away":0.0}

    if has_apib and has_l2:   w1,w2,w3=0.40,0.30,0.10; w_hist=0.20
    elif has_apib:             w1,w2,w3=0.55,0.00,0.10; w_hist=0.35
    elif has_l2:               w1,w2,w3=0.00,0.45,0.10; w_hist=0.45
    else:                      w1,w2,w3=0.00,0.00,0.10; w_hist=0.90

    ch=(hw*w_hist+(apib_sig.get("home_stats",{}).get("win_pct",hw) if apib_sig else hw)*w1
        +(l2_h or hw)*w2+max(min(hw+om["home"],0.95),0.05)*w3)
    ca=(aw*w_hist+(apib_sig.get("away_stats",{}).get("win_pct",aw) if apib_sig else aw)*w1
        +(l2_a or aw)*w2+max(min(aw+om["away"],0.95),0.05)*w3)

    tot=ch+ca or 1.0
    exp_total=round((pts_h_for+pts_a_for)/2,1)

    return {
        "home":ch/tot,"away":ca/tot,"draw":0.0,
        "exp_total":exp_total,
        "pts_h_for":round(pts_h_for,1),"pts_h_ag":round(pts_h_ag,1),
        "has_l1":has_apib,"has_l2":has_l2,
        "has_l3":bool(om["home"]!=0 or om["away"]!=0),
        "src":"API-Basketball + Sofascore",
    }


def combine_baseball(home, away, apibb_sig, mlb_home, mlb_away, sofa_sig, odds_move):
    """
    Béisbol — Pythagorean (exp 1.83) + API-Baseball + MLB Stats API + Sofascore
    """
    # Combina API-Baseball con MLB Stats API (promedio ponderado si ambos disponibles)
    apib_hs=apibb_sig.get("home_form") or apibb_sig.get("home_stats") or {}
    apib_as=apibb_sig.get("away_form") or apibb_sig.get("away_stats") or {}

    def merge_stat(key, apib_val, mlb_val, default):
        v1=apib_val.get(key); v2=mlb_val.get(key)
        if v1 and v2: return (v1+v2)/2
        return v1 or v2 or default

    rs_h=max(min(merge_stat("runs_scored_pg", apib_hs, mlb_home, 4.5),9.0),2.0)
    ra_h=max(min(merge_stat("runs_allowed_pg",apib_hs, mlb_home, 4.5),9.0),2.0)
    rs_a=max(min(merge_stat("runs_scored_pg", apib_as, mlb_away, 4.5),9.0),2.0)
    ra_a=max(min(merge_stat("runs_allowed_pg",apib_as, mlb_away, 4.5),9.0),2.0)

    hw=max(min(pythagorean(rs_h,ra_h,1.83)+0.04,0.85),0.15)
    aw=1.0-hw

    # Ajuste H2H
    h2h=apibb_sig.get("h2h")
    if h2h:
        adj=(h2h["win_pct"]-0.5)*0.08; hw=max(min(hw+adj,0.85),0.15); aw=1.0-hw

    # Sofascore
    l2_h=l2_a=None
    if sofa_sig:
        sf_tot=sofa_sig["home_form"]+sofa_sig["away_form"]
        l2_h=sofa_sig["home_form"]/sf_tot if sf_tot>0 else 0.5
        l2_a=1.0-l2_h

    has_apibb=bool(apib_hs); has_mlb=bool(mlb_home); has_l2=l2_h is not None
    om=odds_move or {"home":0.0,"away":0.0}

    if has_apibb and has_mlb and has_l2: w_base,w_l2,w_om=0.70,0.20,0.10
    elif (has_apibb or has_mlb) and has_l2: w_base,w_l2,w_om=0.70,0.20,0.10
    elif has_apibb or has_mlb: w_base,w_l2,w_om=0.90,0.00,0.10
    else: w_base,w_l2,w_om=0.90,0.00,0.10

    ch=hw*w_base+(l2_h or hw)*w_l2+max(min(hw+om["home"],0.95),0.05)*w_om
    ca=aw*w_base+(l2_a or aw)*w_l2+max(min(aw+om["away"],0.95),0.05)*w_om
    tot=ch+ca or 1.0
    exp_total=round((rs_h+ra_a)/2,2)

    return {
        "home":ch/tot,"away":ca/tot,"draw":0.0,
        "exp_total":exp_total,
        "rs_home":round(rs_h,2),"ra_home":round(ra_h,2),
        "has_l1":has_apibb or has_mlb,"has_l2":has_l2,
        "has_l3":bool(om["home"]!=0 or om["away"]!=0),
        "src":"API-Baseball + MLB Stats API",
    }


def combine_general(home, away, hist_home, hist_away, sofa_sig, odds_move, has_draw):
    """General para tenis — win% + Sofascore."""
    hw=min(hist_home.get("win_pct",0.52)+0.05,0.85)
    aw=hist_away.get("win_pct",0.48)
    if sofa_sig:
        sf_tot=sofa_sig["home_form"]+sofa_sig["away_form"]
        sf_h=sofa_sig["home_form"]/sf_tot if sf_tot>0 else 0.5
        hw=(hw+sf_h)/2; aw=1.0-hw

    om=odds_move or {"home":0.0,"away":0.0}
    hw=max(min(hw+om["home"]*0.5,0.85),0.15)
    aw=max(min(aw+om["away"]*0.5,0.85),0.15)

    if has_draw:
        dw=(hw+aw)*0.28; tot=hw+aw+dw
        r={"home":hw/tot,"away":aw/tot,"draw":dw/tot}
    else:
        tot=hw+aw; r={"home":hw/tot,"away":aw/tot,"draw":0.0}

    r.update({"has_l1":False,"has_l2":sofa_sig is not None,
              "has_l3":bool(om["home"]!=0 or om["away"]!=0),"src":"Sofascore + Histórico"})
    return r


# ══════════════════════════════════════════════════════════════════
#  12. PICKS
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
#  12. PICKS
# ══════════════════════════════════════════════════════════════════

def best_h2h_pick(probs, odds, home, away, has_draw):
    """Retorna el pick con mayor edge. Usa nombre real del equipo."""
    cands = [
        (home, probs["home"], odds["home"]),
        (away, probs["away"], odds["away"]),
    ]
    if has_draw and odds["draw"] > 1:
        cands.append(("Empate", probs.get("draw", 0), odds["draw"]))

    ranked = [
        {"label": l, "prob": p, "odds": o, "edge": edge(p, o)}
        for l, p, o in cands if o > 1
    ]
    ranked.sort(key=lambda x: x["edge"], reverse=True)
    return ranked[0] if ranked else None


def filter_mm_picks(probs, all_totals, model, unit):
    """Solo las líneas que el modelo considera probables (>= 60%) con edge positivo."""
    results = []
    lam_h = probs.get("lam_h")
    lam_a = probs.get("lam_a")
    exp_t = probs.get("exp_total")

    for t in all_totals:
        line = t["line"]

        if model == "poisson" and lam_h and lam_a:
            mas_p, menos_p = poisson_line_prob(lam_h, lam_a, line)
        elif exp_t:
            sigma   = 2.8 if "base" in model else max(exp_t * 0.06, 2.0)
            nd      = NormalDist(exp_t, sigma)
            menos_p = nd.cdf(line)
            mas_p   = 1.0 - menos_p
        else:
            continue

        if mas_p >= MM_PROB_THRESHOLD and t["over_odds"] > 1:
            e = edge(mas_p, t["over_odds"])
            if e > 0:
                results.append({
                    "label": f"Más de {line} {unit}",
                    "prob":  mas_p,
                    "odds":  t["over_odds"],
                    "edge":  e,
                })

        if menos_p >= MM_PROB_THRESHOLD and t["under_odds"] > 1:
            e = edge(menos_p, t["under_odds"])
            if e > 0:
                results.append({
                    "label": f"Menos de {line} {unit}",
                    "prob":  menos_p,
                    "odds":  t["under_odds"],
                    "edge":  e,
                })

    return sorted(results, key=lambda x: x["edge"], reverse=True)


# ══════════════════════════════════════════════════════════════════
#  13. UI — SIMPLE Y DIRECTO
# ══════════════════════════════════════════════════════════════════

def render_match_card(icon, home, away, gtime, h2h_pick, mm_picks):
    """
    Tarjeta limpia por partido:
    - Hora y equipos
    - Ganador con % de probabilidad
    - Líneas Más/Menos posibles
    """
    # Color de fondo según nivel de valor del pick principal
    if h2h_pick and h2h_pick["edge"] > 3:
        card_border = "#22c55e"
    elif h2h_pick and h2h_pick["edge"] > 0:
        card_border = "#eab308"
    else:
        card_border = "#e2e8f0"

    # Armar contenido de Más/Menos
    mm_html = ""
    for mp in mm_picks:
        mm_html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
             padding:6px 0;border-top:1px solid #f1f5f9">
          <span style="font-size:0.85rem;color:#475569">{mp['label']}</span>
          <span style="font-size:0.85rem;font-weight:700;color:#1649a0">
            {mp['prob']*100:.0f}%
          </span>
        </div>"""

    if not mm_picks:
        mm_html = '<div style="font-size:0.8rem;color:#94a3b8;padding-top:6px">Sin líneas Más/Menos con valor suficiente</div>'

    # Pick ganador
    if h2h_pick:
        winner_html = f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin:10px 0">
          <div>
            <div style="font-size:0.65rem;font-weight:700;color:#64748b;
                 text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px">
                 {"✅ GANA" if h2h_pick['edge'] > 3 else "⚠️ POSIBLE GANADOR"}
            </div>
            <div style="font-size:1.15rem;font-weight:800;color:#0a0f1e">
              {h2h_pick['label']}
            </div>
          </div>
          <div style="text-align:right">
            <div style="font-size:2rem;font-weight:800;color:#1649a0;line-height:1">
              {h2h_pick['prob']*100:.0f}%
            </div>
            <div style="font-size:0.72rem;color:#94a3b8;font-family:monospace">
              Cuota {h2h_pick['odds']:.2f} · Edge {h2h_pick['edge']:+.1f}%
            </div>
          </div>
        </div>"""
    else:
        winner_html = '<div style="font-size:0.85rem;color:#94a3b8;margin:10px 0">Sin valor en ganador</div>'

    st.markdown(f"""
<div style="background:white;border-radius:14px;padding:18px 22px;
     margin-bottom:14px;border-left:4px solid {card_border};
     box-shadow:0 1px 4px rgba(0,0,0,0.06)">

  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <div style="font-size:1rem;font-weight:700;color:#0a0f1e">
      {icon} {home} vs {away}
    </div>
    <div style="font-size:0.75rem;color:#94a3b8;font-family:monospace">⏰ {gtime}</div>
  </div>

  {winner_html}

  <div style="margin-top:8px">
    <div style="font-size:0.65rem;font-weight:700;color:#64748b;
         text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">
      Más / Menos
    </div>
    {mm_html}
  </div>

</div>""", unsafe_allow_html=True)


def render_combinada(value_picks):
    """Combinada limpia: un pick por línea, cuota total grande al final."""
    if not value_picks:
        st.markdown("""
<div style="background:#f8fafc;border-radius:12px;padding:16px 20px;
     color:#94a3b8;text-align:center;font-size:0.9rem">
  Sin picks con valor suficiente para armar combinada hoy
</div>""", unsafe_allow_html=True)
        return

    cuota = math.prod(p["odds"] for p in value_picks)
    prob  = math.prod(p["prob"]  for p in value_picks)

    rows = ""
    for p in value_picks:
        rows += f"""
<div style="display:flex;justify-content:space-between;align-items:center;
     padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.1)">
  <div>
    <div style="font-size:0.75rem;color:#93c5fd">{p['match']}</div>
    <div style="font-size:0.92rem;font-weight:700;color:white">{p['label']}</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:0.92rem;font-weight:700;color:white">@ {p['odds']:.2f}</div>
    <div style="font-size:0.72rem;color:#93c5fd">{p['prob']*100:.0f}%</div>
  </div>
</div>"""

    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d2247,#1649a0);
     border-radius:16px;padding:22px 26px;color:white">

  <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.1em;
       text-transform:uppercase;color:#93c5fd;margin-bottom:12px">
    🎯 Combinada del día — {len(value_picks)} selecciones
  </div>

  {rows}

  <div style="display:flex;gap:24px;margin-top:16px;flex-wrap:wrap">
    <div>
      <div style="font-size:0.65rem;color:#93c5fd;text-transform:uppercase;margin-bottom:4px">
        Cuota combinada
      </div>
      <div style="font-size:2.2rem;font-weight:800;letter-spacing:-0.02em">
        {cuota:.2f}
      </div>
    </div>
    <div>
      <div style="font-size:0.65rem;color:#93c5fd;text-transform:uppercase;margin-bottom:4px">
        Probabilidad estimada
      </div>
      <div style="font-size:2.2rem;font-weight:800;letter-spacing:-0.02em">
        {prob*100:.1f}%
      </div>
    </div>
  </div>

</div>""", unsafe_allow_html=True)


def render_backtest_stats(stats):
    if stats["total"] == 0:
        st.caption("Sin historial aún. Los picks de hoy se verificarán mañana.")
        return
    col  = "#22c55e" if stats["accuracy"] >= 55 else "#f59e0b" if stats["accuracy"] >= 45 else "#ef4444"
    pcol = "#22c55e" if stats["profit"] >= 0 else "#ef4444"
    st.markdown(f"""
<div style="background:#0f172a;border-radius:10px;padding:10px 12px;margin-bottom:8px;
     display:flex;gap:12px">
  <div style="flex:1;text-align:center">
    <div style="font-size:0.58rem;color:#64748b;text-transform:uppercase;font-weight:700">Acierto</div>
    <div style="font-size:1.3rem;font-weight:800;color:{col}">{stats['accuracy']}%</div>
    <div style="font-size:0.62rem;color:#64748b">{stats['correct']}/{stats['total']}</div>
  </div>
  <div style="flex:1;text-align:center">
    <div style="font-size:0.58rem;color:#64748b;text-transform:uppercase;font-weight:700">P&L</div>
    <div style="font-size:1.3rem;font-weight:800;color:{pcol}">{stats['profit']:+.2f}u</div>
    <div style="font-size:0.62rem;color:#64748b">por unidad</div>
  </div>
</div>""", unsafe_allow_html=True)
    for mkt, d in stats["by_market"].items():
        if d["total"] > 0:
            acc = d["correct"] / d["total"] * 100
            st.caption(f"{mkt}: {acc:.0f}% ({d['correct']}/{d['total']})")


# ══════════════════════════════════════════════════════════════════
#  14. STREAMLIT APP
# ══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="BetAnalyzer Pro", page_icon="🎯",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
* { font-family:'IBM Plex Sans',sans-serif !important; }
[data-testid="stAppViewContainer"] { background:#f0f4f8; }
[data-testid="stHeader"]           { background:transparent; }
[data-testid="stSidebar"]          { background:#0a0f1e; }
[data-testid="stSidebar"] *        { color:#c9d6e3 !important; }
[data-testid="stSidebar"] h2       { color:#e8edf5 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background:#151c2e; border:1px solid #2d3a50;
    color:#e8edf5 !important; border-radius:8px; }
.hero { background:linear-gradient(135deg,#0a0f1e 0%,#0d2247 50%,#1649a0 100%);
    border-radius:18px; padding:36px 44px; margin-bottom:24px; color:white; }
.hero h1 { margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-0.02em; }
.hero p  { margin:6px 0 0; color:#93c5fd; font-size:0.92rem; }
.hero-meta { margin-top:14px; color:#7dd3fc; font-size:0.76rem;
             font-family:'IBM Plex Mono',monospace; }
.sport-hdr { font-size:0.64rem; font-weight:700; letter-spacing:0.1em;
    text-transform:uppercase; color:#64748b; margin:24px 0 10px;
    padding-left:10px; border-left:3px solid #1649a0; }
</style>
""", unsafe_allow_html=True)

now_cl = datetime.now(CHILE_TZ)
today  = now_cl.strftime("%Y-%m-%d")

st.markdown(f"""
<div class="hero">
  <h1>🎯 LoyaltyBets</h1>
  <p>Análisis deportivo · Picks del día · Combinada</p>
  <div class="hero-meta">
    📅 {now_cl.strftime("%-d de %B, %Y")} &nbsp;|&nbsp;
    🕐 {now_cl.strftime("%H:%M")} (Chile) &nbsp;|&nbsp; Partidos del día
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    odds_key = st.text_input("🔑 Odds API Key", type="password",
                             placeholder="the-odds-api.com")
    apif_key = st.text_input("🔑 API-Sports Key", type="password",
                             placeholder="api-sports.io (gratis)")

    st.markdown("---")
    st.markdown("## 📊 Historial")
    if apif_key:
        updated = bt.verify_pending_picks(apif_key)
        if updated > 0:
            st.success(f"✅ {updated} picks verificados")
    stats = bt.get_stats()
    render_backtest_stats(stats)
    if stats["total"] > 0:
        with st.expander("Ver picks anteriores"):
            for p in stats.get("history", [])[:15]:
                icon = "✅" if p.get("correct") else "❌" if p.get("correct") is False else "⏳"
                st.caption(f"{icon} {p['date']} · {p['pick_label'][:22]} @ {p['pick_odds']} · {p.get('result','Pendiente')}")

    st.markdown("---")
    st.markdown("""
🟢 **Edge > 3%** → Valor real  
🟡 **Edge 0–3%** → Marginal  
🚫 **Edge < 0%** → No apostar  

Más/Menos: solo líneas con  
probabilidad ≥ 60% y edge positivo.
""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────
if not odds_key or not apif_key:
    st.info("👈 Ingresa ambas keys en el panel izquierdo.")
    st.stop()

if st.button("🔄 Analizar partidos de hoy", use_container_width=True):

    all_value_picks = []
    mlb_teams_cache = []

    with st.spinner("📡 Cargando Forebet..."):
        forebet_data = fetch_forebet()

    with st.spinner("📡 Fixtures API-Football..."):
        apif_fixtures = apif_fixtures_today(today, apif_key)

    for sport_name, cfg in SPORTS.items():

        with st.spinner(f"Obteniendo partidos {sport_name}..."):
            games = get_todays_games(odds_key, cfg["odds_keys"])

        if not games:
            continue

        with st.spinner(f"📡 Sofascore {sport_name}..."):
            sofa_events = fetch_sofascore(cfg["sofa_sport"], today)

        if cfg["model"] == "pythagorean_base" and not mlb_teams_cache:
            with st.spinner("Cargando equipos MLB..."):
                mlb_teams_cache = mlb_all_teams()

        st.markdown(f'<div class="sport-hdr">{sport_name} — {len(games)} partido(s)</div>',
                    unsafe_allow_html=True)

        bar = st.progress(0)

        for i, game in enumerate(games):
            bar.progress((i + 1) / len(games))

            home      = game["home_team"]
            away      = game["away_team"]
            gtime     = game["_time_cl"].strftime("%H:%M")
            h2h_o     = extract_h2h_odds(game)
            all_tots  = extract_all_totals(game)
            match_lbl = f"{home} vs {away}"

            if h2h_o["home"] <= 1 and h2h_o["away"] <= 1:
                continue

            # Movimiento de cuotas
            opening  = get_opening_odds(odds_key, game.get("_sport_key",""), game.get("id",""))
            odds_mov = odds_movement_signal(opening, h2h_o)
            sofa_sig = get_sofa_signal(home, away, sofa_events)

            # ── Señal + modelo por deporte ────────────────────────
            if cfg["model"] == "poisson":
                with st.spinner(f"🔬 Analizando {home} vs {away}..."):
                    apif_sig = build_football_signal(home, away, apif_fixtures, apif_key)
                forebet = find_forebet(home, away, forebet_data)
                probs   = combine_football(home, away, apif_sig, forebet, sofa_sig, {}, {}, odds_mov)
                fix_id  = apif_sig.get("fixture_id")

            elif cfg["model"] == "pythagorean_bball":
                with st.spinner(f"🔬 Analizando {home} vs {away}..."):
                    apib_sig = build_basketball_signal(home, away, apif_key)
                hist_h = {"win_pct":0.52,"pts_for_pg":112.0,"pts_ag_pg":112.0}
                hist_a = {"win_pct":0.48,"pts_for_pg":110.0,"pts_ag_pg":112.0}
                probs  = combine_basketball(home, away, apib_sig, sofa_sig, hist_h, hist_a, odds_mov)
                fix_id = None

            elif cfg["model"] == "pythagorean_base":
                with st.spinner(f"🔬 Analizando {home} vs {away}..."):
                    apibb_sig = build_baseball_signal(home, away, apif_key)
                mlb_h_id = mlb_find_id(home, mlb_teams_cache)
                mlb_a_id = mlb_find_id(away, mlb_teams_cache)
                mlb_h    = mlb_team_stats(mlb_h_id) if mlb_h_id else {}
                mlb_a    = mlb_team_stats(mlb_a_id) if mlb_a_id else {}
                probs    = combine_baseball(home, away, apibb_sig, mlb_h, mlb_a, sofa_sig, odds_mov)
                fix_id   = None

            else:
                probs  = combine_general(home, away, {"win_pct":0.52}, {"win_pct":0.48},
                                         sofa_sig, odds_mov, cfg["has_draw"])
                fix_id = None

            # ── Picks ────────────────────────────────────────────
            h2h_pick = best_h2h_pick(probs, h2h_o, home, away, cfg["has_draw"])
            mm_picks  = filter_mm_picks(probs, all_tots, cfg["model"], cfg["unit"])

            # Guardar backtesting
            if h2h_pick:
                bt.save_pick(match_lbl, sport_name, h2h_pick["label"],
                             h2h_pick["odds"], h2h_pick["prob"], h2h_pick["edge"],
                             fixture_id=fix_id, home=home, away=away)
            for mp in mm_picks:
                bt.save_pick(match_lbl, sport_name, mp["label"],
                             mp["odds"], mp["prob"], mp["edge"],
                             fixture_id=fix_id, home=home, away=away)

            # Agregar a combinada solo picks con edge > 3%
            if h2h_pick and h2h_pick["edge"] > 3:
                all_value_picks.append({**h2h_pick, "match": match_lbl})
            for mp in mm_picks:
                if mp["edge"] > 3:
                    all_value_picks.append({**mp, "match": match_lbl})

            # ── Render tarjeta limpia ─────────────────────────────
            render_match_card(cfg["icon"], home, away, gtime, h2h_pick, mm_picks)

        bar.progress(1.0)

    # ── Combinada del día ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🎯 Combinada del día")
    render_combinada(all_value_picks)
