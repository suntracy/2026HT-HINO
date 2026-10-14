# -*- coding: utf-8 -*-
"""
頭家副駕 AI / Boss Co-Driver AI - bilingual prototype (Streamlit)
run: streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="頭家副駕 AI / Boss Co-Driver AI", page_icon="🚚", layout="wide")

OIL_PRICE = 29.9
WORKDAYS = {300: 251, 500: 281, 700: 290}

# ---------- 語言 ----------
lang_choice = st.sidebar.radio("語言 / Language", ["繁體中文", "English"], index=0)
LANG = "zh" if lang_choice == "繁體中文" else "en"

T = {
    "zh": {
        "title": "🚚 頭家副駕 AI",
        "subtitle": "把50萬筆車聯網資料，變成頭家每週看得懂、做得到、算得出效益的建議",
        "demo_note": "｜示範資料：報告20輛表",
        "csv_note": "｜已接 hino_out",
        "tabs": ["① 車隊總覽", "② 健檢報告", "③ LINE週報", "④ 方法與限制"],
        "kpi1": "怠速年省潛力(20輛,10分上限,50%遵守)",
        "kpi2": "700系怠速/車/年",
        "kpi3": "排名修正(標準化後變動≥5名)",
        "kpi4": "疲勞風險(8hr後 vs 第1hr)",
        "filter": "車系篩選",
        "fair_title": "公平評分：原始 vs 標準化（急減速/100km）",
        "fair_info": "直接比原始次數會誤判一半車隊。原型一律套用系統預設門檻（急減速≥16km/h、超速≥11km/h）且只在同車系內比。門檻為推論假設，待主辦確認。",
        "select_car": "選車",
        "m_idle": "怠速 L/日",
        "m_cost": "怠速年油費",
        "m_next": "下一個平日怠速預測 (L)",
        "m_harsh": "標準化急減速",
        "m_fatigue": "疲勞 / DPF",
        "layers": ["描述", "診斷", "預測", "建議"],
        "ai_title": "🤖 AI 判斷",
        "fact_title": "📦 事實包 JSON",
        "line_title": "📱 本週 LINE 週報預覽",
        "qa_title": "💬 問答示範",
        "qa_ph": "頭家提問（試試：這週哪台該進廠？／1655是冷藏車）",
        "qa_shop": "建議優先3台：1.ABC-7610：DPF 17次+怠速最高上升中 2.ABC-7160：P2002 24次 3.ABC-6776：怠速10.9L/日+71%。故障碼確切意義請HINO服務廠確認。",
        "qa_other": "收到。我只會用事實包數字回答。請問是想問油費、疲勞排班、還是進廠？",
        "qa_confirm_title": "⚠️ 需要確認：此動作會改變車輛標籤",
        "qa_confirm_body": "你剛說「{q}」。\n\n我判斷你想把 **{truck}** 標記為**冷藏／特種車**，之後它的怠速不再列為浪費，會影響週報與省油計算。\n\n為防止誤觸或開玩笑亂改，請二次確認。確認後會寫入異動紀錄（誰、何時、改什麼），且可復原。",
        "qa_confirm_yes": "✅ 確認標記",
        "qa_confirm_no": "❌ 取消",
        "qa_confirmed": "了解，已將 {truck} 標記為冷藏車。之後怠速不再列為浪費，改追蹤冷凍機運轉與DPF。謝謝老闆！（已寫入異動紀錄，可復原）",
        "qa_cancelled": "已取消，沒有更動任何標籤。",
        "qa_undo": "↩️ 復原上次標記",
        "qa_undone": "已復原 {truck} 的標記。",
        "qa_log_title": "異動紀錄（稽核用，正式版應寫入DB）",
        "qa_current": "目前冷藏車標籤：",
        "qa_hint": "提醒：涉及改資料的指令一律要二次確認，不會一句話就直接寫入。",
        "model_title": "預測模型（1–7月訓練、8–11月檢驗）",
        "limit": "限制：僅24取樣日、年工作天為推估、門檻/遵守率/怠速-DPF為假設待確認、無駕駛身分只到車輛層級、疲勞僅10輛DMS車。如用模擬資料會標示。",
        "footer": "程式：hino_pipeline_zh.py → hino_out/ → 事實包JSON → LLM撰寫 → 數字比對防呆",
        "overview_bubble": "<b>頭家副駕 AI・車隊週報</b><br>老闆早安！怠速一年約 NT$414,585，4輛700系占大半。若停車超過10分鐘提醒熄火（一半照做），一年約可省 NT$133,925。<br>3輛列入DPF高度觀察：ABC-6776、ABC-7610、ABC-7160，建議安排進廠。<br><span style='font-size:12px;color:#666'>此為模擬資料：數字為取樣期間平均換算，非真實單週推送</span>",
        "fallback_desc": "門檻統一後比較；31%怠速集中10處多為場站",
        "fallback_pred": "怠速r=0.95穩定；惡化>50%即示警；疲勞AUC0.72",
        "fallback_rec": "停車>10分提醒熄火",
        "fallback_ai": "本週先追蹤怠速與超速，數字皆來自左側事實包。想請問：此車是否為特種車或需長時間等貨？",
        "col_map": {"carNum": "車號", "車系": "車系", "公里": "公里", "L/100km": "L/100km",
                    "怠速L/日": "怠速L/日", "怠速年油費": "怠速年油費", "標準化急減速/100km": "標準化急減速/100km",
                    "疲勞": "疲勞", "DPF": "DPF", "標籤": "標籤"},
    },
    "en": {
        "title": "🚚 Boss Co-Driver AI",
        "subtitle": "Turning 500k telematics records into one money-saving action per week for fleet owners",
        "demo_note": "｜Demo data: 20-vehicle table from report",
        "csv_note": "｜Connected to hino_out",
        "tabs": ["① Fleet overview", "② Truck Health Report", "③ LINE weekly report", "④ Methods & limits"],
        "kpi1": "Idling saving potential (20 trucks, 10-min cap, 50% compliance)",
        "kpi2": "700-series idling / truck / year",
        "kpi3": "Ranking corrected (shift ≥5 after normalization)",
        "kpi4": "Fatigue risk (after 8h vs 1st hour)",
        "filter": "Series filter",
        "fair_title": "Fair scoring: raw vs normalized (harsh braking /100km)",
        "fair_info": "Comparing raw counts misleads half the fleet. The prototype always applies system-default thresholds (harsh braking ≥16 km/h, speeding ≥11 km/h) and compares only within the same series. Thresholds are inferred assumptions pending organizer confirmation.",
        "select_car": "Select vehicle",
        "m_idle": "Idling L/day",
        "m_cost": "Idling fuel cost/year",
        "m_next": "Next-weekday idling fcast (L)",
        "m_harsh": "Norm. harsh braking",
        "m_fatigue": "Fatigue / DPF",
        "layers": ["Condition", "Diagnosis", "Predict", "Suggestion"],
        "ai_title": "🤖 AI in plain language",
        "fact_title": "📦 Fact pack JSON",
        "line_title": "📱 LINE weekly report preview",
        "qa_title": "💬 Q&A demo",
        "qa_ph": "Ask like a boss (try: Which truck needs service? / 1655 is a reefer)",
        "qa_shop": "Top 3 for service: 1. ABC-7610: 17 DPF codes + highest rising idling 2. ABC-7160: 24× P2002 3. ABC-6776: 10.9 L/day idling, +71%. Please confirm code meanings with a HINO service shop.",
        "qa_other": "Noted. I only answer with fact-pack numbers. Are you asking about fuel, fatigue rostering, or service?",
        "qa_confirm_title": "⚠️ Confirmation required: this changes a vehicle tag",
        "qa_confirm_body": "You said \"{q}\".\n\nI think you want to tag **{truck}** as a **reefer / special-purpose truck** — its idling will no longer count as waste, affecting weekly reports and saving estimates.\n\nTo prevent jokes or mistakes, please confirm a second time. Confirmed changes go to an audit log (who/when/what) and can be undone.",
        "qa_confirm_yes": "✅ Confirm tag",
        "qa_confirm_no": "❌ Cancel",
        "qa_confirmed": "Got it — {truck} is now tagged as a reefer truck. Its idling will no longer count as waste; we will track reefer-unit hours and DPF instead. Thanks, boss! (logged, undoable)",
        "qa_cancelled": "Cancelled — no tags were changed.",
        "qa_undo": "↩️ Undo last tag",
        "qa_undone": "Undone tag for {truck}.",
        "qa_log_title": "Audit log (prototype only; production should write to DB)",
        "qa_current": "Current reefer tags: ",
        "qa_hint": "Note: any write-action always needs a second confirmation — never applied from a single sentence.",
        "model_title": "Prediction models (train Jan–Jul, test Aug–Nov)",
        "limit": "Limits: only 24 sampled days; annual workdays estimated; thresholds/compliance/idling-DPF links are unconfirmed assumptions; no driver ID so vehicle-level only; fatigue covers 10 DMS trucks only. Simulated data will be labeled.",
        "footer": "Pipeline: hino_pipeline_zh.py → hino_out/ → fact-pack JSON → LLM draft → numeric cross-check",
        "overview_bubble": "<b>Boss Co-Driver AI・Fleet Weekly</b><br>Good morning, boss! Idling costs ~NT$414,585/year, mostly from four 700-series trucks. A 10-min switch-off reminder (50% compliance) saves ~NT$133,925/year.<br>3 trucks on DPF watch: ABC-6776, ABC-7610, ABC-7160 — schedule inspection.<br><span style='font-size:12px;color:#666'>Simulated data: figures averaged from sampled period, not a real weekly push</span>",
        "fallback_desc": "Compared with unified thresholds; 31% of idling at 10 yard/warehouse sites",
        "fallback_pred": "Idling stable (r=0.95); alert if +50% worse; fatigue AUC 0.72",
        "fallback_rec": "10-min switch-off reminder",
        "fallback_ai": "This week track idling and speeding; all numbers from the fact pack on the right. Question: is this a special-purpose truck or often waiting for cargo?",
        "col_map": {"carNum": "Truck", "車系": "Series", "公里": "Km", "L/100km": "L/100km",
                    "怠速L/日": "Idling L/day", "怠速年油費": "Idling NT$/yr", "標準化急減速/100km": "Norm. harsh/100km",
                    "疲勞": "Fatigue", "DPF": "DPF", "標籤": "Tags"},
    },
}
tr = T[LANG]

AI_TEXT = {
    "zh": {
        "ABC-7610": ("ABC-7610 每天怠速約 8.7 公升，一年約 NT$75,233，占這台車油耗的 24%。\n"
            "下半年怠速比上半年增加 74%，而且已出現 17 次 DPF 相關故障碼；長時間怠速可能讓 DPF 難以自行再生（此關聯尚待原廠確認）。\n"
            "本週建議：開啟「停車超過 10 分鐘提醒熄火」，預估一年可省約 NT$32,050，並安排 DPF 檢查。\n"
            "想請問：這台車停車時，是否需要開冷氣或使用車上設備？"),
        "ABC-7160": ("ABC-7160 的疲勞警示共 388 次，占全車隊的 57%，是目前最需要注意安全的車。\n"
            "這台車有 66% 的駕駛時間落在疲勞風險區（當班第3–4小時、值勤8小時後、凌晨1–5點），這些時段包含它 74% 的警示。\n"
            "本週建議：排班時讓這台車每連續駕駛 3 小時安排一次休息，並避免長時間夜班。\n"
            "想請問：這台車最近是否固定由同一位司機開長途或夜班？"),
        "ABC-1999": ("好消息：ABC-1999 的急減速其實不嚴重。原始紀錄每 100 公里 78.7 次（排名第 5），但這台車的門檻疑似設得較敏感；"
            "改用系統預設門檻後只有 0.75 次，排名降到第 16。\n"
            "真正要注意的是超速：用統一標準計算，每 100 公里約 25.6 次超速 11 km/h 以上，在車隊中偏高。\n"
            "本週建議：與司機溝通時肯定煞車習慣，把重點放在市區 40–50 km/h 路段的車速。\n"
            "想請問：這台車主要跑哪一區的配送路線？"),
    },
    "en": {
        "ABC-7610": ("ABC-7610 idles about 8.7 L per day — roughly NT$75,233 per year, 24% of this truck's fuel.\n"
            "Second-half idling rose 74% vs the first half, with 17 DPF-related fault codes; long idling may prevent DPF self-regeneration (link pending HINO confirmation).\n"
            "This week: enable the 10-minute switch-off reminder, saving ~NT$32,050/year, and schedule a DPF check.\n"
            "Question: does this truck need A/C or onboard equipment while parked?"),
        "ABC-7160": ("ABC-7160 has 388 fatigue alerts — 57% of the whole fleet, our top safety concern.\n"
            "66% of its driving falls in fatigue risk zones (3rd–4th hour on duty, after 8 hours, 1–5 a.m.), covering 74% of its alerts.\n"
            "This week: roster a break every 3 hours of continuous driving and avoid long night shifts.\n"
            "Question: is this truck recently driven long-haul/nights by the same driver?"),
        "ABC-1999": ("Good news: ABC-1999's braking is actually fine. Raw records show 78.7 harsh-braking events per 100 km (rank #5), but its threshold looks more sensitive; "
            "with the system-default threshold it drops to 0.75 (rank #16).\n"
            "The real issue is speeding: 25.6 events per 100 km at ≥11 km/h over the limit — high for the fleet.\n"
            "This week: praise the braking, focus coaching on 40–50 km/h urban segments.\n"
            "Question: which delivery zone does this truck mainly run?"),
    },
}

# ---------- 資料：真實 hino_out + models（失敗才用內建示範） ----------
# 報告值僅保留：怠速L/日、怠速占比、天數（vehicles.csv 沒有怠速油耗，需掃49萬列才有，Demo沿用已驗證報告值）
REPORT_IDLE = {
    "ABC-2713": (0.08,0.4,13), "ABC-6325": (0.21,0.9,14), "ABC-1999": (2.20,9.6,15),
    "ABC-5979": (0.07,0.7,21), "ABC-1926": (0.80,3.2,20), "ABC-2650": (0.07,0.3,14),
    "ABC-9818": (0.46,3.7,14), "ABC-1507": (0.00,0.0,12), "ABC-2037": (0.03,0.5,16),
    "ABC-7569": (0.07,0.5,23), "ABC-1868": (0.35,2.0,17), "ABC-7756": (0.75,2.4,18),
    "ABC-5887": (1.35,2.0,17), "ABC-5310": (4.02,11.8,23), "ABC-1655": (1.72,30.6,16),
    "ABC-7610": (8.68,23.9,17), "ABC-2389": (6.05,15.2,19), "ABC-7599": (3.82,10.0,19),
    "ABC-6776": (10.94,9.0,24), "ABC-7160": (6.93,4.9,22),
}

@st.cache_data
def load_events_stats():
    """從真實 events.csv 計算：標準化急減速(>=16)、標準化超速(over>=11,開始筆)、DPF(P2002/P244B)、疲勞。"""
    p = Path("hino_out/events.csv")
    if not p.exists():
        return None
    e = pd.read_csv(p, usecols=["carNum", "type", "phase", "info.speedDifference", "over_kmh", "info.dtcCodes"],
                    low_memory=False)
    g = e.groupby("carNum")
    out = {}
    for car, d in g:
        norm_harsh = ((d["type"] == 7) & (d["info.speedDifference"] >= 16)).sum()
        norm_speed = ((d["type"] == 8) & (d["phase"] == "start") & (d["over_kmh"] >= 11)).sum()
        dpf = ((d["type"] == 9) & (d["info.dtcCodes"].isin(["P2002", "P244B"]))).sum()
        fatigue = ((d["type"] == 12)).sum()
        out[car] = {"norm_harsh_n": int(norm_harsh), "norm_speed_n": int(norm_speed),
                    "dpf_n": int(dpf), "fatigue_n": int(fatigue)}
    return out

@st.cache_data
def load_results():
    p = Path("models/results.json")
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def load_vehicles():
    ev_stats = load_events_stats()
    results = load_results()
    p = Path("hino_out/vehicles.csv")
    if p.exists():
        try:
            v = pd.read_csv(p, low_memory=False)
            rows = []
            for _, r in v.iterrows():
                car = r["carNum"]
                idle_l, idle_pct, days = REPORT_IDLE.get(car, (0, 0, 0))
                dist = float(r.get("distance_km", 0) or 0)
                st_ev = (ev_stats or {}).get(car, {})
                norm_h = st_ev.get("norm_harsh_n", 0) / dist * 100 if dist >= 1 else 0
                norm_s = st_ev.get("norm_speed_n", 0) / dist * 100 if dist >= 1 else 0
                rows.append({
                    "carNum": car, "車系": int(r["Type"]), "天數": days,
                    "公里": round(dist, 1), "L/100km": round(float(r.get("L_per_100km", 0) or 0), 1),
                    "怠速L/日": idle_l, "怠速占%": idle_pct,
                    "原始急減速/100km": round(float(r.get("per100km_decelerate", 0) or 0), 2),
                    "標準化急減速/100km": round(float(norm_h), 2),
                    "標準化超速/100km": round(float(norm_s), 2),
                    "疲勞": int(st_ev.get("fatigue_n", r.get("n_fatigue", 0) or 0)),
                    "DPF": int(st_ev.get("dpf_n", 0)),
                    "n_trips": int(r.get("n_trips", 0) or 0),
                    "enabledCode": r["enabledCode"],
                })
            df = pd.DataFrame(rows)
            return df, True, results
        except Exception as ex:
            st.warning(f"讀取 hino_out 失敗，改用內建示範表：{ex}")
    rows = [
        ("ABC-2713",300,13,1787,15.1,0.08,0.4,0.6,0.62,3.7,0,0),
        ("ABC-6325",300,14,1854,18.0,0.21,0.9,0.6,0.65,7.4,0,0),
        ("ABC-1999",300,15,1334,25.8,2.20,9.6,78.7,0.75,25.6,0,0),
        ("ABC-5979",300,21,1665,13.4,0.07,0.7,83.1,0.84,12.6,0,0),
        ("ABC-1926",300,20,3291,15.0,0.80,3.2,54.9,1.15,6.0,0,0),
        ("ABC-2650",300,14,2169,13.9,0.07,0.3,2.6,2.58,5.8,0,0),
        ("ABC-9818",300,14,1262,14.1,0.46,3.7,161.1,9.35,9.8,0,0),
        ("ABC-1507",300,12,1976,14.7,0.00,0.0,125.3,9.36,24.8,0,0),
        ("ABC-2037",300,16,718,15.5,0.03,0.5,200.1,6.55,6.0,0,0),
        ("ABC-7569",300,23,1900,15.4,0.07,0.5,73.8,0.42,12.7,0,0),
        ("ABC-1868",500,17,1230,24.7,0.35,2.0,3.5,3.50,9.7,20,0),
        ("ABC-7756",500,18,2735,20.4,0.75,2.4,3.0,3.03,9.1,18,0),
        ("ABC-5887",500,17,5017,22.5,1.35,2.0,28.1,1.14,9.3,6,0),
        ("ABC-5310",500,23,3545,22.2,4.02,11.8,1.4,1.38,5.8,94,0),
        ("ABC-1655",500,16,174,51.7,1.72,30.6,0.0,0.00,1.7,21,0),
        ("ABC-7610",700,17,1195,51.5,8.68,23.9,1.7,1.67,2.0,13,17),
        ("ABC-2389",700,19,1632,46.5,6.05,15.2,76.0,1.04,14.3,41,0),
        ("ABC-7599",700,19,1978,36.7,3.82,10.0,44.3,1.62,20.2,16,6),
        ("ABC-6776",700,24,6344,46.0,10.94,9.0,46.1,1.40,16.2,69,1),
        ("ABC-7160",700,22,11154,28.1,6.93,4.9,2.3,2.30,3.7,388,24),
    ]
    df = pd.DataFrame(rows, columns=[
        "carNum","車系","天數","公里","L/100km","怠速L/日","怠速占%",
        "原始急減速/100km","標準化急減速/100km","標準化超速/100km","疲勞","DPF"])
    df["enabledCode"] = df["carNum"]
    df["n_trips"] = 0
    return df, False, {}

df, from_csv, RESULTS = load_vehicles()
df["年工作天"] = df["車系"].map(WORKDAYS)
df["怠速年油費"] = (df["怠速L/日"] * df["年工作天"] * OIL_PRICE).round(0)

def tags(r):
    if LANG == "zh":
        t = []
        if r["怠速年油費"] >= 30000: t.append("🔴 怠速偏高")
        if r["DPF"] > 0 and r["怠速L/日"] >= 5: t.append("⚠️ DPF高度觀察")
        if r["carNum"] in ("ABC-7610","ABC-6776"): t.append("📈 惡化中(+70%)")
        if r["疲勞"] >= 40: t.append("😴 疲勞注意")
        if r["標準化急減速/100km"] >= 5: t.append("🛑 急減速偏高")
        return " ".join(t) if t else "—"
    else:
        t = []
        if r["怠速年油費"] >= 30000: t.append("🔴 High idling")
        if r["DPF"] > 0 and r["怠速L/日"] >= 5: t.append("⚠️ DPF watch")
        if r["carNum"] in ("ABC-7610","ABC-6776"): t.append("📈 Worsening (+70%)")
        if r["疲勞"] >= 40: t.append("😴 Fatigue watch")
        if r["標準化急減速/100km"] >= 5: t.append("🛑 Harsh-braking high")
        return " ".join(t) if t else "—"

df["標籤"] = df.apply(tags, axis=1)

FACTS = {
    "ABC-7610": {"truck":"ABC-7610","series":700,
        "describe":{"idling_L_per_day":8.7,"idling_share_pct":23.9,"idling_cost_NTD_yr":75233},
        "diagnose":{"DPF_codes":17,"hypothesis":"long idling may block DPF regeneration (pending HINO)","H2_vs_H1_pct":74},
        "predict":{"DPF_watch":"high"},"recommend":{"cap_min":10,"saving_NTD_yr_50pct":32050}},
    "ABC-7160": {"truck":"ABC-7160","series":700,
        "describe":{"fatigue_alerts":388,"share_of_fleet_pct":57,"idling_L_per_day":6.9},
        "diagnose":{"risk_zone_drive_pct":66,"risk_zone_alert_pct":74},
        "predict":{"risk_6th_hour_per_15min_pct":66,"fatigue_AUC":0.72},
        "recommend":{"action":"break every 3h, avoid long nights"}},
    "ABC-1999": {"truck":"ABC-1999","series":300,
        "describe":{"raw_harsh_per100km":78.7,"raw_rank":5},
        "diagnose":{"norm_harsh_per100km":0.75,"new_rank":16,"cause_hypothesis":"sensitive threshold","norm_speeding_per100km":25.6},
        "predict":{},"recommend":{"action":"praise braking, coach 40-50 km/h urban speeding"}},
}

# ---------- UI ----------
st.title(tr["title"])
if from_csv:
    src_note = tr["csv_note"] + ("｜events.csv實算+results.json" if LANG == "zh" else "｜events.csv + results.json")
else:
    src_note = tr["demo_note"]
st.caption(tr["subtitle"] + src_note)
if from_csv:
    st.caption("⚠️ 怠速L/日沿用報告值（重算需掃49萬列）；急減速/超速標準化、DPF、疲勞為events.csv實算；預測分數來自models/results.json。"
               if LANG == "zh" else "⚠️ Idling L/day reuses report values; normalized harsh/speeding, DPF, fatigue computed live from events.csv; scores from models/results.json.")
tab1, tab2, tab3, tab4 = st.tabs(tr["tabs"])
st.markdown("""<style>
.bubble{word-break:break-word;white-space:normal}
.ai-big{font-size:19px;line-height:2.0;background:#f1f8e9;border-radius:12px;padding:18px 20px;word-break:break-word}
div[data-testid="stAlert"]{white-space:pre-wrap;word-break:break-word}
div[data-testid="stAlert"] p{white-space:pre-wrap;word-break:break-word}
</style>""", unsafe_allow_html=True)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr["kpi1"], "NT$134,000")
    c2.metric(tr["kpi2"], "NT$64,389", "7.4 L/day" if LANG=="en" else "7.4 L/日")
    c3.metric(tr["kpi3"], "10 / 20")
    c4.metric(tr["kpi4"], "x2.2")
    series = st.multiselect(tr["filter"], [300, 500, 700], default=[300, 500, 700])
    show = df[df["車系"].isin(series)].sort_values("怠速年油費", ascending=False)
    cols = ["carNum","車系","公里","L/100km","怠速L/日","怠速年油費","標準化急減速/100km","疲勞","DPF","標籤"]
    show_disp = show[cols].rename(columns=tr["col_map"])
    st.dataframe(show_disp, use_container_width=True, hide_index=True)
    st.subheader(tr["fair_title"])
    comp = show.set_index("carNum")[["原始急減速/100km","標準化急減速/100km"]].head(10)
    if LANG == "en":
        comp.columns = ["raw /100km", "normalized /100km"]
    _long = comp.reset_index().melt("carNum", var_name="item", value_name="v")
    _ch = alt.Chart(_long).mark_bar().encode(
        x=alt.X("carNum:N", axis=alt.Axis(labelAngle=-45, title=None)),
        y=alt.Y("v:Q", title="/100km"),
        color=alt.Color("item:N", title=""),
        xOffset="item:N",
        tooltip=["carNum", "item", "v"],
    ).properties(height=340)
    st.altair_chart(_ch, use_container_width=True)
    st.info(tr["fair_info"])

with tab2:
    car = st.selectbox(tr["select_car"], df["carNum"].tolist(),
                       index=df["carNum"].tolist().index("ABC-7610"))
    r = df[df["carNum"] == car].iloc[0]
    st.write(f"### {car} ｜ Series {r['車系'] if LANG=='en' else str(r['車系'])+'系'} ｜ {r['標籤']}")
    _iz = (RESULTS.get("idle", {}).get("下一個平日怠速預測_公升", {}) or {}).get(car)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(tr["m_idle"], r["怠速L/日"]); k2.metric(tr["m_cost"], f"NT${r['怠速年油費']:,.0f}")
    k3.metric("⛽ " + tr["m_next"], _iz if _iz is not None else "—")
    k4.metric(tr["m_harsh"], r["標準化急減速/100km"]); k5.metric(tr["m_fatigue"], f"{r['疲勞']} / {r['DPF']}")
    # 疲勞三情境風險：獨立成區（僅 DMS 車有）
    _fz = (RESULTS.get("fatigue", {}).get("各車", {}) or {}).get(car)
    if _fz:
        st.subheader("😴 " + ("疲勞三情境風險" if LANG == "zh" else "Fatigue risk by scenario"))
        _keys = [k for k in _fz.keys()]
        _short = {"zh": ["白天第1hr", "下午第6hr", "凌晨第9hr", "平均"],
                  "en": ["9am 1st hr", "2pm 6th hr", "3am 9th hr", "Avg"]}[LANG]
        _cols = st.columns(4)
        for _c, _k, _s in zip(_cols, _keys, _short):
            _c.metric(_s, f"{_fz[_k]:.0%}")
    else:
        st.caption("😴 " + ("此車無 DMS，無疲勞模型分數" if LANG == "zh" else "No DMS on this truck; no fatigue scores"))
    # 高負載：獨立成區
    _oz = (RESULTS.get("overload", {}).get("各車", {}) or {}).get(car)
    if _oz:
        st.subheader("🔧 " + ("高負載（自算規則，全車一致）" if LANG == "zh" else "High load (uniform rule)"))
        st.table(pd.DataFrame([_oz], hide_index=True))
    # AI 判斷放最大、最上面：主管只看這塊
    st.subheader(tr["ai_title"])
    _ai = AI_TEXT[LANG].get(car, tr["fallback_ai"])
    st.markdown(f'<div class="ai-big">{_ai.replace(chr(10), "<br><br>")}</div>', unsafe_allow_html=True)
    fact = FACTS.get(car, {"truck": car, "note": "table values only" if LANG=="en" else "此車沿用表格數字"})

    # 沒手寫事實包的車：用真實列＋results.json 自動組文字（不再是通用句）
    _fz2 = (RESULTS.get("fatigue", {}).get("各車", {}) or {}).get(car)
    _iz2 = (RESULTS.get("idle", {}).get("下一個平日怠速預測_公升", {}) or {}).get(car)
    _oz2 = (RESULTS.get("overload", {}).get("各車", {}) or {}).get(car)
    _cost = f"NT${r['怠速年油費']:,.0f}"
    if LANG == "zh":
        _auto = {
            "describe": (f"{car}為{r['車系']}系車，取樣期間行駛{r['公里']}公里（{r['n_trips']}趟），"
                         f"平均油耗{r['L/100km']} L/100km。每天怠速約{r['怠速L/日']}公升（占油耗{r['怠速占%']}%），"
                         f"一年怠速油費約{_cost}。標準化急減速{r['標準化急減速/100km']}/100km、"
                         f"超速{r['標準化超速/100km']}/100km，疲勞{r['疲勞']}次、DPF故障{r['DPF']}次。"),
            "diagnose": (f"評分採統一門檻（急減速≥16 km/h、超速≥11 km/h）且只在{r['車系']}系內比較；"
                         f"門檻為推論假設，待主辦確認。"
                         + ("怠速占油耗逾一成，偏高，建議查場站停留。" if r["怠速占%"] >= 10 else "")
                         + ("已有DPF故障碼，優先檢查。" if r["DPF"] > 0 else "")),
            "predict": ("怠速習慣穩定（全車隊上下半年相關 r=0.95）；"
                        + (f"下一個平日怠速預測{_iz2}公升；" if _iz2 is not None else "")
                        + (f"每百公里高負載片段{_oz2.get('每百公里高負載片段')}次；" if _oz2 else "")
                        + (f"疲勞平均風險{_fz2.get('平均風險', 0):.0%}。" if _fz2 else "")),
            "recommend": ("本週一件事：停車超過10分鐘提醒熄火。"
                          + ("安排DPF檢查。" if r["DPF"] > 0 else "")
                          + "想請問：是否為特種車或需長時間等貨？"),
        }
    else:
        _auto = {
            "describe": (f"{car} (Series {r['車系']}): {r['公里']} km over {r['n_trips']} trips, "
                         f"{r['L/100km']} L/100km. Idling {r['怠速L/日']} L/day ({r['怠速占%']}% of fuel), "
                         f"~{_cost}/yr. Norm. harsh {r['標準化急減速/100km']}/100km, speeding {r['標準化超速/100km']}/100km, "
                         f"fatigue {r['疲勞']}, DPF {r['DPF']}."),
            "diagnose": (f"Unified thresholds (harsh ≥16 km/h, speeding ≥11 km/h), Series {r['車系']} only; "
                         f"thresholds are inferred assumptions."
                         + (" Idling share ≥10%, check yard stays." if r["怠速占%"] >= 10 else "")
                         + (" DPF codes present, inspect first." if r["DPF"] > 0 else "")),
            "predict": ("Idling habit is stable (fleet H1–H2 r=0.95); "
                        + (f"next weekday idling forecast {_iz2} L; " if _iz2 is not None else "")
                        + (f"{_oz2.get('每百公里高負載片段')} high-load episodes/100km; " if _oz2 else "")
                        + (f"avg fatigue risk {_fz2.get('平均風險', 0):.0%}." if _fz2 else "")),
            "recommend": ("One action: 10-minute switch-off reminder."
                          + (" Schedule a DPF check." if r["DPF"] > 0 else "")
                          + " Question: special-purpose truck or long cargo waits?"),
        }

    _LAYER_ZH = {
        "truck": "車號", "series": "車系", "note": "備註",
        "idling_L_per_day": "每天怠速公升數", "idling_share_pct": "怠速占油耗(%)",
        "idling_cost_NTD_yr": "一年怠速油費(NT$)", "DPF_codes": "DPF故障碼次數",
        "hypothesis": "原因假設", "H2_vs_H1_pct": "下半年比上半年變化(%)",
        "DPF_watch": "DPF觀察等級", "cap_min": "怠速上限(分鐘)",
        "saving_NTD_yr_50pct": "一半照做時年省(NT$)",
        "fatigue_alerts": "疲勞警示次數", "share_of_fleet_pct": "占全車隊(%)",
        "risk_zone_drive_pct": "風險區駕駛時間(%)", "risk_zone_alert_pct": "風險區警示占比(%)",
        "risk_6th_hour_per_15min_pct": "第6小時每15分鐘警示機率(%)",
        "fatigue_AUC": "疲勞模型AUC", "action": "建議行動",
        "raw_harsh_per100km": "原始急減速(次/100km)", "raw_rank": "原始排名",
        "norm_harsh_per100km": "標準化急減速(次/100km)", "new_rank": "標準化後排名",
        "cause_hypothesis": "原因假設", "norm_speeding_per100km": "標準化超速(次/100km)",
    }
    _LAYER_EN = {
        "truck": "Truck", "series": "Series", "note": "Note",
        "idling_L_per_day": "Idling (L/day)", "idling_share_pct": "Idling share of fuel (%)",
        "idling_cost_NTD_yr": "Yearly idling cost (NT$)", "DPF_codes": "DPF fault count",
        "hypothesis": "Suspected cause", "H2_vs_H1_pct": "H2 vs H1 change (%)",
        "DPF_watch": "DPF watch level", "cap_min": "Idling cap (min)",
        "saving_NTD_yr_50pct": "Yearly saving at 50% compliance (NT$)",
        "fatigue_alerts": "Fatigue alerts", "share_of_fleet_pct": "Share of fleet (%)",
        "risk_zone_drive_pct": "Risk-zone drive time (%)", "risk_zone_alert_pct": "Risk-zone alert share (%)",
        "risk_6th_hour_per_15min_pct": "Alert chance per 15 min, 6th hour (%)",
        "fatigue_AUC": "Fatigue model AUC", "action": "Recommended action",
        "raw_harsh_per100km": "Raw harsh braking (/100km)", "raw_rank": "Raw rank",
        "norm_harsh_per100km": "Normalized harsh braking (/100km)", "new_rank": "Normalized rank",
        "cause_hypothesis": "Suspected cause", "norm_speeding_per100km": "Normalized speeding (/100km)",
    }
    _VAL_ZH = {
        "long idling may block DPF regeneration (pending HINO)": "長時間怠速使DPF難以再生（待原廠確認）",
        "sensitive threshold": "門檻疑似設得較敏感",
        "break every 3h, avoid long nights": "每連續駕駛3小時休息一次，避免長時間夜班",
        "praise braking, coach 40-50 km/h urban speeding": "肯定煞車習慣，重點放在市區40–50 km/h路段車速",
        "high": "高",
    }
    _LBL = _LAYER_ZH if LANG == "zh" else _LAYER_EN
    _INTRO = {
        "describe": "發生了什麼：這台車的現況數字" if LANG == "zh" else "What happened: this truck's current numbers",
        "diagnose": "為什麼會這樣：原因分析（門檻為推論假設）" if LANG == "zh" else "Why: diagnosis (thresholds are inferred)",
        "predict": "接下來會怎樣：模型預測與觀察名單" if LANG == "zh" else "What next: model forecasts and watchlist",
        "recommend": "該做什麼：本週只做一件事" if LANG == "zh" else "What to do: one action this week",
    }

    def _show_layer(d, fallback, auto_key, intro_key):
        # dict 轉中文條列，不秀 JSON；沒手寫包的車用自動文字
        st.caption("💡 " + _INTRO[intro_key])
        if isinstance(d, dict) and d:
            for k, v in d.items():
                _label = _LBL.get(str(k), str(k).replace("_", " "))
                if LANG == "zh" and isinstance(v, str):
                    v = _VAL_ZH.get(v, v)
                st.markdown(f"- **{_label}**：{v}")
        elif car not in FACTS:
            st.write(_auto[auto_key])
        elif isinstance(d, str) and d:
            st.write(d)
        else:
            st.write(fallback)

    with st.expander("📊 " + ("JSON 資料" if LANG == "zh"
                        else "JSON Information"), expanded=False):
        l1, l2, l3, l4 = st.tabs(tr["layers"])
        with l1: _show_layer(fact.get("describe"), tr["fallback_desc"], "describe", "describe")
        with l2: _show_layer(fact.get("diagnose"), tr["fallback_desc"], "diagnose", "diagnose")
        with l3:
            _show_layer(fact.get("predict"), tr["fallback_pred"], "predict", "predict")
            st.caption("👆 詳細模型分數見上方獨立區（疲勞三情境／怠速預測／高負載）" if LANG == "zh"
                       else "👆 See the standalone sections above for full model scores")
        with l4: _show_layer(fact.get("recommend"), tr["fallback_rec"], "recommend", "recommend")
    st.caption("🛡️ " + ("防呆：AI 只能用健檢表的數字寫建議，寫完程式會逐字比對" if LANG == "zh"
                else "Guardrail: the AI may use only numbers from the health-check sheet above; every number is verified afterwards"))

LINE_CSS = """<style>.bubble{background:#f1f8e9;border-radius:12px;padding:10px 14px;margin:8px 0;max-width:520px}
.me{background:#e3f2fd;margin-left:auto}</style>"""

with tab3:
    st.markdown(LINE_CSS, unsafe_allow_html=True)
    st.subheader(tr["line_title"])
    st.caption("以下皆為 AI 官方帳號的推播示意（此為模擬資料）" if LANG == "zh" else "All below are simulated pushes from the AI official account (simulated data)")
    _cells = ["__overview__", "ABC-7610", "ABC-7160", "ABC-1999"]
    for _i in range(0, len(_cells), 2):
        _row = st.columns(2)
        for _j, _col in enumerate(_row):
            if _i + _j < len(_cells):
                _k = _cells[_i + _j]
                with _col:
                    if _k == "__overview__":
                        st.markdown(f'<div class="bubble">{tr["overview_bubble"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="bubble"><b>Boss Co-Driver AI・{_k}</b><br>{AI_TEXT[LANG][_k].replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)
    st.divider()
    st.subheader(tr["qa_title"])
    st.caption(tr["qa_hint"])
    # --- 防亂改：二段確認狀態 ---
    import re as _re
    from datetime import datetime as _dt
    if "pending_reefer" not in st.session_state:
        st.session_state.pending_reefer = None
    if "reefer_tags" not in st.session_state:
        st.session_state.reefer_tags = set()
    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []

    def _find_truck(text):
        m = _re.search(r"ABC-\d+", text.upper())
        return m.group(0) if m else None

    # 若有待確認，先顯示確認卡（不看新輸入）
    if st.session_state.pending_reefer:
        pend = st.session_state.pending_reefer
        st.warning(tr["qa_confirm_title"])
        st.write(tr["qa_confirm_body"].format(q=pend["q"], truck=pend["truck"]))
        c_yes, c_no = st.columns(2)
        if c_yes.button(tr["qa_confirm_yes"], key="confirm_yes"):
            st.session_state.reefer_tags.add(pend["truck"])
            st.session_state.audit_log.append(
                f"{_dt.now().strftime('%m-%d %H:%M')}｜TAG {pend['truck']}=reefer｜src=chat \"{pend['q']}\"｜by=demo-user")
            st.session_state.pending_reefer = None
            st.rerun()
        if c_no.button(tr["qa_confirm_no"], key="confirm_no"):
            st.session_state.audit_log.append(
                f"{_dt.now().strftime('%m-%d %H:%M')}｜CANCEL TAG {pend['truck']}｜src=chat")
            st.session_state.pending_reefer = None
            st.rerun()
        st.stop()

    q = st.text_input(tr["qa_ph"], key="qa_input")
    if q:
        ql = q.lower()
        is_reefer = ("冷藏" in q or "冷凍" in q or "特種" in q or "reefer" in ql
                     or "fridge" in ql or "refrigerat" in ql)
        truck = _find_truck(q)
        if is_reefer:
            # 沒寫車號預設1655，但仍要確認，不能直接寫
            truck = truck or "ABC-1655"
            st.session_state.pending_reefer = {"truck": truck, "q": q}
            st.rerun()
        elif "進廠" in q or "service" in ql or "repair" in ql or "factory" in ql:
            st.info(tr["qa_shop"])
        else:
            st.info(tr["qa_other"])

    # 目前標籤 + 稽核 + 復原
    st.write(tr["qa_current"], ", ".join(sorted(st.session_state.reefer_tags)) or "—")
    if st.session_state.reefer_tags:
        if st.button(tr["qa_undo"], key="undo_tag"):
            last = sorted(st.session_state.reefer_tags)[-1]
            st.session_state.reefer_tags.discard(last)
            st.session_state.audit_log.append(
                f"{_dt.now().strftime('%m-%d %H:%M')}｜UNTAG {last}｜by=demo-user")
            st.success(tr["qa_undone"].format(truck=last))
    if st.session_state.audit_log:
        st.caption(tr["qa_log_title"])
        for line in st.session_state.audit_log[-5:]:
            st.code(line)

with tab4:
    st.subheader(tr["model_title"])
    # 真實檢驗數字（results.json），讀不到才用寫死備援
    try:
        f_m = (RESULTS.get("fatigue", {}).get("\u6aa2\u9a57\u7d50\u679c", {}) or {})
        i_m = (RESULTS.get("idle", {}).get("\u6aa2\u9a57\u7d50\u679c", {}) or {})
        _ov = (RESULTS.get("overload", {}).get("\u6aa2\u9a57\u7d50\u679c", {}) or {})
        o_m = (_ov.get("\u5168\u90e8\u7279\u5fb5", {}) or {})
        auc_f = f_m.get("AUC", 0.721); cap_f = f_m.get("前10%捕捉率", 0.307)
        mae_b = i_m.get("MAE_模型B_加入預定排班", 1.34); mae_h = i_m.get("MAE_單車歷史平均", 1.44)
        auc_o = o_m.get("AUC", 0.71); acc_o = o_m.get("前10%準確率", 0.786)
    except Exception:
        auc_f, cap_f, mae_b, mae_h, auc_o, acc_o = 0.721, 0.307, 1.34, 1.44, 0.71, 0.786
    if LANG == "zh":
        data = [["疲勞風險/15分", auc_f, f"前10%抓{cap_f:.0%}警示", "標高風險車+排班"],
                ["怠速油耗/日", f"MAE {mae_b}L", f"歷史平均MAE {mae_h}L，需排班才贏", "預設用歷史平均"],
                ["引擎高負載/15分", auc_o, f"前10%準確率{acc_o:.0%}", "找高負載車，不主張致油耗"]]
        cols_m = ["模型","檢驗","結果","週報用途"]
    else:
        data = [["Fatigue risk /15min", auc_f, f"Top 10% captures {cap_f:.0%}", "Flag trucks + rostering"],
                ["Idling fuel /day", f"MAE {mae_b}L", f"History-avg MAE {mae_h}L", "Default: history avg"],
                ["High engine load /15min", auc_o, f"Top-10% precision {acc_o:.0%}", "Find loaded trucks; no fuel claim"]]
        cols_m = ["Model","Eval","Result","Use in weekly report"]
    st.table(pd.DataFrame(data, columns=cols_m))
    st.warning(tr["limit"])
    st.caption(tr["footer"])
