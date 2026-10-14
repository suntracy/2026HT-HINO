"""
HINO x GenAI – 資料前處理流程（2026 和泰 AI 黑客松）

執行方式：
    python hino_pipeline_zh.py <輸入檔.xlsx|.csv|.parquet> [輸出資料夾]

輸出（位於輸出資料夾，預設為 ./hino_out）：
    points_clean.parquet   清理後的逐點資料（新增 trip_id、dt_sec、dt_used 等欄位）
    events.csv             每個事件一列（將 event[0..n] 由橫向欄位轉為直式表格）
    trips.csv              每趟行程一列（enabledCode + journeyCode）
    vehicles.csv           每輛車一列（enabledCode）
    data_quality.csv       資料品質檢查
    hino_summary.xlsx      以上所有表格合併為一個 Excel 檔

注意事項（依主辦單位 FAQ）：
  * 一趟行程須以 enabledCode + journeyCode 識別（journeyCode 本身在不同車輛之間「不」唯一）。
  * 客戶可在後台自行調整各車的事件門檻 → 比較各車事件次數時，須列為分析限制。
  * 所有車輛的 fenceId 皆為空值；資料不含駕駛身分（可假設：同一車輛的每趟行程對應一位駕駛）。

欄位名稱與輸出檔名維持英文，以便 notebook 與 hino_models_zh.py 讀取。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 事件代號 → 名稱（依 Event Type 0929 文件與「欄位對照表」工作表）
EVENT_NAMES = {
    1: "driving",            # 行駛中
    2: "idle",               # 怠速（開始／解除）
    3: "parked",             # 熄火
    4: "engineOn",           # 引擎啟動
    5: "fence",              # 進入／離開圍籬
    6: "accelerate",         # 急加速
    7: "decelerate",         # 急減速（急煞）
    8: "speeding",           # 超速（開始／解除）
    9: "dtc",                # DTC 故障碼
    10: "engineOverheat",    # 引擎過熱（開始／解除）
    11: "engineOverloading", # 引擎過載（開始／解除）
    12: "fatigue",           # 疲勞駕駛（DMS 鏡頭）
    13: "seatbelt",          # 未繫安全帶
    14: "phone",             # 使用通訊（手機）
    15: "smoking",           # 抽菸
    16: "distract",          # 分心
    17: "lensWardOff",       # 鏡頭遮蔽
    18: "driverLost",        # 駕駛消失
    19: "pcs",               # 預警式防護系統
    20: "ldws",              # 車道偏離警示系統
}
# 持續型事件：分別有「開始」與「解除」兩筆紀錄（解除那筆帶有 endTime 與 duration）
INTERVAL_EVENTS = {2, 5, 8, 10, 11}
CAR_STATUS = {0: "parking", 1: "driving", 2: "idling"}   # 熄火／行駛／怠速
MAX_GAP_SEC = 120  # 兩筆間隔超過此秒數視為訊號中斷，不計入狀態時間


# ----------------------------------------------------------------------------- 讀取資料
def load(path: Path) -> pd.DataFrame:
    """讀取原始檔。Excel：合併所有含 journeyCode 欄位的工作表，略過欄位說明工作表。"""
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path, low_memory=False)
    sheets = pd.read_excel(path, sheet_name=None)
    data = [df for name, df in sheets.items() if "journeyCode" in df.columns]
    if not data:
        raise ValueError("找不到含有 'journeyCode' 欄位的工作表")
    df = pd.concat(data, ignore_index=True)
    # 另存 parquet 快取，下次執行更快（讀取大型 xlsx 很慢）
    try:
        df.astype({c: "string" for c in df.columns if df[c].dtype == object}).to_parquet(
            path.with_suffix(".parquet"))
    except Exception:
        pass
    return df


# ----------------------------------------------------------------------------- 資料清理
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["enabledCode"] = df["enabledCode"].astype(str)
    df["journeyCode"] = df["journeyCode"].astype("Int64").astype(str)
    df["trip_id"] = df["enabledCode"] + "_" + df["journeyCode"]          # 唯一的行程代碼
    df = df.sort_values(["enabledCode", "time"]).reset_index(drop=True)
    # 行程第一筆有時缺車號 → 依車輛補齊
    if "carNum" in df:
        df["carNum"] = df.groupby("enabledCode")["carNum"].transform(lambda s: s.ffill().bfill())
    # 到「同一趟行程」下一筆的秒數（用於累計各狀態時間）
    nxt = df.groupby("trip_id")["time"].shift(-1)
    df["dt_sec"] = (nxt - df["time"]).dt.total_seconds()
    df["dt_used"] = df["dt_sec"].where(df["dt_sec"] <= MAX_GAP_SEC, 0).fillna(0)
    df["carStatus_name"] = df["carStatus"].map(CAR_STATUS)
    # 超過路段速限多少 km/h（僅在圖資有速限時計算）
    lim = df["gps.speedLimit"]
    df["over_limit_kmh"] = np.where(lim > 0, df["can.canSpeed"] - lim, np.nan)
    return df


# ----------------------------------------------------------------------------- 事件轉置
def explode_events(df: pd.DataFrame) -> pd.DataFrame:
    """把 event[i].xxx 橫向欄位轉成每個事件一列的直式表格。"""
    idx = sorted({int(m.group(1)) for c in df.columns if (m := re.match(r"event\[(\d+)\]\.", c))})
    keep = ["Type", "enabledCode", "carNum", "trip_id", "time", "gps.latitude", "gps.longitude",
            "can.canSpeed", "gps.speedLimit"]
    keep = [c for c in keep if c in df.columns]
    parts = []
    for i in idx:
        pre = f"event[{i}]."
        cols = [c for c in df.columns if c.startswith(pre)]
        sub = df.loc[df[pre + "type"].notna(), keep + cols].copy()
        if sub.empty:
            continue
        sub.columns = keep + [c[len(pre):] for c in cols]               # 去掉 event[i]. 前綴
        # 將 dtcCodes[0..n] 合併為一個字串
        dtc = [c for c in sub.columns if c.startswith("info.dtcCodes")]
        if dtc:
            sub["info.dtcCodes"] = sub[dtc].apply(
                lambda r: "&".join(str(v) for v in r if pd.notna(v)), axis=1)
            sub = sub.drop(columns=dtc)
        sub["slot"] = i
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    ev = pd.concat(parts, ignore_index=True)
    ev["type"] = ev["type"].astype(int)
    ev["event_name"] = ev["type"].map(EVENT_NAMES)
    ev["startTime"] = pd.to_datetime(ev["startTime"], errors="coerce")
    ev["endTime"] = pd.to_datetime(ev.get("endTime"), errors="coerce")
    # phase：start＝開始、end＝解除（含持續時間）、instant＝瞬間事件
    ev["phase"] = np.where(ev["type"].isin(INTERVAL_EVENTS),
                           np.where(ev["endTime"].notna(), "end", "start"), "instant")
    if "info.speed" in ev and "info.speedLimit" in ev:
        # 超速事件觸發時超出速限多少 km/h（僅開始那筆）
        ev["over_kmh"] = np.where(ev["phase"] == "start", ev["info.speed"] - ev["info.speedLimit"], np.nan)
    return ev.sort_values(["enabledCode", "time", "slot"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- 行程表
def trip_table(df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("trip_id")
    t = g.agg(
        enabledCode=("enabledCode", "first"), carNum=("carNum", "first"), Type=("Type", "first"),
        start=("time", "min"), end=("time", "max"), n_points=("time", "size"),
        mileage_start=("can.totalMileage", "min"), mileage_end=("can.totalMileage", "max"),
        fuel_start=("can.engine.totalFuelUsed", "min"), fuel_end=("can.engine.totalFuelUsed", "max"),
        eng_h_start=("can.engine.engineTotalTime", "min"), eng_h_end=("can.engine.engineTotalTime", "max"),
        speed_max=("can.canSpeed", "max"), rpm_max=("can.engine.rpm", "max"),
        coolant_max=("can.engine.engineCoolantTemp", "max"), load_max=("can.engine.engineLoad", "max"),
        battery_min_mV=("line.battery", "min"), fuel_level_start=("line.fuelLevel", "first"),
        fuel_level_end=("line.fuelLevel", "last"),
        can_abnormal_pts=("can.canStatus", lambda s: int((s == 9).sum())),
        over_limit_max=("over_limit_kmh", "max"),
    )
    # 累計型欄位：最後值 − 最初值
    t["duration_min"] = (t["end"] - t["start"]).dt.total_seconds() / 60
    t["distance_km"] = t["mileage_end"] - t["mileage_start"]
    t["fuel_L"] = t["fuel_end"] - t["fuel_start"]
    t["engine_h"] = t["eng_h_end"] - t["eng_h_start"]
    t["L_per_100km"] = np.where(t["distance_km"] >= 1, t["fuel_L"] / t["distance_km"] * 100, np.nan)

    # 各 carStatus 狀態的分鐘數
    st = df.pivot_table(index="trip_id", columns="carStatus_name", values="dt_used",
                        aggfunc="sum", fill_value=0) / 60
    st.columns = [f"min_{c}" for c in st.columns]
    t = t.join(st)
    moving = df[df["can.canSpeed"] > 0]
    t["avg_moving_speed"] = moving.groupby("trip_id")["can.canSpeed"].mean()
    # 超過速限的駕駛時間占比（僅計算有速限的路段）
    has_lim = df[(df["gps.speedLimit"] > 0) & (df["can.canSpeed"] > 0)]
    over = has_lim.assign(o=(has_lim["over_limit_kmh"] > 0) * has_lim["dt_used"])
    agg = over.groupby("trip_id").agg(o=("o", "sum"), tot=("dt_used", "sum"))
    t["pct_time_over_limit"] = (agg["o"] / agg["tot"].replace(0, np.nan) * 100).round(1)

    if not ev.empty:
        counted = ev[ev["phase"] != "end"]  # 持續型事件只算一次（開始那筆）
        cnt = counted.pivot_table(index="trip_id", columns="event_name", values="type",
                                  aggfunc="size", fill_value=0)
        cnt.columns = [f"n_{c}" for c in cnt.columns]
        t = t.join(cnt)
        ends = ev[ev["phase"] == "end"]     # 持續時間取自解除那筆
        dur = ends.pivot_table(index="trip_id", columns="event_name", values="info.duration",
                               aggfunc="sum", fill_value=0)
        dur.columns = [f"sec_{c}" for c in dur.columns]
        t = t.join(dur)
        for c in ("n_accelerate", "n_decelerate", "n_speeding"):
            if c in t:
                t[c.replace("n_", "per100km_")] = np.where(
                    t["distance_km"] >= 1, t[c] / t["distance_km"] * 100, np.nan)
    t = t.drop(columns=["mileage_start", "mileage_end", "fuel_start", "fuel_end",
                        "eng_h_start", "eng_h_end"])
    num = t.select_dtypes("number").columns
    t[num] = t[num].round(2)
    return t.reset_index().sort_values(["enabledCode", "start"])


# ----------------------------------------------------------------------------- 車輛表
def vehicle_table(trips: pd.DataFrame) -> pd.DataFrame:
    sum_cols = [c for c in trips.columns if c.startswith(("n_", "sec_", "min_"))] + [
        "distance_km", "fuel_L", "engine_h", "duration_min"]
    v = trips.groupby("enabledCode").agg(
        carNum=("carNum", "first"), Type=("Type", "first"), n_trips=("trip_id", "size"),
        first_seen=("start", "min"), last_seen=("end", "max"),
        speed_max=("speed_max", "max"), coolant_max=("coolant_max", "max"),
        battery_min_mV=("battery_min_mV", "min"),
    ).join(trips.groupby("enabledCode")[sum_cols].sum())
    v["L_per_100km"] = np.where(v["distance_km"] >= 1, v["fuel_L"] / v["distance_km"] * 100, np.nan)
    if "min_idling" in v and "min_driving" in v:
        v["idle_share_pct"] = v["min_idling"] / (v["min_idling"] + v["min_driving"]) * 100   # 怠速時間占比
    for c in ("n_accelerate", "n_decelerate", "n_speeding"):
        if c in v:
            v[c.replace("n_", "per100km_")] = np.where(
                v["distance_km"] >= 1, v[c] / v["distance_km"] * 100, np.nan)
    num = v.select_dtypes("number").columns
    v[num] = v[num].round(2)
    return v.reset_index()


# ----------------------------------------------------------------------------- 資料品質
def data_quality(raw: pd.DataFrame, df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("rows", len(df), "資料筆數"),
        ("vehicles", df["enabledCode"].nunique(), "車輛數"),
        ("trips", df["trip_id"].nunique(), "行程數"),
        ("time_min", df["time"].min(), "最早時間"),
        ("time_max", df["time"].max(), "最晚時間"),
        ("median_sampling_sec", df["dt_sec"].median(), "兩筆之間秒數的中位數"),
        ("pct_gaps_over_120s", round((df["dt_sec"] > MAX_GAP_SEC).mean() * 100, 2), "間隔 > 120 秒（訊號中斷）的比例 %"),
        ("pct_speedLimit_zero", round((df["gps.speedLimit"] == 0).mean() * 100, 2), "無路段速限的資料比例 %"),
        ("pct_canStatus_abnormal", round((df["can.canStatus"] == 9).mean() * 100, 2), "CAN 狀態異常的資料比例 %"),
        ("pct_carNum_missing_raw", round(raw["carNum"].isna().mean() * 100, 2) if "carNum" in raw else None, "缺車號的資料比例 %（補齊前）"),
        ("mileage_decreases", int((df.groupby("enabledCode")["can.totalMileage"].diff() < 0).sum()), "里程表數值下降次數（異常）"),
        ("fuelUsed_decreases", int((df.groupby("enabledCode")["can.engine.totalFuelUsed"].diff() < 0).sum()), "累計油耗下降次數（異常）"),
        ("duplicate_rows", int(raw.duplicated().sum()), "完全重複的資料列數"),
        ("events_total", len(ev), "事件總筆數"),
    ]
    if not ev.empty:
        for name, n in ev["event_name"].value_counts().items():
            rows.append((f"events_{name}", int(n), f"{name} 事件筆數"))
        if "over_kmh" in ev:
            sp = ev[(ev["type"] == 8) & (ev["phase"] == "start")]
            if len(sp):
                rows.append(("speeding_trigger_over_kmh_min", sp["over_kmh"].min(), "超速觸發時超出速限的最小值（km/h）"))
                rows.append(("speeding_trigger_over_kmh_median", sp["over_kmh"].median(), "超速觸發時超出速限的中位數（km/h）"))
        dec = ev[ev["type"] == 7]
        if len(dec):
            rows.append(("decel_speedDiff_min", dec["info.speedDifference"].min(), "急減速 speedDifference 最小值（km/h）"))
    return pd.DataFrame(rows, columns=["check", "value", "說明"])


# ----------------------------------------------------------------------------- 主程式
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "hino_out")
    out.mkdir(parents=True, exist_ok=True)

    print("讀取資料中...")
    raw = load(src)
    print("清理資料並轉置事件中...")
    df = clean(raw)
    ev = explode_events(df)
    trips = trip_table(df, ev)
    veh = vehicle_table(trips)
    dq = data_quality(raw, df, ev)

    try:  # 大型資料用 parquet 快很多（需 pip install pyarrow）；若無則改存 CSV
        df.to_parquet(out / "points_clean.parquet")
    except ImportError:
        df.to_csv(out / "points_clean.csv", index=False, encoding="utf-8-sig")
    ev.to_csv(out / "events.csv", index=False, encoding="utf-8-sig")
    trips.to_csv(out / "trips.csv", index=False, encoding="utf-8-sig")
    veh.to_csv(out / "vehicles.csv", index=False, encoding="utf-8-sig")
    dq.to_csv(out / "data_quality.csv", index=False, encoding="utf-8-sig")

    write_events = len(ev) < 1_000_000          # Excel 每個工作表上限約 100 萬列
    with pd.ExcelWriter(out / "hino_summary.xlsx") as w:
        dq.to_excel(w, sheet_name="data_quality", index=False)
        veh.to_excel(w, sheet_name="vehicles", index=False)
        trips.to_excel(w, sheet_name="trips", index=False)
        if write_events:
            ev.to_excel(w, sheet_name="events", index=False)

    print(dq.to_string(index=False))
    print(f"\n結果已儲存至 {out.resolve()}")


if __name__ == "__main__":
    main()
