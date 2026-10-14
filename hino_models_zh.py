"""
HINO x GenAI – 頭家副駕AI 的 3 個預測模型
==========================================

執行：python hino_models_zh.py <pipeline輸出資料夾> [模型儲存資料夾]
      （資料夾內需有 hino_pipeline_zh.py 產生的 points_clean.parquet 與 events.csv）

三個模型皆以「未來」資料檢驗（以 1–7 月訓練，9–11 月測試）：
  1. 疲勞風險   – 每車 × 每 15 分鐘時段，是否出現疲勞警示（type 12）。
  2. 怠速預測   – 每車每個工作日怠速（停止不動）所耗油量（公升）。
  3. 引擎高負載 – 每車 × 每 15 分鐘時段，是否出現高負載片段（自行計算，見下方說明）。

輸出：models/fatigue.joblib、models/idle.joblib、models/overload.joblib
      models/results.json（檢驗指標 + 各車風險分數，供原型使用）
需求套件：pandas, numpy, scikit-learn, joblib, pyarrow
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

SPLIT = pd.Timestamp('2025-08-01')          # 此日期前為訓練集，此日期起為測試集
# 資料集的 24 個核心抽樣日（單數月的 1/11/21/月底）
CORE = pd.to_datetime(['2025-01-01', '2025-01-11', '2025-01-21', '2025-01-31', '2025-03-01', '2025-03-11', '2025-03-21',
                       '2025-03-31', '2025-05-01', '2025-05-11', '2025-05-21', '2025-05-31', '2025-07-01', '2025-07-11',
                       '2025-07-21', '2025-07-31', '2025-09-01', '2025-09-11', '2025-09-21', '2025-09-30', '2025-11-01',
                       '2025-11-11', '2025-11-21', '2025-11-30'])
# 2025 年台灣國定假日（含連假、補假）
HOL = pd.to_datetime(['2025-01-01'] + [f'2025-01-{d}' for d in range(25, 32)] + [
    '2025-02-01', '2025-02-02', '2025-02-28', '2025-03-01', '2025-03-02', '2025-04-03', '2025-04-04', '2025-04-05',
    '2025-04-06', '2025-05-01', '2025-05-30', '2025-05-31', '2025-06-01', '2025-09-28', '2025-09-29', '2025-10-04',
    '2025-10-05', '2025-10-06', '2025-10-10', '2025-10-11', '2025-10-12', '2025-10-24', '2025-10-25', '2025-10-26',
    '2025-12-25'])


def gbc():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                          class_weight='balanced', random_state=0)


def eval_clf(train, test, feats, y='y'):
    """訓練後於測試集評分：AUC、AP、前 10% 高風險時段捕捉到的事件比例、單車內 AUC。"""
    m = gbc().fit(train[feats], train[y])
    pr = m.predict_proba(test[feats])[:, 1]
    t = test.assign(pr=pr)
    top = t.sort_values('pr', ascending=False)
    k = int(len(top) * 0.1)
    within = [roc_auc_score(d[y], d.pr) for _, d in t.groupby('enabledCode') if d[y].sum() >= 5 and d[y].nunique() > 1]
    return m, {'AUC': round(roc_auc_score(t[y], pr), 3), 'AP': round(average_precision_score(t[y], pr), 3),
               '基準比例': round(t[y].mean(), 3),
               '前10%捕捉率': round(top[y].iloc[:k].sum() / top[y].sum(), 3),
               '前10%準確率': round(top[y].iloc[:k].mean(), 3),
               '單車內AUC': round(float(np.mean(within)), 3) if within else None}


# --------------------------------------------------------------------------- 基礎資料
def load(d: Path):
    p = pd.read_parquet(d / 'points_clean.parquet')
    ev = pd.read_csv(d / 'events.csv', low_memory=False)
    p['time'] = pd.to_datetime(p['time']).astype('datetime64[ns]')
    ev['time'] = pd.to_datetime(ev['time']).astype('datetime64[ns]')
    p = p.sort_values(['enabledCode', 'time']).reset_index(drop=True)
    p['date'] = p.time.dt.normalize()
    p['hour'] = p.time.dt.hour
    p['is_drive'], p['is_idle'] = p.carStatus == 1, p.carStatus == 2
    p['drive_s'], p['idle_s'] = p.dt_used * p.is_drive, p.dt_used * p.is_idle
    # 油耗增量：同一趟次中下一筆累積油耗 − 本筆；排除負值、異常跳動（> 2 L）與斷訊（> 120 秒）
    g = p.groupby('trip_id')
    d_fuel = g['can.engine.totalFuelUsed'].shift(-1) - p['can.engine.totalFuelUsed']
    p['fuel'] = d_fuel.where((d_fuel >= 0) & (d_fuel <= 2) & (p.dt_sec <= 120), 0)
    # 工作班次 = 間隔 < 30 分鐘的連續趟次
    t = p.groupby('trip_id').agg(ec=('enabledCode', 'first'), start=('time', 'min'), end=('time', 'max')).sort_values(['ec', 'start'])
    gap = (t.start - t.groupby('ec').end.shift()).dt.total_seconds() / 60
    t['block'] = (gap.isna() | (gap >= 30)).cumsum()
    bl = t.groupby('block').agg(enabledCode=('ec', 'first'), b_start=('start', 'min')).sort_values('b_start')
    p = pd.merge_asof(p.sort_values('time'), bl, left_on='time', right_on='b_start', by='enabledCode')
    p['h_in'] = (p.time - p.b_start).dt.total_seconds() / 3600          # 本班次已進行的小時數
    p = p.sort_values(['enabledCode', 'time']).reset_index(drop=True)
    p['cum_drive'] = p.groupby(['enabledCode', 'date']).drive_s.cumsum() / 3600   # 當日累積駕駛小時
    return p, ev


def segments(p, ev, cars, label_types, other_types=()):
    """為車輛 `cars` 建立 15 分鐘時段表；標籤 = 該時段內有屬於 `label_types` 的事件（僅計開始/瞬間事件列）。"""
    x = p[p.enabledCode.isin(cars)].copy()
    x['seg'] = x.time.dt.floor('15min')
    S = x.groupby(['enabledCode', 'seg']).agg(
        Type=('Type', 'first'), drive_min=('drive_s', lambda s: s.sum() / 60), idle_min=('idle_s', lambda s: s.sum() / 60),
        h_in=('h_in', 'min'), cum_drive=('cum_drive', 'min'), spd=('can.canSpeed', 'mean'), spd_max=('can.canSpeed', 'max'),
        rpm=('can.engine.rpm', 'mean'), load=('can.engine.engineLoad', 'mean'), load_max=('can.engine.engineLoad', 'max'),
        lim=('gps.speedLimit', 'mean'), coolant=('can.engine.engineCoolantTemp', 'mean')).reset_index()
    S = S[S.drive_min >= 3]                                  # 只保留實際駕駛 ≥ 3 分鐘的時段
    e = ev[ev.phase != 'end'].assign(seg=lambda d: d.time.dt.floor('15min'))
    S = S.join(e[e.type.isin(label_types)].groupby(['enabledCode', 'seg']).size().rename('n_lab'), on=['enabledCode', 'seg'])
    S = S.join(e[e.type.isin(other_types)].groupby(['enabledCode', 'seg']).size().rename('n_oth'), on=['enabledCode', 'seg'])
    S = S.fillna({'n_lab': 0, 'n_oth': 0}).sort_values(['enabledCode', 'seg']).reset_index(drop=True)
    gS = S.groupby('enabledCode')
    S['lab_1h'] = 0.0
    S['oth_1h'] = 0.0
    for k in range(1, 5):                                   # 前 1 小時內的事件數
        near = (S.seg - gS.seg.shift(k)) <= pd.Timedelta('60min')
        S['lab_1h'] += gS.n_lab.shift(k).fillna(0) * near
        S['oth_1h'] += gS.n_oth.shift(k).fillna(0) * near
    near1 = (S.seg - gS.seg.shift(1)) <= pd.Timedelta('30min')
    for c in ['spd', 'spd_max', 'rpm', 'load', 'load_max', 'idle_min', 'lim', 'coolant']:   # 前一個 15 分鐘的駕駛行為
        S[c + '_prev'] = gS[c].shift(1).where(near1)
    S['y'] = (S.n_lab > 0).astype(int)
    S['hour'] = S.seg.dt.hour
    S['hsin'], S['hcos'] = np.sin(2 * np.pi * S.hour / 24), np.cos(2 * np.pi * S.hour / 24)   # 時刻的週期編碼
    S['night_1_5'] = S.hour.between(1, 5).astype(int)        # 凌晨 1–5 點
    S['post_lunch'] = S.hour.between(12, 14).astype(int)     # 午後 12–14 點
    S['type700'] = (S.Type == 700).astype(int)
    S['type500'] = (S.Type == 500).astype(int)
    return S


def add_history(S, train_mask):
    """車輛歷史事件率（僅以訓練集計算，避免資訊洩漏）。"""
    rate = S[train_mask].groupby('enabledCode').y.mean()
    S['veh_hist'] = S.enabledCode.map(rate).fillna(S[train_mask].y.mean())
    return rate


# --------------------------------------------------------------------------- 1. 疲勞風險
def fatigue_model(p, ev, out):
    cars = ev[ev.type == 12].enabledCode.unique()            # 有疲勞事件的車（具 DMS 的 10 台）
    S = segments(p, ev, cars, label_types=[12], other_types=[14, 16])
    tr = S.seg < SPLIT
    add_history(S, tr)
    FT = ['h_in', 'cum_drive', 'hsin', 'hcos', 'night_1_5', 'post_lunch', 'type700',
          'spd_prev', 'spd_max_prev', 'rpm_prev', 'load_prev', 'idle_min_prev', 'lab_1h', 'oth_1h', 'veh_hist']
    _, metr = eval_clf(S[tr], S[~tr], FT)
    # 最終模型：以全部資料訓練，車輛歷史亦以全部資料計算
    S['veh_hist'] = S.enabledCode.map(S.groupby('enabledCode').y.mean())
    final = gbc().fit(S[FT], S.y)
    # 各車在 3 種情境下的風險分數，其餘特徵維持該車典型的駕駛行為（中位數）
    plate = p.groupby('enabledCode').carNum.first()
    scen = {'白天9點_第1小時': (9, 1.0), '下午14點_第6小時': (14, 6.0), '凌晨3點_第9小時': (3, 9.0)}
    rows = {}
    for car, g in S.groupby('enabledCode'):
        base = g[FT].median(numeric_only=True)
        r = {}
        for name, (hr, hin) in scen.items():
            z = base.copy()
            z['hsin'], z['hcos'] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
            z['night_1_5'], z['post_lunch'] = int(1 <= hr <= 5), int(12 <= hr <= 14)
            z['h_in'], z['cum_drive'] = hin, hin
            z['lab_1h'] = z['oth_1h'] = 0
            r[name] = round(float(final.predict_proba(z.to_frame().T[FT])[:, 1][0]), 3)
        r['平均風險'] = round(float(final.predict_proba(g[FT])[:, 1].mean()), 3)
        rows[plate[car]] = r
    out['fatigue'] = {'說明': '單一 15 分鐘時段內出現疲勞警示的機率（10 台具 DMS 的車）',
                      '時段數': int(len(S)), '檢驗結果': metr, '各車': rows}
    return final, FT


# --------------------------------------------------------------------------- 2. 怠速預測
def idle_model(p, out):
    p = p.assign(idle_fuel=p.fuel * p.is_idle)
    d = p[p.date.isin(CORE)].groupby(['enabledCode', 'date']).agg(
        Type=('Type', 'first'), idle=('idle_fuel', 'sum'), on_h=('dt_used', lambda s: s.sum() / 3600),
        drive_h=('drive_s', lambda s: s.sum() / 3600), first_hour=('hour', 'min')).reset_index()
    d = d[d.on_h >= 0.5].sort_values(['enabledCode', 'date']).reset_index(drop=True)   # 發動 ≥ 0.5 小時才算工作日
    d['day_type'] = d.date.map(lambda x: 3 if x in HOL else {5: 1, 6: 2}.get(x.weekday(), 0))   # 0 平日、1 週六、2 週日、3 國定假日
    d['month'] = d.date.dt.month
    g = d.groupby('enabledCode').idle
    d['prev_day'] = g.shift(1)                                           # 前一個抽樣日的怠速油耗
    d['mean_prev'] = g.transform(lambda s: s.shift(1).expanding().mean())  # 先前各日的平均（無資訊洩漏）
    d['type700'], d['type500'] = (d.Type == 700).astype(int), (d.Type == 500).astype(int)
    tr, te = d[d.date < SPLIT], d[d.date >= SPLIT].copy()
    # 對照基準：全車隊平均、車系平均、該車歷史平均
    res = {'MAE_全車隊平均': mean_absolute_error(te.idle, np.full(len(te), tr.idle.mean())),
           'MAE_車系平均': mean_absolute_error(te.idle, te.Type.map(tr.groupby('Type').idle.mean())),
           'MAE_單車歷史平均': mean_absolute_error(te.idle, te.enabledCode.map(tr.groupby('enabledCode').idle.mean()).fillna(tr.idle.mean()))}
    FA = ['mean_prev', 'prev_day', 'type700', 'type500', 'day_type', 'month']          # 事前即可得知的資訊
    FB = FA + ['drive_h', 'first_hour']                                                # + 預定排班（駕駛時數、出車時刻）
    models = {}
    for name, F in [('A_歷史與日曆', FA), ('B_加入預定排班', FB)]:
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=10,
                                          loss='absolute_error', random_state=0).fit(tr[F], tr.idle)
        res['MAE_模型' + name] = mean_absolute_error(te.idle, m.predict(te[F]))
        models[name] = (m, F)
    res = {k: round(float(v), 2) for k, v in res.items()}
    res['訓練車日數'], res['測試車日數'] = int(len(tr)), int(len(te))
    # 以全部資料訓練的模型 A，預測各車「下一個平日」（12 月）的怠速油耗
    mA = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=10,
                                       loss='absolute_error', random_state=0).fit(d[FA], d.idle)
    plate = p.groupby('enabledCode').carNum.first()
    last = d.groupby('enabledCode').agg(avg=('idle', 'mean'), last=('idle', 'last'), Type=('Type', 'first'))
    nxt = pd.DataFrame({'mean_prev': last.avg, 'prev_day': last['last'], 'type700': (last.Type == 700).astype(int),
                        'type500': (last.Type == 500).astype(int), 'day_type': 0, 'month': 12})
    pred = mA.predict(nxt[FA])
    out['idle'] = {'說明': '單一工作日內停止不動（怠速）所耗油量（公升）', '檢驗結果': res,
                   '下一個平日怠速預測_公升': {plate[i]: round(float(max(v, 0)), 2) for i, v in zip(nxt.index, pred)}}
    return mA, FA


# --------------------------------------------------------------------------- 3. 引擎高負載
def high_load_episodes(p, thr=83, min_sec=5):
    """由原始引擎負載自行計算「過載」，所有車輛採同一規則：
    連續列（同一趟次、間隔 <= 15 秒）負載 > thr%，且持續 >= min_sec 秒。"""
    hi = p['can.engine.engineLoad'] > thr
    brk = (~hi) | (p.trip_id != p.trip_id.shift()) | (p.time.diff().dt.total_seconds() > 15)
    run = brk.cumsum()
    r = p[hi].assign(run=run[hi]).groupby('run').agg(enabledCode=('enabledCode', 'first'), carNum=('carNum', 'first'),
                                                       time=('time', 'min'), t_end=('time', 'max'))
    r['sec'] = (r.t_end - r.time).dt.total_seconds()
    r = r[r.sec >= min_sec]
    return r.assign(type=111, phase='start')[['enabledCode', 'carNum', 'time', 'sec', 'type', 'phase']]   # 111 = 自訂事件代碼


def overload_model(p, ev, out):
    plate = p.groupby('enabledCode').carNum.first()
    km = (p['can.canSpeed'] * p.dt_used / 3600).groupby(p.enabledCode).sum()
    st = ev[(ev.type == 11) & (ev.phase == 'start')]
    ep = high_load_episodes(p)
    hi_time = (p.dt_used * (p['can.engine.engineLoad'] > 83)).groupby(p.enabledCode).sum() / p.dt_used.groupby(p.enabledCode).sum() * 100
    fuel = p.groupby('enabledCode').fuel.sum()
    desc = pd.DataFrame({'plate': plate, 'series': p.groupby('enabledCode').Type.first(), 'km': km, 'L_100km': fuel / km * 100,
                         'sys_events': st.groupby('enabledCode').size(), 'hi_eps': ep.groupby('enabledCode').size(),
                         'pct_load_gt83': hi_time}).fillna({'sys_events': 0, 'hi_eps': 0})
    desc['eps_100km'] = desc.hi_eps / desc.km * 100
    # 同車系內：高負載時間占比 與 百公里油耗 的 Spearman 相關
    corr = {}
    for t, g in desc.groupby('series'):
        if len(g) >= 4:
            corr[str(t)] = round(float(g[['pct_load_gt83', 'L_100km']].corr(method='spearman').iloc[0, 1]), 2)
    # 系統 type 11 事件只出現在 12/20 台車，但其他車也有負載 > 83% 的情況（設定不同）。
    # 因此模型標籤改用對全部 20 台車採同一規則自行計算的「高負載片段」。
    ev2 = pd.concat([ev, ep[['enabledCode', 'time', 'type', 'phase']]], ignore_index=True)
    S = segments(p, ev2, p.enabledCode.unique(), label_types=[111])
    tr = S.seg < SPLIT
    add_history(S, tr)
    F0 = ['veh_hist']
    F1 = ['h_in', 'hsin', 'hcos', 'type700', 'type500', 'spd_prev', 'spd_max_prev', 'rpm_prev', 'load_prev',
          'load_max_prev', 'lim_prev', 'coolant_prev', 'lab_1h']
    _, m0 = eval_clf(S[tr], S[~tr], F0)
    _, m1 = eval_clf(S[tr], S[~tr], F1)
    _, m2 = eval_clf(S[tr], S[~tr], F1 + F0)
    S['veh_hist'] = S.enabledCode.map(S.groupby('enabledCode').y.mean())
    final = gbc().fit(S[F1 + F0], S.y)
    out['overload'] = {
        '說明': '15 分鐘時段內是否出現高負載片段（負載 > 83% 持續 >= 5 秒，依 0929 文件規則，對 20 台車自行計算）',
        '備註': f'系統 type 11 事件僅出現在 {int((desc.sys_events > 0).sum())}/20 台車；其餘車輛仍有負載 > 83% 的情況，推測該事件被關閉或設定不同',
        '時段數': int(len(S)), '高負載時段比例': round(float(S.y.mean()), 3),
        '檢驗結果': {'僅車輛歷史': m0, '前15分鐘駕駛行為': m1, '全部特徵': m2},
        '同車系內高負載占比與百公里油耗_Spearman相關': corr,
        '各車': {r.plate: {'每百公里高負載片段': round(float(r.eps_100km), 1), '負載大於83%時間占比': round(float(r.pct_load_gt83), 1),
                         '系統過載事件數': int(r.sys_events), '百公里油耗_L': round(float(r.L_100km), 1)} for r in desc.itertuples()}}
    return final, F1 + F0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else 'models')
    dst.mkdir(parents=True, exist_ok=True)
    p, ev = load(src)
    out = {'資料切分': f'訓練 < {SPLIT.date()}，測試 >= {SPLIT.date()}'}
    for name, fn in [('fatigue', lambda: fatigue_model(p, ev, out)), ('idle', lambda: idle_model(p, out)),
                     ('overload', lambda: overload_model(p, ev, out))]:
        model, feats = fn()
        joblib.dump({'model': model, 'features': feats}, dst / f'{name}.joblib')
        print(f'[{name}] 檢驗結果：', json.dumps(out[name]['檢驗結果'], ensure_ascii=False))
    (dst / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('已儲存至', dst.resolve())


if __name__ == '__main__':
    main()
