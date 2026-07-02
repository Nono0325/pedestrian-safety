"""
=============================================================================
先行一步 - 多硬體效能對比模擬器
Multi-Hardware Benchmark for YOLOv8n + ByteTrack Pedestrian Detection

支援的硬體配置：
  1. Raspberry Pi 5 (CPU only)          — 基準，已知結果
  2. Raspberry Pi 5 + ONNX INT8         — 軟體優化
  3. Raspberry Pi 5 + Hailo-8L (13T)    — 官方 AI Kit
  4. Raspberry Pi 5 + Hailo-8 (26T)     — 高階 NPU
  5. NVIDIA Jetson Orin Nano 4GB        — CUDA + TensorRT
  6. NVIDIA Jetson Orin NX 8GB          — 更高階 GPU
=============================================================================
"""

import cv2
import time
import json
import platform
import numpy as np
import sys
import os
import gc
import subprocess

print("=" * 65)
print("  先行一步 - 多硬體效能對比模擬器")
print("  Multi-Hardware Edge AI Benchmark")
print("=" * 65)
print()

# ── 依賴檢查 ──
missing = []
try:
    import psutil
except ImportError:
    missing.append("psutil")
try:
    from ultralytics import YOLO
except ImportError:
    missing.append("ultralytics")

if missing:
    for pkg in missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    import psutil
    from ultralytics import YOLO
else:
    import psutil
    from ultralytics import YOLO


# =============================================================================
# 硬體規格資料庫 (基於 2024-2025 真實 Benchmark 數據)
# =============================================================================
#
# 縮放因子計算邏輯：
#   scaled_latency = real_x86_latency * scale_factor
#   FPS = 1000 / scaled_latency
#
# Pi 5 CPU scale=5.5 → ~158ms → ~6.3fps  (已驗證)
# 各 NPU 數據來源：
#   Hailo-8L: Hailo 官方 YOLOv8n benchmark 100-400fps
#   Hailo-8:  Hailo 官方數據，約 Hailo-8L 的 2x
#   Jetson Orin Nano: NVIDIA JetPack 6 TensorRT, 30-60fps
#   Jetson Orin NX:   NVIDIA 官方, 60-120fps
# =============================================================================
HARDWARE_CONFIGS = [
    {
        "id":          "pi5_cpu",
        "name":        "Raspberry Pi 5\n(CPU Only)",
        "short_name":  "Pi 5 CPU",
        "color":       "#ef4444",
        "scale":       5.5,        # 已驗證：~6.3fps
        "price_usd":   80,
        "power_w":     5.0,
        "specs": {
            "CPU":     "Cortex-A76 × 4 @ 2.4GHz",
            "AI加速":  "無 (純 CPU 推論)",
            "RAM":     "8GB LPDDR4X",
            "功耗":    "~5W (推論時)",
            "參考價":  "~NT$ 2,500",
        },
        "notes": "純 CPU 推論，受限於 A76 整數運算能力",
        "thermal_penalty_start": 20,
        "thermal_penalty_pct":   0.08,
    },
    {
        "id":          "pi5_onnx",
        "name":        "Raspberry Pi 5\n+ ONNX INT8",
        "short_name":  "Pi 5 + ONNX",
        "color":       "#f59e0b",
        "scale":       3.5,        # ONNX + INT8 量化提速 ~35-40%
        "price_usd":   80,
        "power_w":     5.0,
        "specs": {
            "CPU":     "Cortex-A76 × 4 @ 2.4GHz",
            "AI加速":  "ONNX Runtime + INT8 量化",
            "RAM":     "8GB LPDDR4X",
            "功耗":    "~5W",
            "參考價":  "~NT$ 2,500 (軟體優化，無額外成本)",
        },
        "notes": "透過模型量化提速，無需額外硬體",
        "thermal_penalty_start": 25,
        "thermal_penalty_pct":   0.06,
    },
    {
        "id":          "pi5_hailo8l",
        "name":        "Raspberry Pi 5\n+ Hailo-8L AI Kit",
        "short_name":  "Pi 5 + Hailo-8L",
        "color":       "#22c55e",
        "scale":       0.22,       # YOLOv8n Hailo-8L 實測 ~100-120fps → ~8-10ms
        "price_usd":   80 + 70,
        "power_w":     8.0,
        "specs": {
            "CPU":     "Cortex-A76 × 4 @ 2.4GHz",
            "AI加速":  "Hailo-8L NPU, 13 TOPS",
            "RAM":     "8GB LPDDR4X",
            "功耗":    "~8W (NPU 全速)",
            "參考價":  "~NT$ 4,700 (Pi 5 + AI Kit M.2 HAT+)",
        },
        "notes": "官方 Raspberry Pi AI Kit，PCIe Gen 3 模式",
        "thermal_penalty_start": 100,  # NPU 散熱佳，不易節流
        "thermal_penalty_pct":   0.02,
    },
    {
        "id":          "pi5_hailo8",
        "name":        "Raspberry Pi 5\n+ Hailo-8 (26T)",
        "short_name":  "Pi 5 + Hailo-8",
        "color":       "#06b6d4",
        "scale":       0.11,       # Hailo-8 26TOPS, ~220-400fps → ~2.5-5ms
        "price_usd":   80 + 130,
        "power_w":     10.0,
        "specs": {
            "CPU":     "Cortex-A76 × 4 @ 2.4GHz",
            "AI加速":  "Hailo-8 NPU, 26 TOPS",
            "RAM":     "8GB LPDDR4X",
            "功耗":    "~10W",
            "參考價":  "~NT$ 6,500",
        },
        "notes": "Hailo-8 高階版，適合多攝影機並行處理",
        "thermal_penalty_start": 100,
        "thermal_penalty_pct":   0.01,
    },
    {
        "id":          "jetson_orin_nano",
        "name":        "NVIDIA Jetson\nOrin Nano 4GB",
        "short_name":  "Jetson Orin Nano",
        "color":       "#76b900",
        "scale":       0.45,       # TensorRT YOLOv8n ~40-60fps → ~17-25ms
        "price_usd":   199,
        "power_w":     10.0,
        "specs": {
            "CPU":     "Cortex-A78AE × 6 @ 1.5GHz",
            "AI加速":  "CUDA GPU + TensorRT, 40 TOPS",
            "RAM":     "4GB LPDDR5",
            "功耗":    "~10W (MAXN mode)",
            "參考價":  "~NT$ 6,200",
        },
        "notes": "CUDA + TensorRT 加速，生態系最完整",
        "thermal_penalty_start": 35,
        "thermal_penalty_pct":   0.04,
    },
    {
        "id":          "jetson_orin_nx",
        "name":        "NVIDIA Jetson\nOrin NX 8GB",
        "short_name":  "Jetson Orin NX",
        "color":       "#a855f7",
        "scale":       0.20,       # TensorRT YOLOv8n ~80-120fps → ~8-12ms
        "price_usd":   399,
        "power_w":     15.0,
        "specs": {
            "CPU":     "Cortex-A78AE × 8 @ 2.0GHz",
            "AI加速":  "CUDA GPU + TensorRT, 70 TOPS",
            "RAM":     "8GB LPDDR5",
            "功耗":    "~15W",
            "參考價":  "~NT$ 12,400",
        },
        "notes": "高效能邊緣 AI，可同時處理多路攝影機",
        "thermal_penalty_start": 45,
        "thermal_penalty_pct":   0.03,
    },
]

TARGET_FPS      = 10      # 目標 FPS
BUDGET_MS       = 100     # 目標延遲上限 (ms)
N_FRAMES        = 50      # 測試幀數
IMGSZ           = 320     # 推論解析度


# ── 合成測試影像 ──
def generate_test_frames(n=N_FRAMES):
    frames = []
    for i in range(n):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for y in range(200):
            val = int(20 + y * 0.3)
            frame[y, :] = [val // 2, val // 3, val]
        frame[300:, :] = [60, 55, 50]
        for x in range(50, 600, 40):
            frame[300:310, x:x + 20] = [200, 200, 200]
        frame[100:300, 0:150]   = [50, 50, 60]
        frame[150:300, 500:640] = [45, 45, 55]
        # 行人 1
        px1 = int(100 + (i / n) * 300)
        cv2.rectangle(frame, (px1 - 15, 190), (px1 + 15, 250), (100, 180, 100), -1)
        cv2.circle(frame, (px1, 175), 15, (200, 160, 120), -1)
        # 行人 2
        cv2.rectangle(frame, (480, 200), (510, 270), (120, 120, 180), -1)
        cv2.circle(frame, (495, 185), 15, (200, 160, 120), -1)
        # 行人 3 (危險，後半段出現)
        if i > n // 2:
            py3 = min(240 + (i - n // 2) * 2, 290)
            cv2.rectangle(frame, (288, py3 - 55), (312, py3), (100, 80, 180), -1)
            cv2.circle(frame, (300, py3 - 68), 13, (200, 160, 120), -1)
        cv2.putText(frame, f"Frame {i + 1:03d}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        frames.append(frame)
    return frames


# ── 取得 x86 基準推論時間 ──
def get_baseline_inference_times(model, frames, warmup=3):
    print(f"  🔥 暖機 {warmup} 幀...")
    for f in frames[:warmup]:
        model.track(f, persist=False, classes=[0], verbose=False, imgsz=IMGSZ)
    gc.collect()

    times = []
    for f in frames:
        t0 = time.perf_counter()
        model.track(f, persist=True, classes=[0],
                    tracker="bytetrack.yaml", verbose=False, imgsz=IMGSZ)
        times.append((time.perf_counter() - t0) * 1000)
    model.predictor = None  # 重置追蹤狀態
    gc.collect()
    return times


# ── 模擬指定硬體的效能 ──
def simulate_hardware(hw_cfg, baseline_times):
    scale  = hw_cfg["scale"]
    tp_start = hw_cfg["thermal_penalty_start"]
    tp_pct   = hw_cfg["thermal_penalty_pct"]

    results = []
    for i, real_ms in enumerate(baseline_times):
        pi_ms = real_ms * scale
        thermal = i >= tp_start
        if thermal:
            pi_ms *= (1 + tp_pct)

        fps = 1000 / pi_ms if pi_ms > 0 else 0
        results.append({
            "frame":    i + 1,
            "pi5_ms":  round(pi_ms, 2),
            "fps":      round(fps, 2),
            "thermal":  thermal,
            "budget_ok": pi_ms < BUDGET_MS,
        })
    return results


# ── 統計 ──
def compute_stats(results):
    ms_list = [r["pi5_ms"] for r in results]
    fps_list = [r["fps"] for r in results]
    return {
        "avg_fps":     round(np.mean(fps_list), 2),
        "min_fps":     round(np.min(fps_list), 2),
        "max_fps":     round(np.max(fps_list), 2),
        "avg_ms":      round(np.mean(ms_list), 2),
        "p95_ms":      round(np.percentile(ms_list, 95), 2),
        "budget_pct":  round(sum(1 for r in results if r["budget_ok"]) / len(results) * 100, 1),
        "thermal_pct": round(sum(1 for r in results if r["thermal"]) / len(results) * 100, 1),
    }


# =============================================================================
# HTML 報表產生器
# =============================================================================
def generate_comparison_report(all_results, output_path):
    # 準備各硬體的摘要
    summaries = []
    for hw, stats in all_results:
        meets = stats["avg_fps"] >= TARGET_FPS
        summaries.append({**hw, **stats, "meets_target": meets})

    # 排序 (FPS 高到低)
    summaries.sort(key=lambda x: x["avg_fps"], reverse=True)

    # 圖表資料
    hw_names = [s["short_name"] for s in summaries]
    fps_vals  = [s["avg_fps"]  for s in summaries]
    ms_vals   = [s["avg_ms"]   for s in summaries]
    budget_vals = [s["budget_pct"] for s in summaries]
    colors     = [s["color"] for s in summaries]
    prices     = [s["price_usd"] for s in summaries]
    powers     = [s["power_w"] for s in summaries]

    # 推薦 (達標且最便宜)
    meets = [s for s in summaries if s["meets_target"]]
    recommended = meets[-1] if meets else None  # 達標中最便宜

    # 卡片 HTML
    cards_html = ""
    for s in summaries:
        meets_badge = '<span class="badge badge-green">✅ 達標</span>' if s["meets_target"] \
                      else '<span class="badge badge-red">❌ 未達標</span>'
        if recommended and s["id"] == recommended["id"]:
            rec_tag = '<div class="rec-tag">⭐ 最佳性價比</div>'
        else:
            rec_tag = ""
        
        # 規格列表
        spec_rows = "".join(
            f'<tr><td class="spec-k">{k}</td><td class="spec-v">{v}</td></tr>'
            for k, v in s["specs"].items()
        )

        fps_color = "#22c55e" if s["meets_target"] else "#ef4444"
        fps_pct   = min(100, s["avg_fps"] / 30 * 100)

        cards_html += f"""
<div class="hw-card {'recommended' if recommended and s['id'] == recommended['id'] else ''}">
  {rec_tag}
  <div class="hw-card-header" style="border-color:{s['color']}">
    <div class="hw-dot" style="background:{s['color']}"></div>
    <div class="hw-title">{s['name'].replace(chr(10), '<br>')}</div>
    {meets_badge}
  </div>
  <div class="hw-fps" style="color:{fps_color}">{s['avg_fps']}<span class="hw-fps-unit">fps</span></div>
  <div class="hw-bar-wrap">
    <div class="hw-bar" style="width:{fps_pct:.1f}%; background:{s['color']}"></div>
    <div class="hw-bar-target"></div>
  </div>
  <div class="hw-stats-row">
    <div class="hw-stat"><div class="hw-stat-val">{s['avg_ms']}ms</div><div class="hw-stat-lbl">平均延遲</div></div>
    <div class="hw-stat"><div class="hw-stat-val">{s['p95_ms']}ms</div><div class="hw-stat-lbl">P95延遲</div></div>
    <div class="hw-stat"><div class="hw-stat-val">{s['budget_pct']}%</div><div class="hw-stat-lbl">預算達標</div></div>
    <div class="hw-stat"><div class="hw-stat-val">${s['price_usd']}</div><div class="hw-stat-lbl">美金售價</div></div>
  </div>
  <table class="spec-table">{spec_rows}</table>
  <div class="hw-note">💡 {s['notes']}</div>
</div>"""

    # 比較表格
    table_rows = ""
    for s in summaries:
        icon = "✅" if s["meets_target"] else "❌"
        fps_style = 'style="color:#22c55e;font-weight:700"' if s["meets_target"] else 'style="color:#ef4444"'
        table_rows += f"""
<tr>
  <td><span style="color:{s['color']};font-weight:600">{s['short_name']}</span></td>
  <td {fps_style}>{s['avg_fps']} fps</td>
  <td>{s['avg_ms']} ms</td>
  <td>{s['p95_ms']} ms</td>
  <td>{s['budget_pct']}%</td>
  <td>{s['thermal_pct']}%</td>
  <td>${s['price_usd']}</td>
  <td>{s['power_w']}W</td>
  <td>{icon}</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>先行一步 — 硬體選型比較報告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#080e1a;--card:#0f1929;--card2:#162035;--border:#1a2d47;
  --text:#e2e8f0;--muted:#64748b;--accent:#3b82f6;
  --green:#22c55e;--amber:#f59e0b;--red:#ef4444;--purple:#a855f7;
}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding-bottom:60px}}

/* HERO */
.hero{{
  background:linear-gradient(135deg,#080e1a 0%,#0f1f3d 50%,#0a1628 100%);
  border-bottom:1px solid var(--border);padding:52px 40px 40px;position:relative;overflow:hidden;
}}
.hero::before{{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 90% 60% at 60% 30%,rgba(59,130,246,.12),transparent);
}}
.hero::after{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,#3b82f6,#a855f7,transparent);
}}
.hero-tag{{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);
  color:#93c5fd;font-size:12px;font-weight:600;
  padding:4px 14px;border-radius:99px;margin-bottom:18px;
  letter-spacing:.06em;text-transform:uppercase;
}}
.hero h1{{
  font-size:clamp(26px,4vw,44px);font-weight:900;line-height:1.12;
  background:linear-gradient(135deg,#fff 0%,#93c5fd 60%,#c4b5fd 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;
}}
.hero p{{color:var(--muted);font-size:15px;max-width:680px;line-height:1.6}}
.hero-pills{{display:flex;gap:12px;margin-top:24px;flex-wrap:wrap}}
.pill{{
  background:rgba(255,255,255,.05);border:1px solid var(--border);
  padding:6px 16px;border-radius:99px;font-size:12px;color:#94a3b8;
  font-family:'JetBrains Mono',monospace;
}}

/* LAYOUT */
.container{{max-width:1300px;margin:0 auto;padding:0 24px}}
.section{{margin-top:48px}}
.section-title{{
  font-size:12px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);
  margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
}}
.section-title::before{{content:'';width:3px;height:16px;background:var(--accent);border-radius:2px}}

/* ALERT BAR */
.alert-bar{{
  border-radius:12px;padding:20px 24px;margin-top:32px;
  display:flex;align-items:center;gap:16px;
  background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.25);
}}
.alert-bar-icon{{font-size:32px}}
.alert-bar-text h3{{font-size:17px;font-weight:700;margin-bottom:4px}}
.alert-bar-text p{{color:var(--muted);font-size:13px}}
.alert-bar-badge{{
  margin-left:auto;background:rgba(34,197,94,.15);
  border:1px solid rgba(34,197,94,.3);color:var(--green);
  padding:6px 16px;border-radius:99px;font-size:13px;font-weight:600;
  white-space:nowrap;
}}

/* HW CARDS */
.hw-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:20px}}
.hw-card{{
  background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:24px;position:relative;transition:transform .2s,border-color .2s;
  overflow:hidden;
}}
.hw-card:hover{{transform:translateY(-3px);border-color:rgba(59,130,246,.4)}}
.hw-card.recommended{{border-color:rgba(34,197,94,.4);box-shadow:0 0 30px rgba(34,197,94,.08)}}
.rec-tag{{
  position:absolute;top:14px;right:14px;
  background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);
  color:var(--green);font-size:11px;font-weight:700;
  padding:3px 10px;border-radius:99px;
}}
.hw-card-header{{
  display:flex;align-items:center;gap:12px;margin-bottom:16px;
  padding-bottom:14px;border-bottom:1px solid;
}}
.hw-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.hw-title{{font-size:14px;font-weight:700;line-height:1.3;flex:1}}
.hw-fps{{font-size:48px;font-weight:900;line-height:1;margin-bottom:8px;font-family:'JetBrains Mono',monospace}}
.hw-fps-unit{{font-size:18px;font-weight:400;color:var(--muted);margin-left:4px}}
.hw-bar-wrap{{
  height:8px;background:var(--border);border-radius:99px;margin-bottom:18px;
  position:relative;overflow:hidden;
}}
.hw-bar{{height:100%;border-radius:99px;transition:width .5s ease}}
.hw-bar-target{{
  position:absolute;top:0;bottom:0;
  left:33.3%;  /* 10fps / 30fps * 100 */
  width:2px;background:#fff;opacity:.3;
}}
.hw-stats-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}}
.hw-stat{{background:var(--card2);border-radius:8px;padding:10px 8px;text-align:center}}
.hw-stat-val{{font-size:15px;font-weight:700;font-family:'JetBrains Mono',monospace}}
.hw-stat-lbl{{font-size:10px;color:var(--muted);margin-top:2px;text-transform:uppercase;letter-spacing:.04em}}
.spec-table{{width:100%;border-collapse:collapse;margin-bottom:12px}}
.spec-k{{font-size:11px;color:var(--muted);padding:4px 0;width:35%}}
.spec-v{{font-size:11px;font-weight:500;padding:4px 0;font-family:'JetBrains Mono',monospace}}
.hw-note{{font-size:11px;color:var(--muted);font-style:italic;line-height:1.5}}

/* CHARTS */
.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
@media(max-width:700px){{.chart-grid{{grid-template-columns:1fr}}}}
.chart-card{{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:24px;
}}
.chart-card h3{{font-size:13px;font-weight:600;color:var(--muted);margin-bottom:18px}}
.chart-wrap{{position:relative;height:260px}}

/* TABLE */
.table-card{{
  background:var(--card);border:1px solid var(--border);
  border-radius:12px;overflow-x:auto;
}}
table.comp{{width:100%;border-collapse:collapse;min-width:700px}}
table.comp thead tr{{background:var(--card2)}}
table.comp th{{
  padding:12px 14px;text-align:left;
  font-size:11px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;
  border-bottom:1px solid var(--border);
}}
table.comp td{{
  padding:10px 14px;font-size:13px;
  border-bottom:1px solid rgba(26,45,71,.6);
  font-family:'JetBrains Mono',monospace;
}}
table.comp tr:last-child td{{border-bottom:none}}
table.comp tr:hover td{{background:rgba(59,130,246,.04)}}

/* BADGE */
.badge{{display:inline-block;padding:3px 9px;border-radius:99px;font-size:11px;font-weight:700}}
.badge-green{{background:rgba(34,197,94,.15);color:var(--green)}}
.badge-red{{background:rgba(239,68,68,.15);color:var(--red)}}

/* FOOTER */
.footer{{
  text-align:center;margin-top:70px;color:var(--muted);
  font-size:12px;padding:0 24px;line-height:1.8;
}}
</style>
</head>
<body>

<!-- HERO -->
<div class="hero">
  <div class="container">
    <div class="hero-tag">🔬 硬體選型報告 · Hardware Benchmark</div>
    <h1>先行一步 — 多硬體效能對比</h1>
    <p>基於真實 YOLOv8n 推論基準，模擬各邊緣 AI 硬體平台的效能表現，協助選擇最適合部署的硬體組合。</p>
    <div class="hero-pills">
      <div class="pill">YOLOv8n @ imgsz=320</div>
      <div class="pill">ByteTrack 追蹤</div>
      <div class="pill">目標 ≥ {TARGET_FPS} fps</div>
      <div class="pill">{N_FRAMES} 幀測試</div>
      <div class="pill">6 種硬體配置</div>
    </div>
  </div>
</div>

<div class="container">

  <!-- 推薦提示 -->
  {'<div class="alert-bar"><div class="alert-bar-icon">⭐</div><div class="alert-bar-text"><h3>最佳性價比推薦：' + recommended["name"].replace(chr(10)," ") + '</h3><p>在達標硬體中，此配置具有最低成本，預估 ' + str(recommended["avg_fps"]) + ' fps，售價約 $' + str(recommended["price_usd"]) + ' USD。</p></div><div class="alert-bar-badge">推薦選擇</div></div>' if recommended else ''}

  <!-- HW CARDS -->
  <div class="section">
    <div class="section-title">各硬體平台模擬結果</div>
    <div class="hw-grid">
      {cards_html}
    </div>
  </div>

  <!-- CHARTS -->
  <div class="section">
    <div class="section-title">效能視覺化對比</div>
    <div class="chart-grid">
      <div class="chart-card">
        <h3>⚡ 平均 FPS 對比 (目標線：{TARGET_FPS} fps)</h3>
        <div class="chart-wrap"><canvas id="fpsChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>⏱ 平均推論延遲 (ms，越低越好)</h3>
        <div class="chart-wrap"><canvas id="msChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>💰 成本 vs FPS 效益分析</h3>
        <div class="chart-wrap"><canvas id="costChart"></canvas></div>
      </div>
      <div class="chart-card">
        <h3>🎯 預算達標率 (幀延遲 &lt; {BUDGET_MS}ms)</h3>
        <div class="chart-wrap"><canvas id="budgetChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- TABLE -->
  <div class="section">
    <div class="section-title">完整數據對比表</div>
    <div class="table-card">
      <table class="comp">
        <thead>
          <tr>
            <th>硬體平台</th>
            <th>平均 FPS</th>
            <th>平均延遲</th>
            <th>P95 延遲</th>
            <th>預算達標</th>
            <th>熱節流</th>
            <th>售價(USD)</th>
            <th>功耗</th>
            <th>達標</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- 選型建議 -->
  <div class="section">
    <div class="section-title">選型決策建議</div>
    <div class="chart-card">
      <table class="comp">
        <thead><tr><th>場景</th><th>推薦硬體</th><th>原因</th></tr></thead>
        <tbody>
          <tr>
            <td>💰 預算有限 (&lt; NT$5,000)</td>
            <td style="color:#22c55e;font-weight:600">Pi 5 + Hailo-8L AI Kit</td>
            <td>官方套件，開箱即用，效能遠超需求</td>
          </tr>
          <tr>
            <td>🔬 開發靈活性優先</td>
            <td style="color:#76b900;font-weight:600">Jetson Orin Nano</td>
            <td>CUDA/TensorRT 生態最完整，支援多模型並行</td>
          </tr>
          <tr>
            <td>📷 多路攝影機 (3+)</td>
            <td style="color:#06b6d4;font-weight:600">Pi 5 + Hailo-8 (26T)</td>
            <td>26 TOPS 算力可並行處理多路串流</td>
          </tr>
          <tr>
            <td>⚡ 極限效能需求</td>
            <td style="color:#a855f7;font-weight:600">Jetson Orin NX 8GB</td>
            <td>70 TOPS GPU 算力，適合複雜 AI 管線</td>
          </tr>
          <tr>
            <td>🛠 零額外成本優化</td>
            <td style="color:#f59e0b;font-weight:600">Pi 5 + ONNX INT8</td>
            <td>僅需轉換模型格式，免費提升 ~35% 效能</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>

</div>

<div class="footer">
  <p>先行一步 (One Step Ahead) · 115年人本環境全國大專院校學生競賽</p>
  <p>效能數據基於 Hailo 官方 Benchmark (2024-2025) 及 NVIDIA JetPack 6 實測數據，縮放計算自 YOLOv8n x86 基準推論時間</p>
</div>

<script>
const labels      = {json.dumps(hw_names)};
const fpsVals     = {json.dumps(fps_vals)};
const msVals      = {json.dumps(ms_vals)};
const budgetVals  = {json.dumps(budget_vals)};
const prices      = {json.dumps(prices)};
const powers      = {json.dumps(powers)};
const bgColors    = {json.dumps(colors)};
const borderColors= bgColors.map(c => c);

const baseOpts = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{
    legend:{{display:false}},
    tooltip:{{
      backgroundColor:'#0f1929',borderColor:'#1a2d47',borderWidth:1,
      titleColor:'#e2e8f0',bodyColor:'#94a3b8',padding:12,
    }}
  }},
  scales:{{
    x:{{ticks:{{color:'#64748b',font:{{size:11}}}},grid:{{color:'#1a2d47'}}}},
    y:{{ticks:{{color:'#64748b',font:{{size:11}}}},grid:{{color:'#1a2d47'}}}},
  }}
}};

// FPS Chart
new Chart(document.getElementById('fpsChart'),{{
  type:'bar',
  data:{{
    labels,
    datasets:[
      {{
        label:'平均 FPS',
        data:fpsVals,
        backgroundColor:bgColors.map(c=>c+'cc'),
        borderColor:borderColors,
        borderWidth:2,borderRadius:6,
      }},
      {{
        label:'目標 {TARGET_FPS} fps',
        data:labels.map(()=>{TARGET_FPS}),
        type:'line',
        borderColor:'#ffffff55',borderWidth:2,borderDash:[6,4],
        pointRadius:0,fill:false,
      }}
    ]
  }},
  options:{{...baseOpts,scales:{{...baseOpts.scales,y:{{...baseOpts.scales.y,min:0}}}}}}
}});

// Latency Chart
new Chart(document.getElementById('msChart'),{{
  type:'bar',
  data:{{
    labels,
    datasets:[{{
      label:'平均延遲 (ms)',
      data:msVals,
      backgroundColor:bgColors.map(c=>c+'bb'),
      borderColor:borderColors,borderWidth:2,borderRadius:6,
    }}]
  }},
  options:{{...baseOpts,scales:{{...baseOpts.scales,y:{{...baseOpts.scales.y,min:0}}}}}}
}});

// Cost vs FPS scatter
new Chart(document.getElementById('costChart'),{{
  type:'scatter',
  data:{{
    datasets: labels.map((name,i) => ({{
      label:name,
      data:[{{x:prices[i],y:fpsVals[i]}}],
      backgroundColor:bgColors[i]+'cc',
      borderColor:bgColors[i],
      borderWidth:2,
      pointRadius:10,pointHoverRadius:14,
    }}))
  }},
  options:{{
    ...baseOpts,
    plugins:{{
      ...baseOpts.plugins,
      legend:{{display:true,labels:{{color:'#94a3b8',font:{{size:10}},boxWidth:10,padding:8}}}}
    }},
    scales:{{
      x:{{...baseOpts.scales.x,title:{{display:true,text:'售價 (USD)',color:'#64748b'}},min:50}},
      y:{{...baseOpts.scales.y,title:{{display:true,text:'FPS',color:'#64748b'}},min:0}},
    }}
  }}
}});

// Budget Chart
new Chart(document.getElementById('budgetChart'),{{
  type:'bar',
  data:{{
    labels,
    datasets:[{{
      label:'預算達標率 (%)',
      data:budgetVals,
      backgroundColor:budgetVals.map(v=>v>=80?'#22c55ecc':v>=40?'#f59e0bcc':'#ef4444cc'),
      borderColor:budgetVals.map(v=>v>=80?'#22c55e':v>=40?'#f59e0b':'#ef4444'),
      borderWidth:2,borderRadius:6,
    }}]
  }},
  options:{{...baseOpts,scales:{{...baseOpts.scales,y:{{...baseOpts.scales.y,min:0,max:100}}}}}}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# =============================================================================
# 主程式
# =============================================================================
def main():
    print(f"🖥  主機: {platform.processor() or platform.machine()}")
    print(f"💾 RAM: {round(psutil.virtual_memory().total/1024**3,1)} GB")
    print(f"🎯 目標: ≥ {TARGET_FPS} fps / < {BUDGET_MS} ms\n")

    # 載入模型
    print("📦 載入 YOLOv8n 模型...")
    model = YOLO("yolov8n.pt")
    print("✅ 模型就緒\n")

    # 產生測試幀
    print(f"🎞  產生 {N_FRAMES} 幀測試影像...")
    frames = generate_test_frames(N_FRAMES)
    print("✅ 完成\n")

    # 取得 x86 基準時間
    print("⏱  測量 x86 基準推論時間 (作為縮放基準)...")
    baseline_times = get_baseline_inference_times(model, frames)
    base_avg = round(np.mean(baseline_times), 2)
    print(f"✅ x86 基準平均: {base_avg}ms/幀\n")

    # 對每個硬體配置進行模擬
    print("=" * 65)
    print("  🚀 多硬體效能模擬")
    print("=" * 65)

    all_results = []
    for hw in HARDWARE_CONFIGS:
        print(f"\n▶  [{hw['short_name']:20s}] 縮放因子 {hw['scale']}x")
        results = simulate_hardware(hw, baseline_times)
        stats   = compute_stats(results)
        all_results.append((hw, stats))
        
        meets = "✅" if stats["avg_fps"] >= TARGET_FPS else "❌"
        print(f"   FPS: {stats['avg_fps']:6.2f} | Latency: {stats['avg_ms']:7.2f}ms | "
              f"Budget: {stats['budget_pct']:5.1f}% | {meets}")

    # 報表
    print("\n")
    print("=" * 65)
    print("  📊 摘要")
    print("=" * 65)
    for hw, stats in sorted(all_results, key=lambda x: x[1]["avg_fps"], reverse=True):
        icon = "✅" if stats["avg_fps"] >= TARGET_FPS else "❌"
        print(f"  {icon} {hw['short_name']:22s}: {stats['avg_fps']:6.2f} fps  ({stats['avg_ms']:.1f}ms)")
    print("=" * 65)

    # 產生 HTML
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hardware_comparison_report.html"
    )
    generate_comparison_report(all_results, report_path)
    print(f"\n📄 報表已儲存：{report_path}")

    import webbrowser
    webbrowser.open(f"file:///{report_path.replace(os.sep, '/')}")
    print("🌐 已開啟 HTML 報表於瀏覽器")


if __name__ == "__main__":
    main()
