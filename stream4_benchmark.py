"""
=============================================================================
先行一步 - 4路 ESP32-CAM 並行處理 × 多硬體需求計算器
4-Stream Concurrent Processing Simulator

問題核心:
  - 同時接收 4 個 ESP32-CAM 串流
  - 每路目標 ≥ 10 fps 的 AI 辨識
  - 計算各硬體需要「幾台」才能達標
  - 考量: 多執行緒、記憶體、熱節流、網路頻寬

模擬策略:
  - 在 x86 機器測量實際 YOLOv8n 推論基準
  - 套用各硬體縮放因子
  - 模擬 Round-Robin 4 路分配
  - 計算單機 / 集群需求
=============================================================================
"""

import cv2, time, json, platform, numpy as np, sys, os, gc, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

print("=" * 65)
print("  先行一步 - 4路 ESP32-CAM 並行處理需求模擬器")
print("=" * 65)
print()

# ── 依賴 ──
try:
    import psutil
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "psutil", "-q"])
    import psutil
    from ultralytics import YOLO

# =============================================================================
# 配置
# =============================================================================
N_CAMERAS      = 4      # ESP32-CAM 數量
TARGET_FPS     = 10     # 每路目標 FPS
BUDGET_MS      = 100    # 每幀延遲預算 (ms)
N_FRAMES       = 40     # 每路測試幀數
IMGSZ          = 320    # YOLOv8n 推論尺寸
STREAM_FPS     = 10     # ESP32-CAM 發送速率
MJPEG_KBPS     = 150    # ESP32-CAM MJPEG 每路頻寬 (KB/s @ 640x480)

# =============================================================================
# 硬體資料庫
# =============================================================================
HARDWARE_CONFIGS = [
    {
        "id": "pi5_cpu", "short": "Pi 5\nCPU only", "color": "#ef4444",
        "scale": 5.5,          # 單路 fps ≈ 6.3
        "max_parallel": 1,     # 最多幾路並行 (記憶體 & 頻寬限制)
        "ram_per_stream_mb": 400,
        "total_ram_mb": 7800,
        "price_usd": 80,
        "power_w": 5.0,
        "specs": {
            "推論後端": "CPU (Cortex-A76×4)",
            "算力": "~0.8 GFLOPS/core",
            "RAM": "8GB",
            "單路FPS": "~6.3 fps",
        },
        "note": "單路已不達標，集群方案：每台各跑 1 路",
    },
    {
        "id": "pi5_onnx", "short": "Pi 5\n+ONNX INT8", "color": "#f59e0b",
        "scale": 3.5,          # 單路 fps ≈ 10.0
        "max_parallel": 1,
        "ram_per_stream_mb": 350,
        "total_ram_mb": 7800,
        "price_usd": 80,
        "power_w": 5.0,
        "specs": {
            "推論後端": "ONNX Runtime + INT8",
            "算力": "同 CPU，模型量化節省",
            "RAM": "8GB",
            "單路FPS": "~10.0 fps",
        },
        "note": "剛好達標，但無餘裕，建議每台跑 1 路",
    },
    {
        "id": "pi5_hailo8l", "short": "Pi 5\n+Hailo-8L", "color": "#22c55e",
        "scale": 0.22,         # 單路 fps ≈ 110
        "max_parallel": 4,     # NPU 可並行 4 路
        "ram_per_stream_mb": 300,
        "total_ram_mb": 7800,
        "price_usd": 150,
        "power_w": 8.0,
        "specs": {
            "推論後端": "Hailo-8L NPU",
            "算力": "13 TOPS",
            "RAM": "8GB",
            "單路FPS": "~110 fps",
        },
        "note": "1 台即可並行處理 4 路，餘裕極大",
    },
    {
        "id": "pi5_hailo8", "short": "Pi 5\n+Hailo-8", "color": "#06b6d4",
        "scale": 0.11,         # 單路 fps ≈ 220
        "max_parallel": 8,     # 26TOPS 可處理更多路
        "ram_per_stream_mb": 300,
        "total_ram_mb": 7800,
        "price_usd": 210,
        "power_w": 10.0,
        "specs": {
            "推論後端": "Hailo-8 NPU",
            "算力": "26 TOPS",
            "RAM": "8GB",
            "單路FPS": "~220 fps",
        },
        "note": "1 台可輕鬆處理 4 路，甚至 8 路",
    },
    {
        "id": "jetson_orin_nano", "short": "Jetson\nOrin Nano", "color": "#76b900",
        "scale": 0.45,         # 單路 fps ≈ 50
        "max_parallel": 4,
        "ram_per_stream_mb": 400,
        "total_ram_mb": 3800,
        "price_usd": 199,
        "power_w": 10.0,
        "specs": {
            "推論後端": "CUDA + TensorRT",
            "算力": "40 TOPS",
            "RAM": "4GB",
            "單路FPS": "~50 fps",
        },
        "note": "1 台可並行 4 路，但 RAM 較緊（4GB）",
    },
    {
        "id": "jetson_orin_nx", "short": "Jetson\nOrin NX", "color": "#a855f7",
        "scale": 0.20,         # 單路 fps ≈ 110
        "max_parallel": 8,
        "ram_per_stream_mb": 400,
        "total_ram_mb": 7800,
        "price_usd": 399,
        "power_w": 15.0,
        "specs": {
            "推論後端": "CUDA + TensorRT",
            "算力": "70 TOPS",
            "RAM": "8GB",
            "單路FPS": "~110 fps",
        },
        "note": "1 台輕鬆處理 4 路，可擴展至 8+ 路",
    },
]


# =============================================================================
# 合成影像
# =============================================================================
def make_frame(cam_id, frame_idx, n_frames):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    hue = [30, 60, 120, 200][cam_id % 4]
    for y in range(200):
        img[y, :] = [max(0, hue - y // 5), max(0, 30 - y // 10), max(0, 60 - y // 5)]
    img[300:, :] = [55, 50, 45]
    for x in range(50, 600, 40):
        img[300:308, x:x + 18] = [180, 180, 180]
    # 行人
    px = int(80 + (frame_idx / n_frames) * 480)
    cv2.rectangle(img, (px - 14, 200), (px + 14, 260), (80 + cam_id * 30, 160, 100), -1)
    cv2.circle(img, (px, 185), 14, (200, 160, 120), -1)
    cv2.putText(img, f"CAM-{cam_id + 1} | Frame {frame_idx + 1:03d}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    return img


# =============================================================================
# 測量 x86 基準
# =============================================================================
def get_baseline(model, n_frames=N_FRAMES):
    print("  🔥 暖機...")
    dummy = [make_frame(0, i, n_frames) for i in range(3)]
    for f in dummy:
        model.track(f, persist=False, classes=[0], verbose=False, imgsz=IMGSZ)
    gc.collect()

    times = []
    for i in range(n_frames):
        f = make_frame(0, i, n_frames)
        t0 = time.perf_counter()
        model.track(f, persist=True, classes=[0],
                    tracker="bytetrack.yaml", verbose=False, imgsz=IMGSZ)
        times.append((time.perf_counter() - t0) * 1000)
    model.predictor = None
    gc.collect()
    return np.array(times)


# =============================================================================
# 模擬單台硬體處理 N 路串流 (Round-Robin 時間切片)
# =============================================================================
def simulate_single_device(hw, baseline_ms, n_cams=N_CAMERAS):
    """
    模擬一台硬體以 Round-Robin 方式處理 n_cams 路串流
    每幀耗時 = baseline_ms × scale
    4 路輪流 → 每路有效 FPS = 1 / (4 × single_frame_time)
    """
    scale = hw["scale"]
    tp_start = 20
    tp_pct   = 0.05

    per_frame_ms = []
    for i, bms in enumerate(baseline_ms):
        ms = bms * scale
        if i >= tp_start:
            ms *= (1 + tp_pct)
        per_frame_ms.append(ms)

    per_frame_ms = np.array(per_frame_ms)

    # 4 路 Round-Robin：每路等待其他 3 路的時間
    total_cycle_ms = per_frame_ms * n_cams
    per_cam_fps    = 1000 / total_cycle_ms  # 每路實際 FPS

    # 記憶體：n_cams 路 × 每路佔用
    ram_needed = hw["ram_per_stream_mb"] * n_cams
    ram_ok     = ram_needed <= hw["total_ram_mb"]

    return {
        "per_cam_fps_arr":  per_cam_fps,
        "avg_per_cam_fps":  round(float(np.mean(per_cam_fps)), 2),
        "min_per_cam_fps":  round(float(np.min(per_cam_fps)), 2),
        "avg_cycle_ms":     round(float(np.mean(total_cycle_ms)), 2),
        "ram_needed_mb":    ram_needed,
        "ram_ok":           ram_ok,
        "meets_target":     np.mean(per_cam_fps) >= TARGET_FPS and ram_ok,
    }


# =============================================================================
# 計算最少需要幾台 (集群方案)
# =============================================================================
def compute_cluster_need(hw, baseline_ms, n_cams=N_CAMERAS):
    """
    找出最少需要幾台此硬體才能讓所有 4 路都達標
    集群策略：每台分配盡量多路，直到所有路都達標
    """
    results = []
    for n_devices in range(1, n_cams + 1):
        # n_devices 台，n_cams 路平均分配
        cams_per_device = n_cams / n_devices  # 每台平均幾路
        # 每台跑 ceil(n_cams/n_devices) 路
        cams_this = int(np.ceil(cams_per_device))

        scale = hw["scale"]
        cycle_ms = np.mean(baseline_ms) * scale * cams_this
        fps_per_cam = 1000 / cycle_ms

        ram_needed = hw["ram_per_stream_mb"] * cams_this
        ram_ok = ram_needed <= hw["total_ram_mb"]

        meets = fps_per_cam >= TARGET_FPS and ram_ok
        results.append({
            "n_devices":      n_devices,
            "cams_per_device": cams_this,
            "fps_per_cam":    round(fps_per_cam, 2),
            "ram_per_device": ram_needed,
            "ram_ok":         ram_ok,
            "meets":          meets,
            "total_cost_usd": n_devices * hw["price_usd"],
            "total_power_w":  n_devices * hw["power_w"],
        })
        if meets:
            break  # 找到最少台數，停止

    return results


# =============================================================================
# HTML 報表
# =============================================================================
def generate_report(hw_summaries, all_cluster, baseline_avg, output_path):
    # 建議方案
    feasible = [(hw, c[-1]) for hw, c in all_cluster if c[-1]["meets"]]
    # 依總成本排序
    feasible.sort(key=lambda x: x[1]["total_cost_usd"])
    best = feasible[0] if feasible else None

    # ── 硬體卡片 ──
    cards_html = ""
    for hw, sim in hw_summaries:
        n_needed = next((c[-1]["n_devices"] for hh, c in all_cluster if hh["id"] == hw["id"]), "?")
        fps = sim["avg_per_cam_fps"]
        meets = sim["meets_target"]

        if meets:
            verdict_html = f'<div class="verdict-pass">✅ 1 台達標 · {fps} fps/路</div>'
        else:
            fps_1dev = fps
            verdict_html = f'<div class="verdict-fail">⚠️ 1台不足 ({fps_1dev}fps) · 需 {n_needed} 台</div>'

        spec_rows = "".join(f'<tr><td class="sk">{k}</td><td class="sv">{v}</td></tr>'
                            for k, v in hw["specs"].items())
        
        is_best = best and hw["id"] == best[0]["id"]
        cards_html += f"""
<div class="hw-card {'hw-best' if is_best else ''}">
  {'<div class="best-tag">⭐ 最佳方案</div>' if is_best else ''}
  <div class="hw-top" style="border-color:{hw['color']}">
    <div class="hw-dot" style="background:{hw['color']}"></div>
    <div class="hw-name">{hw['short'].replace(chr(10),'<br>')}</div>
  </div>
  {verdict_html}
  <div class="hw-big-num" style="color:{'#22c55e' if meets else '#f59e0b'}">{fps}<span class="unit"> fps</span></div>
  <div class="hw-sub">每路平均 · 1台處理4路</div>
  <div class="hw-need-box">
    <div class="hw-need-num">{'1' if meets else str(n_needed)}</div>
    <div class="hw-need-lbl">台達標最少需求</div>
  </div>
  <table class="stbl">{spec_rows}</table>
  <div class="hw-note">💡 {hw['note']}</div>
</div>"""

    # ── 集群對比表 ──
    cluster_table = ""
    for hw, cluster in all_cluster:
        for c in cluster:
            ok = "✅" if c["meets"] else "❌"
            ram_ok = "✅" if c["ram_ok"] else "⚠️ RAM不足"
            cluster_table += f"""<tr>
  <td style="color:{hw['color']};font-weight:600">{hw['short'].replace(chr(10),' ')}</td>
  <td style="text-align:center;font-weight:700">{c['n_devices']}</td>
  <td>{c['cams_per_device']}</td>
  <td style="color:{'#22c55e' if c['meets'] else '#f59e0b'};font-weight:700">{c['fps_per_cam']} fps</td>
  <td>{c['ram_per_device']} MB {ram_ok}</td>
  <td>${c['total_cost_usd']}</td>
  <td>{c['total_power_w']}W</td>
  <td style="font-size:18px">{ok}</td>
</tr>"""

    # ── 圖表數據 ──
    chart_hw     = [hw["short"].replace("\n"," ") for hw, _ in hw_summaries]
    chart_fps    = [sim["avg_per_cam_fps"] for _, sim in hw_summaries]
    chart_colors = [hw["color"] for hw, _ in hw_summaries]
    chart_n      = [next((c[-1]["n_devices"] for hh, c in all_cluster if hh["id"] == hw["id"]),4)
                    for hw, _ in hw_summaries]
    chart_cost   = [next((c[-1]["total_cost_usd"] for hh, c in all_cluster if hh["id"] == hw["id"]),999)
                    for hw, _ in hw_summaries]

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>先行一步 — 4路串流硬體需求分析</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#07101e;--card:#0d1b2e;--card2:#112036;--border:#182d47;
  --text:#e2e8f0;--muted:#60748a;--accent:#3b82f6;
  --green:#22c55e;--amber:#f59e0b;--red:#ef4444;
}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);padding-bottom:70px}}

/* HERO */
.hero{{
  background:linear-gradient(135deg,#07101e 0%,#0e1f3a 50%,#091426 100%);
  border-bottom:1px solid var(--border);padding:52px 40px 44px;position:relative;overflow:hidden;
}}
.hero::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 50% at 65% 25%,rgba(59,130,246,.13),transparent);}}
.hero::after{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent 0%,#3b82f6 40%,#a855f7 70%,transparent 100%);}}
.tag{{display:inline-flex;align-items:center;gap:6px;
  background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.3);
  color:#93c5fd;font-size:11px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  padding:4px 14px;border-radius:99px;margin-bottom:18px;}}
.hero h1{{font-size:clamp(26px,4vw,46px);font-weight:900;line-height:1.1;
  background:linear-gradient(135deg,#fff,#93c5fd 55%,#c4b5fd);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:12px;}}
.hero-sub{{color:var(--muted);font-size:15px;max-width:700px;line-height:1.65}}
.stat-row{{display:flex;gap:28px;margin-top:28px;flex-wrap:wrap}}
.stat-item .lbl{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}}
.stat-item .val{{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#cbd5e1;margin-top:2px}}

/* LAYOUT */
.c{{max-width:1320px;margin:0 auto;padding:0 24px}}
.sec{{margin-top:48px}}
.sec-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;}}
.sec-title::before{{content:'';width:3px;height:16px;background:var(--accent);border-radius:2px;flex-shrink:0}}

/* BEST ALERT */
.best-alert{{
  border-radius:14px;padding:22px 28px;margin-top:32px;
  background:linear-gradient(135deg,rgba(34,197,94,.07),rgba(59,130,246,.05));
  border:1px solid rgba(34,197,94,.25);
  display:flex;align-items:center;gap:20px;flex-wrap:wrap;
}}
.best-alert-icon{{font-size:42px;line-height:1;flex-shrink:0}}
.best-alert h3{{font-size:18px;font-weight:800;margin-bottom:5px}}
.best-alert p{{color:var(--muted);font-size:13px;line-height:1.6}}
.best-badge{{
  margin-left:auto;background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);
  color:var(--green);font-weight:700;font-size:13px;padding:8px 20px;border-radius:99px;white-space:nowrap;
}}

/* HW CARDS */
.hw-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:18px}}
.hw-card{{
  background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:22px;position:relative;overflow:hidden;
  transition:transform .2s,border-color .2s;
}}
.hw-card:hover{{transform:translateY(-3px)}}
.hw-best{{border-color:rgba(34,197,94,.35);box-shadow:0 0 35px rgba(34,197,94,.07)}}
.best-tag{{
  position:absolute;top:12px;right:12px;
  background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);
  color:var(--green);font-size:11px;font-weight:700;padding:3px 10px;border-radius:99px;
}}
.hw-top{{
  display:flex;align-items:center;gap:10px;padding-bottom:14px;
  margin-bottom:12px;border-bottom:1px solid;
}}
.hw-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.hw-name{{font-size:14px;font-weight:700;line-height:1.3}}
.verdict-pass{{
  background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);
  color:var(--green);border-radius:8px;padding:8px 12px;font-size:12px;font-weight:600;margin-bottom:12px;
}}
.verdict-fail{{
  background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.25);
  color:var(--amber);border-radius:8px;padding:8px 12px;font-size:12px;font-weight:600;margin-bottom:12px;
}}
.hw-big-num{{font-size:52px;font-weight:900;line-height:1;font-family:'JetBrains Mono',monospace}}
.unit{{font-size:20px;font-weight:400;color:var(--muted)}}
.hw-sub{{font-size:11px;color:var(--muted);margin-top:2px;margin-bottom:14px}}
.hw-need-box{{
  display:flex;align-items:center;gap:12px;
  background:var(--card2);border-radius:10px;padding:12px 14px;margin-bottom:14px;
}}
.hw-need-num{{font-size:34px;font-weight:900;font-family:'JetBrains Mono',monospace;color:var(--accent)}}
.hw-need-lbl{{font-size:11px;color:var(--muted);line-height:1.4}}
.stbl{{width:100%;border-collapse:collapse;margin-bottom:10px}}
.sk{{font-size:11px;color:var(--muted);padding:3px 0;width:38%}}
.sv{{font-size:11px;font-weight:500;font-family:'JetBrains Mono',monospace;padding:3px 0}}
.hw-note{{font-size:11px;color:var(--muted);font-style:italic;line-height:1.5}}

/* CHARTS */
.cg{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
@media(max-width:700px){{.cg{{grid-template-columns:1fr}}}}
.cc{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px}}
.cc h3{{font-size:12px;color:var(--muted);font-weight:600;margin-bottom:14px}}
.cw{{position:relative;height:240px}}

/* CLUSTER TABLE */
.tcard{{background:var(--card);border:1px solid var(--border);border-radius:12px;overflow-x:auto}}
.tbl{{width:100%;border-collapse:collapse;min-width:650px}}
.tbl thead tr{{background:var(--card2)}}
.tbl th{{padding:11px 14px;text-align:left;font-size:11px;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border);}}
.tbl td{{padding:10px 14px;font-size:13px;border-bottom:1px solid rgba(24,45,71,.5);
  font-family:'JetBrains Mono',monospace;}}
.tbl tr:last-child td{{border-bottom:none}}
.tbl tr:hover td{{background:rgba(59,130,246,.04)}}

/* NETWORK BOX */
.net-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px}}
.net-card{{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px}}
.net-title{{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}}
.net-val{{font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#93c5fd}}
.net-sub{{font-size:11px;color:var(--muted);margin-top:4px}}

/* FOOTER */
.footer{{text-align:center;margin-top:60px;color:var(--muted);font-size:12px;line-height:1.8}}
</style>
</head>
<body>

<div class="hero">
  <div class="c">
    <div class="tag">📡 4路 ESP32-CAM 並行 · 硬體需求分析</div>
    <h1>同時處理 4 路 ESP32-CAM<br>需要幾台設備？</h1>
    <div class="hero-sub">基於 YOLOv8n + ByteTrack 推論基準，模擬各硬體平台同時接收 4 路 MJPEG 串流並執行即時 AI 辨識的效能，計算最少需要幾台設備才能讓每路達到 ≥{TARGET_FPS} fps。</div>
    <div class="stat-row">
      <div class="stat-item"><div class="lbl">攝影機路數</div><div class="val">{N_CAMERAS} × ESP32-CAM</div></div>
      <div class="stat-item"><div class="lbl">每路目標 FPS</div><div class="val">≥ {TARGET_FPS} fps</div></div>
      <div class="stat-item"><div class="lbl">AI 模型</div><div class="val">YOLOv8n @ {IMGSZ}px</div></div>
      <div class="stat-item"><div class="lbl">x86 基準延遲</div><div class="val">{round(baseline_avg,1)} ms/幀</div></div>
    </div>
  </div>
</div>

<div class="c">

  <!-- 最佳方案提示 -->
  {'<div class="best-alert"><div class="best-alert-icon">🏆</div><div><h3>最低成本達標方案：' + best[0]["short"].replace(chr(10)," ") + '</h3><p>最少需要 <strong>' + str(best[1]["n_devices"]) + ' 台</strong>，可同時處理 4 路串流，每路平均 <strong>' + str(best[1]["fps_per_cam"]) + ' fps</strong>，總成本約 <strong>$' + str(best[1]["total_cost_usd"]) + ' USD</strong>，功耗 <strong>' + str(best[1]["total_power_w"]) + 'W</strong>。</p></div><div class="best-badge">最佳性價比 ✅</div></div>' if best else ''}

  <!-- HW CARDS -->
  <div class="sec">
    <div class="sec-title">各硬體 · 1台接4路的效能</div>
    <div class="hw-grid">{cards_html}</div>
  </div>

  <!-- 圖表 -->
  <div class="sec">
    <div class="sec-title">效能視覺化</div>
    <div class="cg">
      <div class="cc">
        <h3>⚡ 每路平均 FPS (1台處理4路，目標線 {TARGET_FPS}fps)</h3>
        <div class="cw"><canvas id="c1"></canvas></div>
      </div>
      <div class="cc">
        <h3>🏗 最少需要幾台才能讓4路全部達標</h3>
        <div class="cw"><canvas id="c2"></canvas></div>
      </div>
      <div class="cc">
        <h3>💰 達標最低總成本 (USD)</h3>
        <div class="cw"><canvas id="c3"></canvas></div>
      </div>
      <div class="cc">
        <h3>🔌 達標設備總功耗 (W)</h3>
        <div class="cw"><canvas id="c4"></canvas></div>
      </div>
    </div>
  </div>

  <!-- 集群方案明細 -->
  <div class="sec">
    <div class="sec-title">集群方案計算明細</div>
    <div class="tcard">
      <table class="tbl">
        <thead>
          <tr>
            <th>硬體</th><th>台數</th><th>每台幾路</th>
            <th>每路FPS</th><th>RAM需求</th>
            <th>總成本</th><th>總功耗</th><th>達標</th>
          </tr>
        </thead>
        <tbody>{cluster_table}</tbody>
      </table>
    </div>
  </div>

  <!-- 網路需求 -->
  <div class="sec">
    <div class="sec-title">📡 網路與頻寬需求 (4路 ESP32-CAM)</div>
    <div class="net-grid">
      <div class="net-card">
        <div class="net-title">每路 MJPEG 頻寬</div>
        <div class="net-val">{MJPEG_KBPS} KB/s</div>
        <div class="net-sub">640×480 @ {STREAM_FPS}fps，中品質壓縮</div>
      </div>
      <div class="net-card">
        <div class="net-title">4路總計頻寬</div>
        <div class="net-val">{N_CAMERAS * MJPEG_KBPS} KB/s</div>
        <div class="net-sub">≈ {round(N_CAMERAS * MJPEG_KBPS * 8 / 1000, 1)} Mbps，Wi-Fi 足夠</div>
      </div>
      <div class="net-card">
        <div class="net-title">建議 Wi-Fi 規格</div>
        <div class="net-val">Wi-Fi 4+</div>
        <div class="net-sub">802.11n @ 2.4GHz 已足夠，建議 5GHz 降低干擾</div>
      </div>
      <div class="net-card">
        <div class="net-title">每路解碼延遲</div>
        <div class="net-val">~20 ms</div>
        <div class="net-sub">MJPEG 解碼 + 網路傳輸延遲</div>
      </div>
      <div class="net-card">
        <div class="net-title">端對端總延遲</div>
        <div class="net-val">AI: + 延遲</div>
        <div class="net-sub">網路 20ms + 推論延遲 + ByteTrack</div>
      </div>
      <div class="net-card">
        <div class="net-title">mDNS 自動發現</div>
        <div class="net-val">已實作</div>
        <div class="net-sub">esp32-safety.local 自動偵測，無需手動填 IP</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  先行一步 (One Step Ahead) · 115年人本環境全國大專院校學生競賽<br>
  數據說明：FPS 由 x86 基準推論時間 ({round(baseline_avg,1)}ms) × 硬體縮放因子計算，Round-Robin 4路時間切片模型
</div>

<script>
const labels  = {json.dumps(chart_hw)};
const fps4    = {json.dumps(chart_fps)};
const nDev    = {json.dumps(chart_n)};
const costs   = {json.dumps(chart_cost)};
const powers  = {json.dumps([hw['power_w']*n for (hw,_),n in zip(hw_summaries,chart_n)])};
const colors  = {json.dumps(chart_colors)};

const bo = {{
  responsive:true, maintainAspectRatio:false,
  plugins:{{legend:{{display:false}},tooltip:{{
    backgroundColor:'#0d1b2e',borderColor:'#182d47',borderWidth:1,
    titleColor:'#e2e8f0',bodyColor:'#94a3b8',padding:12
  }}}},
  scales:{{
    x:{{ticks:{{color:'#64748b',font:{{size:10}}}},grid:{{color:'#182d47'}}}},
    y:{{ticks:{{color:'#64748b',font:{{size:10}}}},grid:{{color:'#182d47'}}}},
  }}
}};

new Chart(document.getElementById('c1'),{{type:'bar',data:{{
  labels,datasets:[
    {{label:'FPS/路',data:fps4,backgroundColor:colors.map(c=>c+'bb'),borderColor:colors,borderWidth:2,borderRadius:5}},
    {{label:'目標{TARGET_FPS}fps',data:labels.map(()=>{TARGET_FPS}),type:'line',borderColor:'#ffffff44',borderWidth:2,borderDash:[6,4],pointRadius:0,fill:false}}
  ]}},options:{{...bo,scales:{{...bo.scales,y:{{...bo.scales.y,min:0}}}}}}
}});

new Chart(document.getElementById('c2'),{{type:'bar',data:{{
  labels,datasets:[{{
    label:'最少台數',data:nDev,
    backgroundColor:nDev.map(n=>n===1?'#22c55ecc':n<=2?'#f59e0bcc':'#ef4444cc'),
    borderColor:nDev.map(n=>n===1?'#22c55e':n<=2?'#f59e0b':'#ef4444'),
    borderWidth:2,borderRadius:5
  }}]}},options:{{...bo,scales:{{...bo.scales,y:{{...bo.scales.y,min:0,ticks:{{...bo.scales.y.ticks,stepSize:1}}}}}}}}
}});

new Chart(document.getElementById('c3'),{{type:'bar',data:{{
  labels,datasets:[{{
    label:'USD',data:costs,
    backgroundColor:costs.map(c=>c<200?'#22c55ecc':c<400?'#f59e0bcc':'#ef4444cc'),
    borderWidth:2,borderRadius:5
  }}]}},options:{{...bo,scales:{{...bo.scales,y:{{...bo.scales.y,min:0}}}}}}
}});

new Chart(document.getElementById('c4'),{{type:'bar',data:{{
  labels,datasets:[{{
    label:'Watt',data:powers,
    backgroundColor:colors.map(c=>c+'99'),borderColor:colors,borderWidth:2,borderRadius:5
  }}]}},options:{{...bo,scales:{{...bo.scales,y:{{...bo.scales.y,min:0}}}}}}
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
    print(f"🎯 目標: {N_CAMERAS} 路 × ≥{TARGET_FPS} fps\n")

    print("📦 載入 YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print("✅ 就緒\n")

    print("⏱  測量 x86 基準推論時間...")
    baseline_ms = get_baseline(model, N_FRAMES)
    baseline_avg = float(np.mean(baseline_ms))
    print(f"✅ 基準: {round(baseline_avg, 2)} ms/幀  ({round(1000/baseline_avg,1)} fps)\n")

    print("=" * 65)
    print(f"  🔄 模擬 {N_CAMERAS} 路 Round-Robin 並行處理")
    print("=" * 65)

    hw_summaries = []
    all_cluster  = []

    for hw in HARDWARE_CONFIGS:
        sim     = simulate_single_device(hw, baseline_ms, N_CAMERAS)
        cluster = compute_cluster_need(hw, baseline_ms, N_CAMERAS)
        hw_summaries.append((hw, sim))
        all_cluster.append((hw, cluster))

        icon = "✅" if sim["meets_target"] else "⚠️"
        n_need = cluster[-1]["n_devices"]
        print(f"  {icon} {hw['short']:20s}: {sim['avg_per_cam_fps']:6.2f} fps/路  "
              f"→ 最少需 {n_need} 台  (${hw['price_usd']*n_need} USD)")

    print("=" * 65)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "stream4_hardware_report.html")
    generate_report(hw_summaries, all_cluster, baseline_avg, out)
    print(f"\n📄 報表：{out}")

    import webbrowser
    webbrowser.open(f"file:///{out.replace(os.sep, '/')}")
    print("🌐 已開啟報表")


if __name__ == "__main__":
    main()
