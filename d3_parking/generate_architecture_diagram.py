"""
Generates d3_parking/assets/system_architecture.png
End-to-end D3 Parking system architecture diagram.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── colour palette ──────────────────────────────────────────────────────────
C_CAM    = "#2196F3"   # blue — cameras
C_EDGE   = "#4CAF50"   # green — edge unit
C_MODEL  = "#8BC34A"   # light green — models inside edge
C_MCP    = "#FF9800"   # orange — MCP server
C_BOT    = "#9C27B0"   # purple — bots
C_USER   = "#F44336"   # red — user
C_ARROW  = "#90A4AE"   # grey — arrows
C_TEXT   = "#ECEFF1"   # near-white text
C_LABEL  = "#B0BEC5"   # subtitle grey
C_BG     = "#1E2229"   # box background

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.35):
    """Draw a rounded rectangle with centred label."""
    b = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=2, edgecolor=color,
        facecolor=C_BG, zorder=3
    )
    ax.add_patch(b)
    # colour bar on top
    bar = FancyBboxPatch(
        (x - w/2, y + h/2 - 0.32), w, 0.32,
        boxstyle=f"round,pad=0,rounding_size=0.1",
        linewidth=0, facecolor=color, zorder=4,
        clip_on=True
    )
    ax.add_patch(bar)
    ax.text(x, y + h/2 - 0.18, label,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", zorder=5, clip_on=True)
    if sublabel:
        ax.text(x, y - 0.08, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=C_LABEL, zorder=5, wrap=True, clip_on=True)

def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW, style="->",
          lw=1.8, rad=0.0):
    ax.annotate("",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color,
            lw=lw, connectionstyle=f"arc3,rad={rad}"
        ), zorder=2
    )
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.22, label,
                ha="center", va="center", fontsize=7,
                color=C_LABEL, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="#0f1117", ec="none", alpha=0.8))

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 — Cameras  (left side)
# ═══════════════════════════════════════════════════════════════════════════
for i, (cam_y, cam_label) in enumerate([(8.0, "IP Camera 1\n(PoE, 4MP)"),
                                         (5.5, "IP Camera 2\n(PoE, 4MP)"),
                                         (3.0, "IP Camera N\n(PoE, 4MP)")]):
    box(ax, 1.6, cam_y, 2.4, 1.3, C_CAM, f"Camera {i+1}" if i < 2 else "Camera N",
        sublabel="4MP · IR · ONVIF\nRTSP stream")

# dots between cam 2 and N
ax.text(1.6, 4.22, "⋮", ha="center", va="center", fontsize=18,
        color=C_LABEL, zorder=5)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 — Edge Compute Unit (centre-left)
# ═══════════════════════════════════════════════════════════════════════════
box(ax, 5.8, 5.5, 3.6, 6.8, C_EDGE, "Edge Compute Unit",
    sublabel="NVIDIA Jetson Orin Nano\nCUDA · TensorRT · 8 GB")

# sub-boxes inside edge unit
box(ax, 5.8, 7.2, 2.8, 1.1, C_MODEL, "Frame Capture",
    sublabel="RTSP → JPEG decode")
box(ax, 5.8, 5.7, 2.8, 1.1, C_MODEL, "Crop Extractor",
    sublabel="14 ROI polygons → 96×128 crops")
box(ax, 5.8, 4.15, 2.8, 1.3, C_MODEL, "ResNet18 Classifier",
    sublabel="Free / Occupied\nconf 0.00–0.23 | 0.83–1.00")
box(ax, 5.8, 2.65, 2.8, 1.0, C_MODEL, "YOLOv8n Detector",
    sublabel="mAP@0.5 = 0.951 (fallback)")

# internal arrows inside edge unit
arrow(ax, 5.8, 6.65, 5.8, 6.27, label="frame", color="#66BB6A")
arrow(ax, 5.8, 5.15, 5.8, 4.82, label="crops", color="#66BB6A")
arrow(ax, 5.8, 3.50, 5.8, 3.17, label="full frame", color="#66BB6A")


# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 — MCP Server (centre)
# ═══════════════════════════════════════════════════════════════════════════
box(ax, 10.5, 5.5, 3.4, 6.8, C_MCP, "MCP Server",
    sublabel="FastAPI · WebSocket\nPostgreSQL · Cloud / VPS")

box(ax, 10.5, 7.3,  2.6, 1.0, "#FFA726", "Occupancy State",
    sublabel="slot_id · free/occupied · conf")
box(ax, 10.5, 5.9,  2.6, 1.0, "#FFA726", "Reservation DB",
    sublabel="user · slot · duration · GPS")
box(ax, 10.5, 4.3,  2.6, 1.0, "#FFA726", "MCP Tool Registry",
    sublabel="get_slots · reserve · directions\ncancel · lot_status")
box(ax, 10.5, 2.7,  2.6, 1.0, "#FFA726", "WebSocket Broadcaster",
    sublabel="real-time push to bots")

# internal MCP arrows
arrow(ax, 10.5, 6.80, 10.5, 6.43, color="#FFB74D")
arrow(ax, 10.5, 5.38, 10.5, 4.83, color="#FFB74D")
arrow(ax, 10.5, 3.79, 10.5, 3.22, color="#FFB74D")

# ═══════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Messaging bots + user
# ═══════════════════════════════════════════════════════════════════════════
box(ax, 15.0, 8.0, 3.0, 1.4, C_BOT, "WhatsApp Bot",
    sublabel="Twilio Business API\nnl query → MCP call → reply")
box(ax, 15.0, 5.5, 3.0, 1.4, C_BOT, "Telegram Bot",
    sublabel="python-telegram-bot\ninline buttons · live location")
box(ax, 15.0, 3.0, 3.0, 1.4, "#00BCD4", "Web Dashboard",
    sublabel="Real-time lot map\nadmin controls")

# user box
box(ax, 15.0, 0.85, 3.0, 1.1, C_USER, "End User",
    sublabel="Mobile · WhatsApp / Telegram")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ARROWS between blocks
# ═══════════════════════════════════════════════════════════════════════════
# cameras → edge unit
for cam_y in [8.0, 5.5, 3.0]:
    arrow(ax, 2.81, cam_y, 3.99, cam_y, label="RTSP", color=C_CAM, lw=1.5)

# edge unit → MCP (state push)
arrow(ax, 7.61, 5.5, 8.79, 5.5,
      label="HTTPS POST\nslot_states JSON", color=C_EDGE, lw=2)

# MCP ↔ WhatsApp bot
arrow(ax, 12.22, 7.1, 13.49, 7.8,  label="tool response", color=C_MCP, lw=1.8, rad=-0.25)
arrow(ax, 13.49, 7.4, 12.22, 6.4,  label="MCP tool call",  color=C_BOT,  lw=1.8, rad=-0.25)

# MCP ↔ Telegram bot
arrow(ax, 12.22, 5.2, 13.49, 5.2, label="tool response", color=C_MCP, lw=1.8)
arrow(ax, 13.49, 5.8, 12.22, 5.8, label="MCP tool call",  color=C_BOT,  lw=1.8)

# MCP → Web dashboard (read-only)
arrow(ax, 12.22, 3.5, 13.49, 3.0, label="WebSocket\npush", color=C_MCP, lw=1.5, rad=0.2)

# bots ↔ user
arrow(ax, 15.0, 7.28, 15.0, 6.62,  label="reply + GPS", color=C_BOT, lw=1.5)
arrow(ax, 15.05, 4.79, 15.05, 3.72, label="reply + GPS", color=C_BOT, lw=1.5)
arrow(ax, 14.95, 3.72, 14.95, 4.79, label="query",        color=C_USER, lw=1.2)
arrow(ax, 14.95, 6.62, 14.95, 7.28, label="query",        color=C_USER, lw=1.2)

# user → WhatsApp (bottom to top connection)
arrow(ax, 15.0, 1.42, 15.0, 2.28, label="", color=C_USER, lw=1.2)

# ═══════════════════════════════════════════════════════════════════════════
# TITLE + LEGEND
# ═══════════════════════════════════════════════════════════════════════════
ax.text(9.0, 9.7, "D3 Parking — End-to-End System Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold",
        color=C_TEXT, zorder=6)
ax.text(9.0, 9.3,
        "Overhead camera → Edge inference (ResNet18 / YOLOv8n) → MCP Server → WhatsApp / Telegram → User",
        ha="center", va="center", fontsize=8.5, color=C_LABEL, zorder=6)

legend_items = [
    mpatches.Patch(color=C_CAM,   label="Camera Hardware"),
    mpatches.Patch(color=C_EDGE,  label="Edge Compute (Jetson)"),
    mpatches.Patch(color=C_MCP,   label="MCP Server (Cloud)"),
    mpatches.Patch(color=C_BOT,   label="Messaging Bots"),
    mpatches.Patch(color="#00BCD4", label="Web Dashboard"),
    mpatches.Patch(color=C_USER,  label="End User"),
]
ax.legend(handles=legend_items, loc="lower left",
          fontsize=8, framealpha=0.3,
          facecolor=C_BG, edgecolor=C_ARROW,
          labelcolor=C_TEXT, ncol=3,
          bbox_to_anchor=(0.01, 0.01))

plt.tight_layout(pad=0.3)
out = "d3_parking/assets/system_architecture.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
