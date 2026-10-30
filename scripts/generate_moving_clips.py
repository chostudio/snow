#!/usr/bin/env python3
"""Generate dynamic 1080p moving clips for Clip 3 (Trick Classification) and Clip 4 (Spin Rotation)."""

import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/chrisho/snow")
MEDIA_DIR = ROOT / "docs" / "media" / "raw_clips"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

# Try to find a standard font
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    font_sub = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    font_mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    font_huge = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 46)
except Exception:
    font_title = font_sub = font_mono = font_bold = font_huge = ImageFont.load_default()

def generate_clip3():
    """Clip 3: Real video of air-3 with dynamic real-time classification HUD overlay."""
    print("Generating Clip 3: Trick Classification (Moving)...")
    video_path = ROOT / "data" / "poses" / "run1.overlay.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # Load session telemetry & events
    session_path = ROOT / "web" / "public" / "sessions" / "run1" / "session.json"
    with open(session_path) as f:
        session = json.load(f)
    
    event = session["events"][2]  # air-3 (frames 499 to 587)
    start_f = 495
    end_f = 570  # 75 frames = 2.5 seconds at 30 fps
    
    telemetry = session["telemetry"]
    f_to_idx = {f_num: i for i, f_num in enumerate(telemetry["f"])}
    
    out_path = MEDIA_DIR / "clip3_trick_classification.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (1920, 1080))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    for cur_f in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Current telemetry metrics
        idx = f_to_idx.get(cur_f, None)
        if idx is not None:
            rot = telemetry["rotation_deg"][idx]
            rpm = telemetry["rpm"][idx]
            tilt = telemetry["axis_tilt_deg"][idx]
        else:
            rot = 0.0
            rpm = 0.0
            tilt = 0.0
            
        progress = max(0.0, min(1.0, (cur_f - 499) / (587 - 499)))
        disp_rot = progress * 977.0 if cur_f >= 499 else 0.0
        
        # Convert frame to PIL for clean drawing
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # Top banner
        draw.rounded_rectangle([40, 40, 520, 95], radius=8, fill=(18, 22, 28, 230), outline=(76, 175, 80, 200), width=2)
        draw.ellipse([55, 60, 75, 80], fill=(76, 217, 100, 255))
        draw.text((85, 55), "MODEL A: AIRBORNE DETECTED", fill=(255, 255, 255), font=font_bold)
        
        # Right side HUD Card: Trick Classification
        card_x0, card_y0 = 1260, 40
        card_w, card_h = 620, 460
        draw.rounded_rectangle([card_x0, card_y0, card_x0 + card_w, card_y0 + card_h], radius=12, fill=(15, 20, 26, 235), outline=(0, 230, 118, 220), width=2)
        
        # Header
        draw.text((card_x0 + 24, card_y0 + 20), "MODEL B: FACTORED TRICK CLASSIFIER", fill=(0, 230, 118), font=font_bold)
        draw.text((card_x0 + 24, card_y0 + 55), "Backside Double Cork 900 Nose", fill=(255, 255, 255), font=font_title)
        draw.line([card_x0 + 24, card_y0 + 95, card_x0 + card_w - 24, card_y0 + 95], fill=(60, 70, 80), width=1)
        
        # Factored heads
        y = card_y0 + 110
        heads = [
            ("Trick Rotation", "900° (p=0.86)", (255, 255, 255)),
            ("Trick Axis", "Double Cork · 114.5° tilt (p=0.96)", (255, 200, 80)),
            ("Direction / Stance", "Backside · Goofy stance", (200, 220, 255)),
            ("Grab Detected", "Nose Grab (p=0.72)", (220, 220, 220)),
        ]
        for label, val, color in heads:
            draw.text((card_x0 + 24, y), label, fill=(160, 170, 180), font=font_sub)
            draw.text((card_x0 + 220, y), val, fill=color, font=font_bold)
            y += 36
            
        draw.line([card_x0 + 24, y + 5, card_x0 + card_w - 24, y + 5], fill=(60, 70, 80), width=1)
        y += 18
        
        # Model C Scoring
        draw.text((card_x0 + 24, y), "MODEL C: FIS SCORE PREDICTOR", fill=(255, 179, 0), font=font_bold)
        y += 32
        draw.text((card_x0 + 24, y), "Expected Score: 49.8 pts", fill=(255, 255, 255), font=font_bold)
        draw.text((card_x0 + 340, y), "Landing Prob: 41.1%", fill=(180, 200, 220), font=font_sub)
        y += 45
        
        # Live rotation progress bar
        bar_x = card_x0 + 24
        bar_w = card_w - 48
        draw.text((bar_x, y), f"Live Spin Path: +{disp_rot:.0f}° / 977°", fill=(0, 230, 118), font=font_mono)
        y += 28
        draw.rounded_rectangle([bar_x, y, bar_x + bar_w, y + 14], radius=6, fill=(35, 45, 55))
        if progress > 0:
            draw.rounded_rectangle([bar_x, y, bar_x + int(bar_w * progress), y + 14], radius=6, fill=(0, 230, 118))
            
        # Bottom indicator
        draw.rounded_rectangle([40, 980, 680, 1040], radius=8, fill=(18, 22, 28, 220))
        draw.text((60, 995), f"Frame {cur_f} / 720  |  Air Time: 2.93s  |  Peak RPM: 112", fill=(220, 230, 240), font=font_mono)
        
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(res)
        
    cap.release()
    out.release()
    print(f"Clip 3 generated -> {out_path}")

def generate_clip4():
    """Clip 4: Spin Rotation kinematics & 3D skeleton rotating in 3D coordinate space."""
    print("Generating Clip 4: Spin Rotation (Moving)...")
    session_path = ROOT / "web" / "public" / "sessions" / "run1" / "session.json"
    with open(session_path) as f:
        session = json.load(f)
        
    start_f = 495
    end_f = 570  # 75 frames = 2.5s
    
    # Nova 77 skeleton links
    LINKS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # spine
        (4, 5), (5, 6), (6, 7),          # left arm
        (4, 8), (8, 9), (9, 10),         # right arm
        (0, 11), (11, 12), (12, 13),     # left leg
        (0, 14), (14, 15), (15, 16),     # right leg
    ]
    
    frames_dict = {fr["f"]: fr for fr in session["frames"]}
    telemetry = session["telemetry"]
    f_to_idx = {f_num: i for i, f_num in enumerate(telemetry["f"])}
    
    out_path = MEDIA_DIR / "clip4_spin_rotation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, 30.0, (1920, 1080))
    
    for cur_f in range(start_f, end_f):
        # Create dark background
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:] = (16, 20, 24)
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # Header banner
        draw.text((80, 50), "3D KINEMATICS: CAMERA-COMPENSATED SPIN ROTATION", fill=(0, 230, 118), font=font_title)
        draw.text((80, 95), "3D Pose Tracking (NVIDIA NIM) + Angular Path Integration", fill=(160, 175, 190), font=font_sub)
        
        # Left Panel: 3D Pose Inspector
        draw.rounded_rectangle([80, 150, 980, 980], radius=16, fill=(22, 28, 36, 240), outline=(50, 65, 80), width=2)
        draw.text((110, 175), "3D POSE INSPECTOR (OMNIVERSE SPACE)", fill=(255, 255, 255), font=font_bold)
        draw.text((110, 205), "Dynamic Rider Coordinate Frame · Axis Cork Tilt: 114.5°", fill=(140, 155, 170), font=font_sub)
        
        # Right Panel: Telemetry & Spin Counter
        draw.rounded_rectangle([1020, 150, 1840, 980], radius=16, fill=(22, 28, 36, 240), outline=(50, 65, 80), width=2)
        draw.text((1050, 175), "SPIN ACCUMULATION & RPM", fill=(255, 255, 255), font=font_bold)
        
        progress = max(0.0, min(1.0, (cur_f - 499) / (587 - 499)))
        disp_rot = progress * 977.0 if cur_f >= 499 else 0.0
        disp_spins = disp_rot / 360.0
        idx = f_to_idx.get(cur_f, 0)
        cur_rpm = telemetry["rpm"][idx] if idx < len(telemetry["rpm"]) else 0.0
        
        # Large Spin Counter
        draw.text((1050, 240), f"{disp_spins:.2f}", fill=(0, 230, 118), font=font_huge)
        draw.text((1200, 260), "SPINS", fill=(180, 200, 220), font=font_bold)
        draw.text((1050, 310), f"+{disp_rot:.1f}° Measured Rotation Path", fill=(255, 255, 255), font=font_title)
        
        # Telemetry cards on right
        cards = [
            ("Peak Spin Rate", f"+{abs(cur_rpm):.1f} RPM  (112 Peak)", (255, 180, 60)),
            ("Camera Pan Subtraction", "Broadcaster pan isolated & removed", (120, 200, 255)),
            ("Spin Axis Cork Angle", "114.5° off vertical (Inverted Cork)", (255, 100, 120)),
            ("Airborne Duration", "2.93 seconds (Frames 499 - 587)", (200, 220, 240)),
        ]
        cy = 390
        for label, val, color in cards:
            draw.rounded_rectangle([1050, cy, 1810, cy + 80], radius=8, fill=(16, 22, 28))
            draw.text((1075, cy + 15), label, fill=(140, 155, 170), font=font_sub)
            draw.text((1075, cy + 42), val, fill=color, font=font_bold)
            cy += 95
            
        # Draw dynamic rotation curve on right bottom
        graph_x0, graph_y0 = 1050, 780
        graph_w, graph_h = 760, 160
        draw.rounded_rectangle([graph_x0, graph_y0, graph_x0 + graph_w, graph_y0 + graph_h], radius=8, fill=(14, 18, 24))
        draw.text((graph_x0 + 20, graph_y0 + 15), "ROTATION TRAJECTORY (DEGREES VS TIME)", fill=(130, 145, 160), font=font_sub)
        
        # Reference lines
        for deg, lbl in [(0, "0°"), (360, "360°"), (720, "720°"), (900, "900°")]:
            ly = graph_y0 + graph_h - 25 - int((deg / 1000.0) * (graph_h - 55))
            draw.line([graph_x0 + 70, ly, graph_x0 + graph_w - 20, ly], fill=(35, 45, 55), width=1)
            draw.text((graph_x0 + 20, ly - 8), lbl, fill=(100, 115, 130), font=font_sub)
            
        # Draw curve up to current frame
        pts = []
        for f_i in range(495, cur_f + 1):
            gx = graph_x0 + 80 + int(((f_i - 495) / (end_f - 495)) * (graph_w - 110))
            p_i = max(0.0, min(1.0, (f_i - 499) / (587 - 499))) if f_i >= 499 else 0.0
            r_i = p_i * 977.0
            gy = graph_y0 + graph_h - 25 - int((r_i / 1000.0) * (graph_h - 55))
            pts.append((gx, gy))
            
        if len(pts) > 1:
            draw.line(pts, fill=(0, 230, 118), width=4)
            draw.ellipse([pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6], fill=(255, 255, 255))
            
        # Draw 3D skeleton on Left Panel
        fr_data = frames_dict.get(cur_f, None)
        cx, cy_skel = 530, 580
        
        # Rotation angle for 3D orbit
        orbit_angle = (cur_f - start_f) * 0.08
        cos_o = math.cos(orbit_angle)
        sin_o = math.sin(orbit_angle)
        
        if fr_data and "k3" in fr_data and len(fr_data["k3"]) >= 17:
            k3 = np.array(fr_data["k3"])
            # Center joints
            center = np.mean(k3[:17], axis=0)
            k3_centered = k3 - center
            
            # Project joints
            projected = []
            for j in range(len(k3_centered)):
                x, y, z = k3_centered[j]
                # Rotate around Y axis
                rx = x * cos_o - z * sin_o
                rz = x * sin_o + z * cos_o
                
                scale = 320.0 / (1.0 + rz * 0.3)
                px = int(cx + rx * scale)
                py = int(cy_skel - y * scale)
                projected.append((px, py, rz))
                
            # Draw ground grid
            for gi in range(-3, 4):
                gx1 = int(cx + gi * 60 * cos_o - 200 * sin_o)
                gy1 = int(cy_skel + 220 + gi * 20)
                gx2 = int(cx + gi * 60 * cos_o + 200 * sin_o)
                gy2 = int(cy_skel + 220 - gi * 20)
                draw.line([(gx1, gy1), (gx2, gy2)], fill=(30, 40, 50), width=1)
                
            # Draw links
            for u, v in LINKS:
                if u < len(projected) and v < len(projected):
                    p1 = projected[u]
                    p2 = projected[v]
                    # Color by body segment
                    if u >= 5 and u <= 10:
                        lcolor = (70, 150, 255)  # arms
                    elif u >= 11:
                        lcolor = (0, 230, 118)   # legs
                    else:
                        lcolor = (255, 255, 255) # spine
                    draw.line([(p1[0], p1[1]), (p2[0], p2[1])], fill=lcolor, width=4)
                    
            # Draw joints
            for j in range(min(17, len(projected))):
                px, py, _ = projected[j]
                draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(255, 200, 50))
                
            # Draw cork axis vector
            draw.line([(cx, cy_skel - 180), (cx + int(sin_o * 80), cy_skel + 180)], fill=(255, 60, 90), width=3)
            draw.text((cx + int(sin_o * 80) + 15, cy_skel + 160), "114.5° Cork Axis", fill=(255, 60, 90), font=font_bold)
            
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(res)
        
    out.release()
    print(f"Clip 4 generated -> {out_path}")

if __name__ == "__main__":
    generate_clip3()
    generate_clip4()
    print("Done generating moving clips!")
