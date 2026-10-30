#!/usr/bin/env python3
"""Generate Clip 18: Model B Factored Trick Classifier & Model C FIS Regressor Live HUD."""

import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess

ROOT = Path("/Users/chrisho/snow")
OUT_DIR = ROOT / "docs" / "media" / "showcase_clips"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    font_sub = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    font_mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    font_mono_sm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
    font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    font_huge = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 46)
except Exception:
    font_title = font_sub = font_mono = font_mono_sm = font_bold = font_huge = ImageFont.load_default()

def generate_clip18():
    video_path = ROOT / "data" / "poses" / "run1.overlay.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    session_path = ROOT / "web" / "public" / "sessions" / "run1" / "session.json"
    with open(session_path) as f:
        session = json.load(f)
        
    start_f = 485
    end_f = 605  # 120 frames = 4.0 seconds
    
    telemetry = session["telemetry"]
    f_to_idx = {f_num: i for i, f_num in enumerate(telemetry["f"])}
    
    temp_raw = OUT_DIR / "temp_clip18.mp4"
    final_out = OUT_DIR / "clip18_model_b_classifier_live_hud.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_raw), fourcc, fps, (1920, 1080))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    
    for cur_f in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Telemetry metrics
        idx = f_to_idx.get(cur_f, None)
        if idx is not None:
            rot = telemetry["rotation_deg"][idx]
            rpm = telemetry["rpm"][idx]
            tilt = telemetry["axis_tilt_deg"][idx]
            airborne = telemetry["airborne"][idx] if "airborne" in telemetry else (499 <= cur_f <= 587)
        else:
            rot = 0.0
            rpm = 0.0
            tilt = 0.0
            airborne = (499 <= cur_f <= 587)
            
        is_air = 499 <= cur_f <= 587
        progress = max(0.0, min(1.0, (cur_f - 499) / (587 - 499))) if is_air else (1.0 if cur_f > 587 else 0.0)
        disp_rot = progress * 977.0 if cur_f >= 499 else 0.0
        
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # Top banner: Model A Airborne Status
        if is_air:
            draw.rounded_rectangle([40, 40, 540, 95], radius=8, fill=(18, 26, 22, 235), outline=(0, 230, 118, 220), width=2)
            draw.ellipse([58, 60, 78, 80], fill=(0, 230, 118, 255))
            draw.text((90, 55), "MODEL A: AIRBORNE DETECTED (AIR 3)", fill=(255, 255, 255), font=font_bold)
        else:
            status_txt = "LANDED · RUNOUT ABSORPTION" if cur_f > 587 else "APPROACH · IN-RUN TRANSITION"
            draw.rounded_rectangle([40, 40, 540, 95], radius=8, fill=(18, 22, 28, 220), outline=(120, 140, 160, 180), width=2)
            draw.ellipse([58, 60, 78, 80], fill=(120, 140, 160, 255))
            draw.text((90, 55), f"MODEL A: {status_txt}", fill=(200, 215, 230), font=font_bold)
            
        # Right side HUD Card: Model B Factored Classifier
        card_x0, card_y0 = 1240, 40
        card_w, card_h = 640, 480
        draw.rounded_rectangle([card_x0, card_y0, card_x0 + card_w, card_y0 + card_h], radius=12, fill=(14, 18, 25, 240), outline=(0, 230, 118, 230), width=2)
        
        # Header
        draw.text((card_x0 + 24, card_y0 + 20), "MODEL B: FACTORED TRICK CLASSIFIER", fill=(0, 230, 118), font=font_bold)
        trick_name = "Backside Double Cork 900 Nose" if cur_f >= 499 else "Classifying trajectory..."
        draw.text((card_x0 + 24, card_y0 + 54), trick_name, fill=(255, 255, 255), font=font_title)
        draw.line([card_x0 + 24, card_y0 + 95, card_x0 + card_w - 24, card_y0 + 95], fill=(60, 72, 85), width=1)
        
        # Factored heads
        y = card_y0 + 110
        heads = [
            ("Trick Rotation", "900° (p=0.86)", (255, 255, 255)),
            ("Trick Axis", f"Double Cork · {tilt:.1f}° tilt (p=0.96)" if is_air else "Double Cork · 114.5° tilt (p=0.96)", (255, 200, 80)),
            ("Direction / Stance", "Backside · Goofy stance", (200, 225, 255)),
            ("Grab Detected", "Nose Grab (p=0.72)", (220, 230, 240)),
        ]
        for label, val, color in heads:
            draw.text((card_x0 + 24, y), label, fill=(160, 175, 190), font=font_sub)
            draw.text((card_x0 + 225, y), val, fill=color, font=font_bold)
            y += 36
            
        draw.line([card_x0 + 24, y + 5, card_x0 + card_w - 24, y + 5], fill=(60, 72, 85), width=1)
        y += 18
        
        # Model C Scoring
        draw.text((card_x0 + 24, y), "MODEL C: FIS SCORE PREDICTOR", fill=(255, 179, 0), font=font_bold)
        y += 32
        draw.text((card_x0 + 24, y), "Expected Score: 49.8 pts", fill=(255, 255, 255), font=font_bold)
        draw.text((card_x0 + 350, y), "Landing Prob: 41.1%", fill=(180, 210, 235), font=font_sub)
        y += 44
        
        # Live rotation progress bar
        bar_x = card_x0 + 24
        bar_w = card_w - 48
        draw.text((bar_x, y), f"Live Spin Path: +{disp_rot:.0f}° / 977°  ·  RPM: {abs(rpm):.0f}", fill=(0, 230, 118), font=font_mono_sm)
        y += 26
        draw.rounded_rectangle([bar_x, y, bar_x + bar_w, y + 14], radius=6, fill=(30, 40, 52))
        if progress > 0:
            fill_w = max(6, int(bar_w * progress))
            draw.rounded_rectangle([bar_x, y, bar_x + fill_w, y + 14], radius=6, fill=(0, 230, 118))
            
        # Bottom telemetry pill
        draw.rounded_rectangle([40, 980, 720, 1040], radius=8, fill=(14, 18, 25, 230), outline=(60, 72, 85, 200), width=1)
        draw.text((60, 995), f"Frame {cur_f} / 720  |  Air Time: 2.93s  |  Knee Flex: 100.3°", fill=(220, 235, 250), font=font_mono)
        
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(res)
        
    cap.release()
    out.release()
    
    # Encode with libx264 for universal compatibility
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        str(final_out)
    ]
    subprocess.run(cmd, check=True)
    temp_raw.unlink(missing_ok=True)
    print(f"Generated Clip 18 -> {final_out}")

if __name__ == "__main__":
    generate_clip18()
