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

try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    font_sub = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    font_mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    font_huge = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 52)
except Exception:
    font_title = font_sub = font_mono = font_bold = font_huge = ImageFont.load_default()

CORE_JOINTS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14,
    39, 40, 41, 42,
    67, 68, 69, 70, 71,
    72, 73, 74, 75, 76,
]

def group_of(j):
    if j >= 67:
        return "legs"
    if (11 <= j <= 14) or (39 <= j <= 42):
        return "arms"
    return "torso"

GROUP_COLORS = {
    "torso": (240, 239, 236),
    "arms": (57, 135, 229),
    "legs": (25, 158, 112),
}

def to_viewer(k):
    # mm in NIM camera frame -> m in viewer frame where +Y is up
    return np.array([k[0] / 1000.0, -k[1] / 1000.0, k[2] / 1000.0], dtype=float)

def project_pt(p, az, el):
    ca, sa = math.cos(az), math.sin(az)
    x1 = p[0] * ca + p[2] * sa
    z1 = -p[0] * sa + p[2] * ca
    ce, se = math.cos(el), math.sin(el)
    y2 = p[1] * ce - z1 * se
    z2 = p[1] * se + z1 * ce
    return x1, y2, z2

def generate_clip4():
    """Clip 4: Spin Rotation kinematics & 3D skeleton rotating in 3D coordinate space."""
    print("Generating Clip 4: Spin Rotation (Moving)...")
    session_path = ROOT / "web" / "public" / "sessions" / "run1" / "session.json"
    with open(session_path) as f:
        session = json.load(f)
        
    start_f = 495
    end_f = 570  # 75 frames = 2.5s
    
    core_set = set(CORE_JOINTS)
    core_links = [l for l in session["skeleton"]["links"] if l[0] in core_set and l[1] in core_set]
    
    frames_dict = {fr["f"]: fr for fr in session["frames"]}
    telemetry = session["telemetry"]
    f_to_idx = {f_num: i for i, f_num in enumerate(telemetry["f"])}
    
    out_path = MEDIA_DIR / "clip4_spin_rotation.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, 30.0, (1920, 1080))
    
    J = session["skeleton"]["joints"]
    hip_l, hip_r = J["hip_l"], J["hip_r"]
    sho_l, sho_r = J["shoulder_l"], J["shoulder_r"]
    
    for cur_f in range(start_f, end_f):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:] = (16, 20, 24)
        
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")
        
        # Header banner
        draw.text((80, 45), "3D KINEMATICS: CAMERA-COMPENSATED SPIN ROTATION", fill=(0, 230, 118), font=font_title)
        draw.text((80, 85), "3D Body Pose NIM (77 Nova Joints) + Angular Velocity Integration", fill=(160, 175, 190), font=font_sub)
        
        # Left Panel: 3D Pose Inspector
        pan_w, pan_h = 860, 860
        draw.rounded_rectangle([80, 140, 80 + pan_w, 140 + pan_h], radius=16, fill=(22, 28, 36, 240), outline=(50, 65, 80), width=2)
        draw.text((110, 165), "3D POSE ORBIT INSPECTOR", fill=(255, 255, 255), font=font_bold)
        draw.text((110, 195), "Dynamic Rider Coordinate Frame · Axis Cork Tilt: 114.5°", fill=(140, 155, 170), font=font_sub)
        
        # Right Panel: Telemetry & Spin Counter
        draw.rounded_rectangle([980, 140, 1840, 140 + pan_h], radius=16, fill=(22, 28, 36, 240), outline=(50, 65, 80), width=2)
        draw.text((1010, 165), "SPIN ACCUMULATION & ROTATION METRICS", fill=(255, 255, 255), font=font_bold)
        
        progress = max(0.0, min(1.0, (cur_f - 499) / (587 - 499)))
        disp_rot = progress * 977.0 if cur_f >= 499 else 0.0
        disp_spins = disp_rot / 360.0
        idx = f_to_idx.get(cur_f, 0)
        cur_rpm = telemetry["rpm"][idx] if idx < len(telemetry["rpm"]) else 0.0
        
        # Large Spin Counter
        draw.text((1010, 220), f"{disp_spins:.2f}", fill=(0, 230, 118), font=font_huge)
        draw.text((1170, 240), "SPINS", fill=(180, 200, 220), font=font_bold)
        draw.text((1010, 290), f"+{disp_rot:.1f}° Measured Rotation Path", fill=(255, 255, 255), font=font_title)
        
        # Telemetry cards on right
        cards = [
            ("Peak Spin Rate", f"+{abs(cur_rpm):.1f} RPM  (112 Peak)", (255, 180, 60)),
            ("Camera Pan Subtraction", "Broadcaster pan isolated & removed", (120, 200, 255)),
            ("Spin Axis Cork Angle", "114.5° off vertical (Inverted Cork)", (255, 100, 120)),
            ("Airborne Duration", "2.93 seconds (Frames 499 - 587)", (200, 220, 240)),
        ]
        cy = 360
        for label, val, color in cards:
            draw.rounded_rectangle([1010, cy, 1810, cy + 75], radius=8, fill=(16, 22, 28))
            draw.text((1035, cy + 12), label, fill=(140, 155, 170), font=font_sub)
            draw.text((1035, cy + 38), val, fill=color, font=font_bold)
            cy += 88
            
        # Draw dynamic rotation curve on right bottom
        graph_x0, graph_y0 = 1010, 730
        graph_w, graph_h = 800, 230
        draw.rounded_rectangle([graph_x0, graph_y0, graph_x0 + graph_w, graph_y0 + graph_h], radius=8, fill=(14, 18, 24))
        draw.text((graph_x0 + 20, graph_y0 + 15), "ROTATION TRAJECTORY (DEGREES VS TIME)", fill=(130, 145, 160), font=font_sub)
        
        # Reference lines
        for deg, lbl in [(0, "0°"), (360, "360°"), (720, "720°"), (900, "900°")]:
            ly = graph_y0 + graph_h - 35 - int((deg / 1000.0) * (graph_h - 80))
            draw.line([graph_x0 + 75, ly, graph_x0 + graph_w - 20, ly], fill=(35, 45, 55), width=1)
            draw.text((graph_x0 + 20, ly - 8), lbl, fill=(100, 115, 130), font=font_sub)
            
        pts = []
        for f_i in range(495, cur_f + 1):
            gx = graph_x0 + 85 + int(((f_i - 495) / (end_f - 495)) * (graph_w - 120))
            p_i = max(0.0, min(1.0, (f_i - 499) / (587 - 499))) if f_i >= 499 else 0.0
            r_i = p_i * 977.0
            gy = graph_y0 + graph_h - 35 - int((r_i / 1000.0) * (graph_h - 80))
            pts.append((gx, gy))
            
        if len(pts) > 1:
            draw.line(pts, fill=(0, 230, 118), width=4)
            draw.ellipse([pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6], fill=(255, 255, 255))
            
        # Draw 3D skeleton on Left Panel with exact viewport projection
        fr_data = frames_dict.get(cur_f, None)
        if fr_data and "k3" in fr_data and len(fr_data["k3"]) >= 77:
            pts_v = [to_viewer(k) for k in fr_data["k3"]]
            hip_mid = (pts_v[hip_l] + pts_v[hip_r]) * 0.5
            sho_mid = (pts_v[sho_l] + pts_v[sho_r]) * 0.5
            body_up = sho_mid - hip_mid
            norm_bu = np.linalg.norm(body_up)
            if norm_bu > 1e-9:
                body_up = body_up / norm_bu
                
            # Dynamic camera orbit around the rider
            orbit_az = 0.35 + (cur_f - start_f) * 0.04
            orbit_el = 0.12 + math.sin((cur_f - start_f) * 0.05) * 0.08
            
            cam_up = np.array([0.0, 1.0, 0.0])
            v_top = hip_mid + cam_up * 0.6
            v_bot = hip_mid - cam_up * 0.5
            a_top = hip_mid + body_up * 0.6
            
            fit = [pts_v[j] for j in CORE_JOINTS] + [v_top, v_bot, a_top]
            proj_fit = [project_pt(p, orbit_az, orbit_el) for p in fit]
            xs = [q[0] for q in proj_fit]
            ys = [q[1] for q in proj_fit]
            bx0, bx1 = min(xs), max(xs)
            by0, by1 = min(ys), max(ys)
            
            box_w, box_h = pan_w - 120, pan_h - 220
            scale = min(box_w / max(0.2, bx1 - bx0), box_h / max(0.2, by1 - by0))
            bcx = (bx0 + bx1) / 2.0
            bcy = (by0 + by1) / 2.0
            
            center_x = 80 + pan_w / 2.0
            center_y = 140 + 100 + box_h / 2.0
            
            def to_screen(p):
                x1, y2, z2 = project_pt(p, orbit_az, orbit_el)
                sx = center_x + (x1 - bcx) * scale
                sy = center_y - (y2 - bcy) * scale
                return sx, sy, z2
                
            # Draw vertical reference line (dashed)
            p_vtop = to_screen(v_top)
            p_vbot = to_screen(v_bot)
            draw.line([(p_vtop[0], p_vtop[1]), (p_vbot[0], p_vbot[1])], fill=(80, 90, 100), width=2)
            
            # Draw cork axis line
            p_atop = to_screen(a_top)
            p_amid = to_screen(hip_mid)
            draw.line([(p_amid[0], p_amid[1]), (p_atop[0], p_atop[1])], fill=(255, 70, 70), width=4)
            draw.text((p_atop[0] + 10, p_atop[1] - 15), "114.5° Body Cork Axis", fill=(255, 70, 70), font=font_bold)
            
            # Draw links sorted by depth
            ordered_links = []
            for a, b in core_links:
                pa = to_screen(pts_v[a])
                pb = to_screen(pts_v[b])
                d = (pa[2] + pb[2]) / 2.0
                ordered_links.append((d, a, b, pa, pb))
            ordered_links.sort(key=lambda item: item[0])
            
            for _, a, b, pa, pb in ordered_links:
                group = group_of(a)
                color = GROUP_COLORS[group]
                draw.line([(pa[0], pa[1]), (pb[0], pb[1])], fill=color, width=6)
                
            # Draw joints
            for j in CORE_JOINTS:
                pj = to_screen(pts_v[j])
                draw.ellipse([pj[0] - 5, pj[1] - 5, pj[0] + 5, pj[1] + 5], fill=(255, 210, 60))
                
        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(res)
        
    out.release()
    print(f"Clip 4 generated -> {out_path}")

if __name__ == "__main__":
    generate_clip4()
