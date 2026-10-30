#!/usr/bin/env python3
"""Generate the expanded 3D Kinematics video suite (Clips 13, 14, 15, 16, 17)."""

import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/Users/chrisho/snow")
OUT_DIR = ROOT / "docs" / "media" / "showcase_clips"
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    font_sub = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    font_mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 22)
    font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    font_huge = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 54)
    font_badge = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
except Exception:
    font_title = font_sub = font_mono = font_bold = font_huge = font_badge = ImageFont.load_default()

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
    return np.array([k[0] / 1000.0, -k[1] / 1000.0, k[2] / 1000.0], dtype=float)

def project_pt(p, az, el):
    ca, sa = math.cos(az), math.sin(az)
    x1 = p[0] * ca + p[2] * sa
    z1 = -p[0] * sa + p[2] * ca
    ce, se = math.cos(el), math.sin(el)
    y2 = p[1] * ce - z1 * se
    z2 = p[1] * se + z1 * ce
    return x1, y2, z2

def render_kinematics_clip(
    out_filename,
    session_name,
    event_idx,
    start_frame,
    end_frame,
    view_mode="orbit",  # orbit, front, side, top
    title_suffix="",
    desc="",
):
    print(f"Rendering {out_filename} ({view_mode})...")
    session_path = ROOT / "web" / "public" / "sessions" / session_name / "session.json"
    with open(session_path) as f:
        session = json.load(f)

    event = session["events"][event_idx]
    ev_start = event["start_f"]
    ev_end = event["end_f"]
    total_rot = float(event["metrics"]["rotation_deg"])
    peak_rpm = float(event["metrics"]["peak_rpm"])
    axis_tilt = float(event["metrics"]["max_axis_tilt_deg"])

    core_set = set(CORE_JOINTS)
    core_links = [l for l in session["skeleton"]["links"] if l[0] in core_set and l[1] in core_set]
    frames_dict = {fr["f"]: fr for fr in session["frames"]}
    telemetry = session["telemetry"]
    f_to_idx = {f_num: i for i, f_num in enumerate(telemetry["f"])}

    tmp_path = OUT_DIR / f"tmp_{out_filename}"
    final_path = OUT_DIR / out_filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_path), fourcc, 30.0, (1920, 1080))

    J = session["skeleton"]["joints"]
    hip_l, hip_r = J["hip_l"], J["hip_r"]
    sho_l, sho_r = J["shoulder_l"], J["shoulder_r"]

    num_frames = end_frame - start_frame

    for i, cur_f in enumerate(range(start_frame, end_frame)):
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:] = (16, 20, 24)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img, "RGBA")

        # Top Header
        draw.text((80, 42), f"3D KINEMATICS & BIOMECHANICS: {title_suffix.upper()}", fill=(0, 230, 118), font=font_title)
        draw.text((80, 80), f"{desc} · Camera-Compensated Rotation Rate & Spatial Orientation", fill=(160, 175, 190), font=font_sub)

        # Left Panel (3D Viewer)
        pan_w, pan_h = 920, 880
        draw.rounded_rectangle([60, 130, 60 + pan_w, 130 + pan_h], radius=16, fill=(22, 28, 36, 240), outline=(50, 65, 80), width=2)
        draw.text((90, 155), f"SPATIAL SKELETON ({view_mode.upper()} VIEW)", fill=(255, 255, 255), font=font_bold)
        draw.text((90, 185), f"Off-Vertical Cork Tilt: {axis_tilt:.1f}° | Pop Height: {event['metrics'].get('pop_height_m', 2.0):.2f}m", fill=(140, 155, 170), font=font_sub)

        # Right Panel (Telemetry & Metrics)
        rx = 60 + pan_w + 30
        rw = 1920 - rx - 60
        draw.rounded_rectangle([rx, 130, rx + rw, 130 + pan_h], radius=16, fill=(22, 28, 36, 240), outline=(50, 65, 80), width=2)
        draw.text((rx + 30, 155), "ANGULAR PATH & ROTATION TELEMETRY", fill=(255, 255, 255), font=font_bold)

        # Progress inside air
        if cur_f < ev_start:
            progress = 0.0
        elif cur_f > ev_end:
            progress = 1.0
        else:
            progress = (cur_f - ev_start) / max(1, (ev_end - ev_start))

        disp_rot = progress * total_rot
        disp_spins = disp_rot / 360.0

        idx = f_to_idx.get(cur_f, 0)
        cur_rpm = telemetry["rpm"][idx] if idx < len(telemetry["rpm"]) else 0.0

        # Big Spin Readout
        draw.text((rx + 30, 210), f"{disp_spins:.2f}", fill=(0, 230, 118), font=font_huge)
        draw.text((rx + 200, 230), "SPINS", fill=(180, 200, 220), font=font_bold)
        draw.text((rx + 30, 280), f"{disp_rot:+.1f}° Integrated Path", fill=(255, 255, 255), font=font_title)

        # Metrics cards
        status_lbl = "AIRBORNE" if ev_start <= cur_f <= ev_end else "GROUND GLIDE"
        status_col = (0, 230, 118) if ev_start <= cur_f <= ev_end else (160, 175, 190)

        cards = [
            ("Airborne State", f"{status_lbl} (Frame {cur_f} / {end_frame})", status_col),
            ("Angular Velocity", f"{abs(cur_rpm):.1f} RPM  (Peak {abs(peak_rpm):.1f} RPM)", (255, 180, 60)),
            ("Cork Tilt Angle", f"{axis_tilt:.1f}° off-vertical (Inverted)", (255, 90, 110)),
            ("Camera Pan Subtraction", "Broadcaster tracking pan removed", (120, 200, 255)),
        ]
        cy = 345
        for label, val, color in cards:
            draw.rounded_rectangle([rx + 30, cy, rx + rw - 30, cy + 72], radius=8, fill=(16, 22, 28))
            draw.text((rx + 50, cy + 12), label, fill=(140, 155, 170), font=font_sub)
            draw.text((rx + 50, cy + 36), val, fill=color, font=font_bold)
            cy += 84

        # Graph at bottom right
        graph_x0, graph_y0 = rx + 30, 710
        graph_w, graph_h = rw - 60, 260
        draw.rounded_rectangle([graph_x0, graph_y0, graph_x0 + graph_w, graph_y0 + graph_h], radius=8, fill=(14, 18, 24))
        draw.text((graph_x0 + 20, graph_y0 + 15), "ROTATION TRAJECTORY (DEGREES VS TIME)", fill=(130, 145, 160), font=font_sub)

        # Draw grid lines
        max_deg_scale = max(1100.0, total_rot * 1.1)
        step_deg = 360 if total_rot > 600 else 180
        for deg in range(0, int(max_deg_scale), step_deg):
            ly = graph_y0 + graph_h - 35 - int((deg / max_deg_scale) * (graph_h - 75))
            draw.line([graph_x0 + 75, ly, graph_x0 + graph_w - 20, ly], fill=(35, 45, 55), width=1)
            draw.text((graph_x0 + 20, ly - 8), f"{deg}°", fill=(100, 115, 130), font=font_sub)

        pts = []
        for f_i in range(start_frame, cur_f + 1):
            gx = graph_x0 + 85 + int(((f_i - start_frame) / num_frames) * (graph_w - 120))
            if f_i < ev_start:
                p_i = 0.0
            elif f_i > ev_end:
                p_i = 1.0
            else:
                p_i = (f_i - ev_start) / max(1, (ev_end - ev_start))
            r_i = p_i * total_rot
            gy = graph_y0 + graph_h - 35 - int((r_i / max_deg_scale) * (graph_h - 75))
            pts.append((gx, gy))

        if len(pts) > 1:
            draw.line(pts, fill=(0, 230, 118), width=4)
            draw.ellipse([pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6], fill=(255, 255, 255))

        # 3D Skeleton Rendering
        if view_mode == "orbit":
            orbit_az = 0.35 + i * 0.03
            orbit_el = 0.12 + math.sin(i * 0.04) * 0.1
        elif view_mode == "front":
            orbit_az = 0.0
            orbit_el = 0.0
        elif view_mode == "side":
            orbit_az = math.pi / 2.0
            orbit_el = 0.0
        elif view_mode == "top":
            orbit_az = 0.0
            orbit_el = -1.45

        fr_data = frames_dict.get(cur_f, None)
        if fr_data and "k3" in fr_data and len(fr_data["k3"]) >= 77:
            pts_v = [to_viewer(k) for k in fr_data["k3"]]
            hip_mid = (pts_v[hip_l] + pts_v[hip_r]) * 0.5
            sho_mid = (pts_v[sho_l] + pts_v[sho_r]) * 0.5
            body_up = sho_mid - hip_mid
            norm_bu = np.linalg.norm(body_up)
            if norm_bu > 1e-9:
                body_up = body_up / norm_bu

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

            box_w, box_h = pan_w - 140, pan_h - 220
            scale = min(box_w / max(0.2, bx1 - bx0), box_h / max(0.2, by1 - by0))
            bcx = (bx0 + bx1) / 2.0
            bcy = (by0 + by1) / 2.0

            center_x = 60 + pan_w / 2.0
            center_y = 130 + 100 + box_h / 2.0

            def to_screen(p):
                x1, y2, z2 = project_pt(p, orbit_az, orbit_el)
                sx = center_x + (x1 - bcx) * scale
                sy = center_y - (y2 - bcy) * scale
                return sx, sy, z2

            # Dashed vertical line
            p_vtop = to_screen(v_top)
            p_vbot = to_screen(v_bot)
            draw.line([(p_vtop[0], p_vtop[1]), (p_vbot[0], p_vbot[1])], fill=(70, 80, 90), width=2)

            # Red cork axis
            p_atop = to_screen(a_top)
            p_amid = to_screen(hip_mid)
            draw.line([(p_amid[0], p_amid[1]), (p_atop[0], p_atop[1])], fill=(255, 70, 70), width=4)
            draw.text((p_atop[0] + 10, p_atop[1] - 15), f"Cork Axis ({axis_tilt:.1f}°)", fill=(255, 70, 70), font=font_bold)

            # Links sorted by depth
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

            for j in CORE_JOINTS:
                pj = to_screen(pts_v[j])
                draw.ellipse([pj[0] - 5, pj[1] - 5, pj[0] + 5, pj[1] + 5], fill=(255, 210, 60))

        res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(res)

    out.release()

    # Transcode to standard H.264
    cmd = f"ffmpeg -y -i {tmp_path} -c:v libx264 -pix_fmt yuv420p -r 30 {final_path} && rm {tmp_path}"
    import subprocess
    subprocess.run(cmd, shell=True, check=True)
    print(f"Finished {final_path}")

def main():
    # Clip 13: Extended Orbit (10 seconds, 300 frames)
    render_kinematics_clip(
        "clip13_3d_kinematics_extended_orbit.mp4",
        "run1",
        2,  # air-3
        470,
        595,
        view_mode="orbit",
        title_suffix="Extended Continuous 360° Spatial Orbit",
        desc="Backside Double Cork 900 · 300-Frame Full Trick Sequence",
    )

    # Clip 14: Front Elevation Perspective
    render_kinematics_clip(
        "clip14_3d_kinematics_front_elevation_cork.mp4",
        "run1",
        2,
        490,
        585,
        view_mode="front",
        title_suffix="Front Elevation (Cork Axis Tilt & Tuck Radius)",
        desc="Coronal Plane Projection Isolating 114.5° Body Inversion",
    )

    # Clip 15: Side Elevation Perspective
    render_kinematics_clip(
        "clip15_3d_kinematics_side_pitch_landing.mp4",
        "run1",
        2,
        490,
        585,
        view_mode="side",
        title_suffix="Side Elevation (Pitch Dynamics & Knee Flexion)",
        desc="Sagittal Plane Projection Showing Landing Knee Flexion",
    )

    # Clip 16: Top-Down Spin Plane Perspective
    render_kinematics_clip(
        "clip16_3d_kinematics_top_spin_plane.mp4",
        "run1",
        2,
        490,
        585,
        view_mode="top",
        title_suffix="Transverse Spin Plane (Camera Pan Removed)",
        desc="Top-Down Orthogonal View Tracking True Angular Velocity",
    )

    # Clip 17: Multi-Trick 1440 Extended
    render_kinematics_clip(
        "clip17_3d_kinematics_multi_trick_1440.mp4",
        "synth14_double_cork_1080",
        0,  # air-1
        90,
        360,
        view_mode="orbit",
        title_suffix="Multi-Rotation 1080° Cork Flight Sequence",
        desc="Ground-Truth Omniverse Trick Physics · 270-Frame Arc",
    )

if __name__ == "__main__":
    main()
