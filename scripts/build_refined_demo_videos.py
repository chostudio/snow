#!/usr/bin/env python3
"""Build Two Refined Step-by-Step Demo Videos for Snow VAR per User Specifications:

Exact Order for Both Videos:
1. Base clip: Clean snowboarder footage (nothing on it)
2. Skeleton of the clip: 77-joint Nova skeleton overlay
3. Kinematics and spin telemetry clip: 3D kinematics, rotation rate & spin counter
4. Side-by-side NVIDIA Omniverse Replicator: Split-screen original vs Omniverse USD digital twin (NO standalone Omniverse)
5. 'What trick is this?' clip: Clear, uncluttered AI trick identification card
6. Dashboard scrolling: Refilmed with video playing, vertical scroll, horizontal spin scrub, vertical scroll

Layout & Rules:
- Content card moved to LEFT side of the screen so it NEVER blocks the rider.
- NO voice overlay (completely clean audio / clean video).
- Installs both projects into CapCut Desktop via VectCutAPI.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests

ROOT = Path("/Users/chrisho/snow")
RAW_RUN1 = ROOT / "data" / "clips" / "run1.mp4"
SHOWCASE_DIR = ROOT / "docs" / "media" / "showcase_clips"
OUT_DIR = ROOT / "docs" / "media"
INSTALL_SCRIPT = Path("/Users/chrisho/Desktop/video-editing/pipeline/pipeline/install_draft.py")
CAPCUT_DIR = Path(os.path.expanduser("~/Movies/CapCut/User Data/Projects/com.lveditor.draft"))

try:
    font_step = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 28)
    font_desc = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    font_badge = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
except Exception:
    font_step = font_title = font_desc = font_badge = ImageFont.load_default()

def draw_left_content_card(frame_bgr, step_num, total_steps, step_name, title, desc, badge_txt, badge_color_rgb, progress_ratio):
    """Draws sleek floating card on the LEFT side of the screen.
    Leaves the center, right, and lower areas completely open for the rider and landing.
    """
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    
    # 1. Top-Left Step Header Pill
    step_pill_txt = f"STEP {step_num} OF {total_steps}  ·  {step_name.upper()}"
    pill_w = len(step_pill_txt) * 11 + 45
    draw.rounded_rectangle([50, 45, 50 + pill_w, 95], radius=10, fill=(12, 16, 22, 230), outline=(*badge_color_rgb, 220), width=2)
    draw.ellipse([68, 62, 82, 76], fill=(*badge_color_rgb, 255))
    draw.text((95, 58), step_pill_txt, fill=(255, 255, 255), font=font_step)
    
    # 2. Left-Side Content Card (Width: 540px, Height: 180px)
    card_x, card_y, card_w, card_h = 50, 115, 560, 185
    draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=14, fill=(10, 14, 20, 235), outline=(60, 72, 85, 200), width=2)
    
    # Accent color vertical stripe on left
    draw.rounded_rectangle([card_x + 4, card_y + 12, card_x + 10, card_y + card_h - 12], radius=3, fill=(*badge_color_rgb, 255))
    
    # Title
    draw.text((card_x + 24, card_y + 20), title, fill=(255, 255, 255), font=font_title)
    
    # Description (wrap to 2 lines cleanly)
    words = desc.split(" ")
    line1, line2 = "", ""
    for w in words:
        if len(line1 + " " + w) <= 44:
            line1 = (line1 + " " + w).strip()
        else:
            line2 = (line2 + " " + w).strip()
            
    draw.text((card_x + 24, card_y + 68), line1, fill=(205, 220, 235), font=font_desc)
    if line2:
        draw.text((card_x + 24, card_y + 98), line2, fill=(205, 220, 235), font=font_desc)
        
    # Technical Pill Tag inside card at bottom
    tag_w = len(badge_txt) * 10 + 26
    tag_x = card_x + 24
    tag_y = card_y + 135
    draw.rounded_rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + 36], radius=6, fill=(22, 30, 42, 240), outline=(*badge_color_rgb, 180), width=1)
    draw.text((tag_x + 12, tag_y + 8), badge_txt, fill=(*badge_color_rgb, 255), font=font_badge)
    
    # 3. Bottom Progress Bar
    bar_y = 1066
    bar_h = 8
    draw.rectangle([0, bar_y, 1920, bar_y + bar_h], fill=(20, 25, 32, 220))
    active_w = int(1920 * progress_ratio)
    if active_w > 0:
        draw.rectangle([0, bar_y, active_w, bar_y + bar_h], fill=(*badge_color_rgb, 255))
        
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def render_scene_multi(subclips, step_num, total_steps, step_name, title, desc, badge_txt, badge_color_rgb, out_video_path, overall_start_ratio, overall_end_ratio):
    total_expected_frames = int(sum(sc["duration"] for sc in subclips) * 30.0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_raw = out_video_path.parent / f"temp_{out_video_path.name}"
    writer = cv2.VideoWriter(str(temp_raw), fourcc, 30.0, (1920, 1080))
    
    global_frame = 0
    annotated = None
    for sc in subclips:
        cap = cv2.VideoCapture(str(sc["source"]))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_f = int(sc["start"] * src_fps)
        num_f = int(sc["duration"] * 30.0)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(num_f):
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[0] != 1080 or frame.shape[1] != 1920:
                frame = cv2.resize(frame, (1920, 1080))
                
            progress_ratio = overall_start_ratio + (global_frame / max(1, total_expected_frames - 1)) * (overall_end_ratio - overall_start_ratio)
            annotated = draw_left_content_card(
                frame, step_num, total_steps, step_name, title, desc,
                badge_txt, badge_color_rgb, progress_ratio
            )
            writer.write(annotated)
            global_frame += 1
        cap.release()
        
    while global_frame < total_expected_frames and annotated is not None:
        writer.write(annotated)
        global_frame += 1
        
    writer.release()
    
    # Transcode to clean h264 (video only, NO audio stream)
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18", "-r", "30",
        "-an",
        str(out_video_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    temp_raw.unlink(missing_ok=True)
    duration_s = total_expected_frames / 30.0
    print(f"  Rendered Step {step_num}/{total_steps}: {title} ({duration_s:.1f}s, {total_expected_frames} frames)")

def build_video_1():
    print("\n=== Building Video 1: Refined Step-by-Step Workflow (1080p ~53s) ===")
    scenes = [
        # 1. Base clip: clean video, nothing on it
        {
            "subclips": [
                {"source": RAW_RUN1, "start": 15.5, "duration": 7.5}
            ],
            "step_name": "Ingest Broadcast Footage",
            "title": "1. Ingest Competition Video",
            "desc": "Standard 2D television broadcast · Zero wearable sensors, suit markers, or athlete setup.",
            "badge": "MONOCULAR 2D INPUT",
            "color": (0, 230, 118)  # Emerald
        },
        # 2. Skeleton of the clip
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip05_snowboarder_skeleton_run1_air3_double_cork.mp4", "start": 0.0, "duration": 3.8},
                {"source": SHOWCASE_DIR / "clip05_snowboarder_skeleton_run1_air3_double_cork.mp4", "start": 0.5, "duration": 3.7}
            ],
            "step_name": "3D Skeleton Tracking",
            "title": "2. NVIDIA Body Pose NIM",
            "desc": "Recovers 77-joint Nova skeleton in 3D metric camera space · Full kinematic chain.",
            "badge": "77 3D JOINTS (NOVA)",
            "color": (0, 229, 255)  # Cyan
        },
        # 3. Kinematics and spin telemetry clip
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip13_3d_kinematics_extended_orbit.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip14_3d_kinematics_front_elevation_cork.mp4", "start": 0.0, "duration": 4.0}
            ],
            "step_name": "Kinematics & Spin Telemetry",
            "title": "3. 3D Kinematics & Rotation",
            "desc": "Continuous 360° orbit tracks +977° camera-removed spin, peak 112 RPM, and 114.5° cork tilt.",
            "badge": "CAMERA PAN SUBTRACTED",
            "color": (76, 217, 100)  # Light Green
        },
        # 4. Side-by-side NVIDIA Omniverse Replicator (original vs Omniverse, NO isolated Omniverse)
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", "start": 0.0, "duration": 4.0}
            ],
            "step_name": "Omniverse Digital Twin",
            "title": "4. Omniverse Side-by-Side Twin",
            "desc": "Real broadcast footage directly compared with NVIDIA Isaac Sim USD physics simulation.",
            "badge": "USD DIGITAL TWIN",
            "color": (255, 179, 0)  # Amber
        },
        # 5. 'What trick is this?' clip (clear, uncluttered, focused)
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip23_what_trick_is_this.mp4", "start": 0.0, "duration": 3.5},
                {"source": SHOWCASE_DIR / "clip23_what_trick_is_this.mp4", "start": 1.0, "duration": 4.0}
            ],
            "step_name": "Trick Classification & Score",
            "title": "5. What Trick Is This?",
            "desc": "Model B decomposes rotation, axis, stance & grab · Model C predicts 49.8 official FIS points.",
            "badge": "FACTORED AI CLASSIFIER",
            "color": (179, 136, 255)  # Purple
        },
        # 6. Dashboard scrolling (video playing, vertical scroll, horizontal spin scrub, vertical scroll)
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip22_dashboard_scroll_refilmed.mp4", "start": 0.0, "duration": 14.5}
            ],
            "step_name": "Interactive VAR Review",
            "title": "6. Snow VAR Dashboard Review",
            "desc": "Video playing live, vertical scroll, interactive spin scrubbing, and biomechanics cues.",
            "badge": "INTERACTIVE VAR REPLAY",
            "color": (64, 169, 255)  # Blue
        }
    ]
    
    total_duration = sum(sum(sc["duration"] for sc in s["subclips"]) for s in scenes)
    print(f"Total Video 1 Duration: {total_duration:.1f}s ({len(scenes)} scenes)")
    
    temp_clips = []
    accum_time = 0.0
    for idx, sc in enumerate(scenes):
        sc_dur = sum(sub["duration"] for sub in sc["subclips"])
        out_clip = OUT_DIR / f"temp_v1_refined_{idx:02d}.mp4"
        start_ratio = accum_time / total_duration
        end_ratio = (accum_time + sc_dur) / total_duration
        render_scene_multi(
            subclips=sc["subclips"],
            step_num=idx + 1,
            total_steps=len(scenes),
            step_name=sc["step_name"],
            title=sc["title"],
            desc=sc["desc"],
            badge_txt=sc["badge"],
            badge_color_rgb=sc["color"],
            out_video_path=out_clip,
            overall_start_ratio=start_ratio,
            overall_end_ratio=end_ratio
        )
        temp_clips.append(out_clip)
        accum_time += sc_dur
        
    concat_txt = OUT_DIR / "concat_v1_refined.txt"
    with open(concat_txt, "w") as f:
        for tc in temp_clips:
            f.write(f"file '{tc.resolve()}'\n")
            
    final_v1 = OUT_DIR / "snow_demo_walkthrough_1080p.mp4"
    cmd_cat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        "-an",
        str(final_v1)
    ]
    subprocess.run(cmd_cat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    for tc in temp_clips:
        tc.unlink(missing_ok=True)
    concat_txt.unlink(missing_ok=True)
    print(f"==> Video 1 Complete (NO voice overlay): {final_v1}")
    return final_v1

def build_video_2():
    print("\n=== Building Video 2: Refined Biomechanics & Judge Review (1080p ~48s) ===")
    scenes = [
        # 1. Base clip
        {
            "subclips": [
                {"source": RAW_RUN1, "start": 3.0, "duration": 7.0}
            ],
            "step_name": "Competition Feed",
            "title": "STAGE 1 · Raw Broadcast Footage",
            "desc": "Monocular 2D television stream · Clean rider approach and kicker launch with zero athlete markers.",
            "badge": "2D BROADCAST",
            "color": (0, 229, 255)  # Cyan
        },
        # 2. Skeleton of the clip
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip06_snowboarder_skeleton_run1_air1_melon.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip07_snowboarder_skeleton_run1_air2_cork.mp4", "start": 0.0, "duration": 3.5}
            ],
            "step_name": "Pose Estimation",
            "title": "STAGE 2 · NVIDIA Body Pose NIM",
            "desc": "Estimates 77 joints across melon grab and corked rotation · Solves depth foreshortening.",
            "badge": "NVIDIA NIM",
            "color": (0, 230, 118)  # Emerald
        },
        # 3. Kinematics and spin telemetry
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip14_3d_kinematics_front_elevation_cork.mp4", "start": 0.0, "duration": 3.5},
                {"source": SHOWCASE_DIR / "clip15_3d_kinematics_side_pitch_landing.mp4", "start": 0.0, "duration": 3.5}
            ],
            "step_name": "Multi-Axis Kinematics",
            "title": "STAGE 3 · Off-Axis Cork & Landing Flex",
            "desc": "Measures 114.5° off-axis cork angle and 100.3° landing knee flexion absorption.",
            "badge": "BIOMECHANICS",
            "color": (76, 217, 100)  # Light Green
        },
        # 4. Side-by-side Omniverse Replicator (original vs Omniverse, NO isolated Omniverse)
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", "start": 0.0, "duration": 3.5}
            ],
            "step_name": "Digital Twin Comparison",
            "title": "STAGE 4 · Broadcast vs Omniverse USD",
            "desc": "Direct side-by-side physics validation · Eliminates camera operator pan distortion.",
            "badge": "USD DIGITAL TWIN",
            "color": (255, 179, 0)  # Amber
        },
        # 5. 'What trick is this?' clip
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip23_what_trick_is_this.mp4", "start": 0.0, "duration": 3.5},
                {"source": SHOWCASE_DIR / "clip23_what_trick_is_this.mp4", "start": 0.5, "duration": 3.5}
            ],
            "step_name": "Trick Classification",
            "title": "STAGE 5 · What Trick Is This?",
            "desc": "Classifies 'Backside Double Cork 900 Nose' · Predicts 49.8 official FIS judge points.",
            "badge": "FACTORED CLASSIFICATION",
            "color": (179, 136, 255)  # Purple
        },
        # 6. Dashboard scrolling (video playing, vertical scroll, horizontal spin scrub, vertical scroll)
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip22_dashboard_scroll_refilmed.mp4", "start": 1.5, "duration": 12.0}
            ],
            "step_name": "Objective VAR Replay",
            "title": "STAGE 6 · Decision-Support Dashboard",
            "desc": "Video playing, frame-accurate spin scrubbing, and objective FIS scoring provenance.",
            "badge": "DECISION SUPPORT",
            "color": (64, 169, 255)  # Blue
        }
    ]
    
    total_duration = sum(sum(sc["duration"] for sc in s["subclips"]) for s in scenes)
    print(f"Total Video 2 Duration: {total_duration:.1f}s ({len(scenes)} scenes)")
    
    temp_clips = []
    accum_time = 0.0
    for idx, sc in enumerate(scenes):
        sc_dur = sum(sub["duration"] for sub in sc["subclips"])
        out_clip = OUT_DIR / f"temp_v2_refined_{idx:02d}.mp4"
        start_ratio = accum_time / total_duration
        end_ratio = (accum_time + sc_dur) / total_duration
        render_scene_multi(
            subclips=sc["subclips"],
            step_num=idx + 1,
            total_steps=len(scenes),
            step_name=sc["step_name"],
            title=sc["title"],
            desc=sc["desc"],
            badge_txt=sc["badge"],
            badge_color_rgb=sc["color"],
            out_video_path=out_clip,
            overall_start_ratio=start_ratio,
            overall_end_ratio=end_ratio
        )
        temp_clips.append(out_clip)
        accum_time += sc_dur
        
    concat_txt = OUT_DIR / "concat_v2_refined.txt"
    with open(concat_txt, "w") as f:
        for tc in temp_clips:
            f.write(f"file '{tc.resolve()}'\n")
            
    final_v2 = OUT_DIR / "snow_demo_biomechanics_1080p.mp4"
    cmd_cat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        "-an",
        str(final_v2)
    ]
    subprocess.run(cmd_cat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    for tc in temp_clips:
        tc.unlink(missing_ok=True)
    concat_txt.unlink(missing_ok=True)
    print(f"==> Video 2 Complete (NO voice overlay): {final_v2}")
    return final_v2

def install_capcut_project(project_name, video_file):
    print(f"\nInstalling CapCut Desktop Project: '{project_name}'...")
    resp = requests.post("http://localhost:9001/create_draft", json={"width": 1920, "height": 1080})
    resp_data = resp.json()
    url = resp_data["output"]["draft_url"]
    draft_id = url.split("draft_id=")[1].split("&")[0] if "draft_id=" in url else url.split("/")[-1]
    
    cap = cv2.VideoCapture(str(video_file))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = frames / fps
    cap.release()
    
    add_payload = {
        "draft_id": draft_id,
        "video_url": str(video_file.resolve()),
        "start": 0.0,
        "end": dur,
        "target_start": 0.0,
        "width": 1920,
        "height": 1080
    }
    requests.post("http://localhost:9001/add_video", json=add_payload)
    requests.post("http://localhost:9001/save_draft", json={"draft_id": draft_id})
    
    dest_folder = CAPCUT_DIR / project_name
    if dest_folder.exists():
        shutil.rmtree(dest_folder)
        
    env = os.environ.copy()
    env["CAPCUT_VENV"] = "/Users/chrisho/.capcut-pipeline-venv"
    cmd = [
        "python3", str(INSTALL_SCRIPT),
        draft_id,
        "--name", project_name,
        "--no-quit"
    ]
    subprocess.run(cmd, env=env, check=True)
    print(f"Successfully installed '{project_name}' into CapCut: {dest_folder}")

def main():
    v1 = build_video_1()
    v2 = build_video_2()
    
    install_capcut_project("snow_demo_walkthrough", v1)
    install_capcut_project("snow_demo_biomechanics", v2)
    
    print("\n==========================================")
    print("BOTH REFINED DEMO VIDEOS GENERATED & INSTALLED!")
    print(f"Video 1 (53.0s): {v1}")
    print(f"Video 2 (47.0s): {v2}")
    print("CapCut Project 1: ~/Movies/CapCut/User Data/Projects/com.lveditor.draft/snow_demo_walkthrough")
    print("CapCut Project 2: ~/Movies/CapCut/User Data/Projects/com.lveditor.draft/snow_demo_biomechanics")
    print("==========================================")

if __name__ == "__main__":
    main()
