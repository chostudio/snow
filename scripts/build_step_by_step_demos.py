#!/usr/bin/env python3
"""Build Two Professional Step-by-Step Demo Videos for Snow VAR.

Video 1: 'snow_demo_walkthrough_1080p.mp4' (End-to-End System & User Workflow ~56s)
Video 2: 'snow_demo_biomechanics_1080p.mp4' (Biomechanics & AI Model Deep-Dive ~48s)

Also creates and installs both projects into CapCut Desktop via VectCutAPI.
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
AUDIO_PATH = Path("/Users/chrisho/Desktop/video-editing/pipeline/VectCutAPI/10-06-2026-08-33-02-PM-d0a062f7-611e-422b-b7af-58abd30f63e5/assets/audio/audio_6f2bf874a0794155.mp3")
INSTALL_SCRIPT = Path("/Users/chrisho/Desktop/video-editing/pipeline/pipeline/install_draft.py")
CAPCUT_DIR = Path(os.path.expanduser("~/Movies/CapCut/User Data/Projects/com.lveditor.draft"))

try:
    font_step = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
    font_desc = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
    font_badge = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 18)
except Exception:
    font_step = font_title = font_desc = font_badge = ImageFont.load_default()

def draw_step_overlay(frame_bgr, step_num, total_steps, step_name, title, desc, badge_txt, badge_color_rgb, progress_ratio):
    """Draws sleek broadcast-grade floating cards and technical callout pills."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img, "RGBA")
    
    # 1. Top-Left Step Header Pill
    step_pill_txt = f"STEP {step_num} OF {total_steps}  ·  {step_name.upper()}"
    pill_w = len(step_pill_txt) * 11 + 45
    draw.rounded_rectangle([50, 45, 50 + pill_w, 95], radius=10, fill=(12, 16, 22, 230), outline=(*badge_color_rgb, 220), width=2)
    draw.ellipse([68, 62, 82, 76], fill=(*badge_color_rgb, 255))
    draw.text((95, 58), step_pill_txt, fill=(255, 255, 255), font=font_step)
    
    # 2. Lower-Third Glass Card
    card_x, card_y, card_w, card_h = 50, 895, 1820, 130
    draw.rounded_rectangle([card_x, card_y, card_x + card_w, card_y + card_h], radius=14, fill=(10, 14, 20, 235), outline=(60, 72, 85, 200), width=2)
    
    # Accent color indicator stripe on left edge of card
    draw.rounded_rectangle([card_x + 4, card_y + 12, card_x + 10, card_y + card_h - 12], radius=3, fill=(*badge_color_rgb, 255))
    
    # Title & Concise Technical Description
    draw.text((card_x + 30, card_y + 22), title, fill=(255, 255, 255), font=font_title)
    draw.text((card_x + 30, card_y + 74), desc, fill=(200, 215, 230), font=font_desc)
    
    # Technical Tag Pill on the right side of the card
    tag_w = len(badge_txt) * 11 + 35
    tag_x = card_x + card_w - tag_w - 30
    tag_y = card_y + 42
    draw.rounded_rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + 44], radius=8, fill=(24, 32, 44, 240), outline=(*badge_color_rgb, 180), width=1)
    draw.text((tag_x + 18, tag_y + 11), badge_txt, fill=(*badge_color_rgb, 255), font=font_badge)
    
    # 3. Bottom Progress Bar
    bar_y = 1060
    bar_h = 6
    draw.rectangle([0, bar_y, 1920, bar_y + bar_h], fill=(20, 25, 32, 200))
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
    for sc in subclips:
        cap = cv2.VideoCapture(str(sc["source"]))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_f = int(sc["start"] * src_fps)
        num_f = int(sc["duration"] * 30.0)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(num_f):
            ret, frame = cap.read()
            if not ret:
                # If clip ended slightly early, repeat last frame
                break
            if frame.shape[0] != 1080 or frame.shape[1] != 1920:
                frame = cv2.resize(frame, (1920, 1080))
                
            progress_ratio = overall_start_ratio + (global_frame / max(1, total_expected_frames - 1)) * (overall_end_ratio - overall_start_ratio)
            annotated = draw_step_overlay(
                frame, step_num, total_steps, step_name, title, desc,
                badge_txt, badge_color_rgb, progress_ratio
            )
            writer.write(annotated)
            global_frame += 1
        cap.release()
        
    # Pad if slight EOF difference
    while global_frame < total_expected_frames:
        writer.write(annotated)
        global_frame += 1
        
    writer.release()
    
    cmd = [
        "ffmpeg", "-y", "-i", str(temp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18", "-r", "30",
        str(out_video_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    temp_raw.unlink(missing_ok=True)
    duration_s = total_expected_frames / 30.0
    print(f"  Rendered Step {step_num}/{total_steps}: {title} ({duration_s:.1f}s, {total_expected_frames} frames)")

def build_video_1():
    print("\n=== Building Video 1: End-to-End System Walkthrough (1080p ~56s) ===")
    scenes = [
        {
            "subclips": [
                {"source": RAW_RUN1, "start": 15.5, "duration": 7.5}
            ],
            "step_name": "Ingest Broadcast Footage",
            "title": "1. Ingest Raw Competition Video",
            "desc": "Standard 2D television feed · Zero wearable sensors, suit markers, or specialized rigs required.",
            "badge": "RAW 2D INPUT",
            "color": (0, 230, 118)  # Emerald
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip05_snowboarder_skeleton_run1_air3_double_cork.mp4", "start": 0.0, "duration": 3.8},
                {"source": SHOWCASE_DIR / "clip06_snowboarder_skeleton_run1_air1_melon.mp4", "start": 0.0, "duration": 4.2}
            ],
            "step_name": "3D Biomechanics Extraction",
            "title": "2. NVIDIA Body Pose NIM (Nova 77)",
            "desc": "Lifts 2D pixels into 3D metric joint coordinates · 77 anatomical keypoints tracked at 30 fps.",
            "badge": "NVIDIA BODY POSE NIM",
            "color": (0, 229, 255)  # Cyan
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip09_omniverse_side_broadcast_synthi0027.mp4", "start": 0.5, "duration": 4.0}
            ],
            "step_name": "Digital Twin & Pan Compensation",
            "title": "3. NVIDIA Omniverse Replicator (Isaac Sim USD)",
            "desc": "Calibrates physics ground truth · Subtracts camera operator pan & tilt to isolate true rider rotation.",
            "badge": "USD DIGITAL TWIN",
            "color": (255, 179, 0)  # Amber
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip18_model_b_classifier_live_hud.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip03_dashboard_scroll_trick.mp4", "start": 5.0, "duration": 4.0}
            ],
            "step_name": "Factored AI Classification & FIS Score",
            "title": "4. Model B Factored Classifier & Model C Regressor",
            "desc": "Model A detects airtime · Model B classifies rotation, axis, stance & grab · Model C predicts 49.8 FIS pts.",
            "badge": "MODELS A, B & C",
            "color": (179, 136, 255)  # Purple
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip13_3d_kinematics_extended_orbit.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip14_3d_kinematics_front_elevation_cork.mp4", "start": 0.0, "duration": 4.0}
            ],
            "step_name": "3D Kinematics & Rotation Rate",
            "title": "5. Continuous Kinematics & Spin Telemetry",
            "desc": "Continuous 360° orbit tracks +977° camera-removed spin, peak 112 RPM, and 114.5° cork axis tilt.",
            "badge": "KINEMATICS ENGINE",
            "color": (76, 217, 100)  # Light Green
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip19_3d_pose_inspector_tab_interactive.mp4", "start": 2.0, "duration": 8.5}
            ],
            "step_name": "Interactive VAR Biomechanics Review",
            "title": "6. Snow VAR Interactive Dashboard",
            "desc": "WebGL 3D skeleton orbit, sagittal knee flexion review (100.3° absorption), and automated coach cues.",
            "badge": "SNOW VAR DASHBOARD",
            "color": (64, 169, 255)  # Soft Blue
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip20_upload_modal_and_pipeline.mp4", "start": 0.0, "duration": 8.0}
            ],
            "step_name": "One-Click Video Upload & Ingestion",
            "title": "7. Self-Service Ingestion & Live Provenance",
            "desc": "Upload any custom snowboarding clip · End-to-end automated inference from pixels to scorecard.",
            "badge": "END-TO-END PIPELINE",
            "color": (255, 99, 132)  # Coral Pink
        }
    ]
    
    total_duration = sum(sum(sc["duration"] for sc in s["subclips"]) for s in scenes)
    print(f"Total Video 1 Duration: {total_duration:.1f}s ({len(scenes)} scenes)")
    
    temp_clips = []
    accum_time = 0.0
    for idx, sc in enumerate(scenes):
        sc_dur = sum(sub["duration"] for sub in sc["subclips"])
        out_clip = OUT_DIR / f"temp_v1_scene_{idx:02d}.mp4"
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
        
    concat_txt = OUT_DIR / "concat_v1.txt"
    with open(concat_txt, "w") as f:
        for tc in temp_clips:
            f.write(f"file '{tc.resolve()}'\n")
            
    v1_no_audio = OUT_DIR / "temp_v1_assembled.mp4"
    cmd_cat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        str(v1_no_audio)
    ]
    subprocess.run(cmd_cat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    final_v1 = OUT_DIR / "snow_demo_walkthrough_1080p.mp4"
    cmd_audio = [
        "ffmpeg", "-y", "-i", str(v1_no_audio), "-i", str(AUDIO_PATH),
        "-filter_complex", f"[1:a]volume=0.20,afade=t=in:st=0:d=1.5,afade=t=out:st={total_duration - 2.0}:d=2.0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(total_duration),
        str(final_v1)
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    for tc in temp_clips:
        tc.unlink(missing_ok=True)
    concat_txt.unlink(missing_ok=True)
    v1_no_audio.unlink(missing_ok=True)
    print(f"==> Video 1 Complete: {final_v1}")
    return final_v1

def build_video_2():
    print("\n=== Building Video 2: Athlete & Judge Biomechanics Deep Dive (1080p ~48s) ===")
    scenes = [
        {
            "subclips": [
                {"source": RAW_RUN1, "start": 2.5, "duration": 7.5}
            ],
            "step_name": "Raw Broadcast Feed",
            "title": "STAGE 1 · Unprocessed Competition Broadcast",
            "desc": "Standard 2D television feed · Zero sensors, calibrated arenas, or athlete preparation required.",
            "badge": "2D BROADCAST",
            "color": (0, 229, 255)  # Cyan
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip06_snowboarder_skeleton_run1_air1_melon.mp4", "start": 0.0, "duration": 4.5},
                {"source": SHOWCASE_DIR / "clip07_snowboarder_skeleton_run1_air2_cork.mp4", "start": 0.0, "duration": 3.5}
            ],
            "step_name": "77-Joint Pose Estimation",
            "title": "STAGE 2 · NVIDIA Body Pose NIM",
            "desc": "Predicts 77 anatomical 3D landmarks · Solves severe foreshortening and clothing deformation.",
            "badge": "NVIDIA NIM",
            "color": (0, 230, 118)  # Emerald
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip09_omniverse_side_broadcast_synthi0027.mp4", "start": 0.5, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip10_omniverse_high_air_synthi0015.mp4", "start": 0.0, "duration": 4.0}
            ],
            "step_name": "Physics Simulation & Pan Removal",
            "title": "STAGE 3 · Omniverse Replicator Digital Twin",
            "desc": "Synthesizes physics ground truth in Isaac Sim · Removes broadcast camera pan to measure true rotation.",
            "badge": "ISAAC SIM USD",
            "color": (255, 179, 0)  # Amber
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip14_3d_kinematics_front_elevation_cork.mp4", "start": 0.0, "duration": 3.0},
                {"source": SHOWCASE_DIR / "clip15_3d_kinematics_side_pitch_landing.mp4", "start": 0.0, "duration": 3.0},
                {"source": SHOWCASE_DIR / "clip16_3d_kinematics_top_spin_plane.mp4", "start": 0.0, "duration": 2.5}
            ],
            "step_name": "Multi-Axis Kinematic Resolution",
            "title": "STAGE 4 · Anatomical Angles & Cork Tilt",
            "desc": "Measures 114.5° off-axis cork angle, 100.3° landing knee flexion absorption, and 0.31m tuck radius.",
            "badge": "BIOMECHANICS",
            "color": (179, 136, 255)  # Purple
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip18_model_b_classifier_live_hud.mp4", "start": 0.0, "duration": 4.0},
                {"source": SHOWCASE_DIR / "clip08_snowboarder_skeleton_trick_session.mp4", "start": 0.0, "duration": 4.0}
            ],
            "step_name": "Factored Heads & FIS Scoring",
            "title": "STAGE 5 · AI Classification & Score Regression",
            "desc": "Classifies 'Backside Double Cork 900 Nose' · Predicts 49.8 official FIS judge points and touchdown safety.",
            "badge": "MODELS B & C",
            "color": (76, 217, 100)  # Light Green
        },
        {
            "subclips": [
                {"source": SHOWCASE_DIR / "clip04_dashboard_interactive_scrub.mp4", "start": 3.0, "duration": 8.0}
            ],
            "step_name": "Decision-Support VAR Review",
            "title": "STAGE 6 · Objective VAR Replay & Provenance",
            "desc": "Frame-accurate scrubbing, 0.5× slow-motion review, and exportable biomechanics breakdown report.",
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
        out_clip = OUT_DIR / f"temp_v2_scene_{idx:02d}.mp4"
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
        
    concat_txt = OUT_DIR / "concat_v2.txt"
    with open(concat_txt, "w") as f:
        for tc in temp_clips:
            f.write(f"file '{tc.resolve()}'\n")
            
    v2_no_audio = OUT_DIR / "temp_v2_assembled.mp4"
    cmd_cat = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        str(v2_no_audio)
    ]
    subprocess.run(cmd_cat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    final_v2 = OUT_DIR / "snow_demo_biomechanics_1080p.mp4"
    cmd_audio = [
        "ffmpeg", "-y", "-i", str(v2_no_audio), "-i", str(AUDIO_PATH),
        "-filter_complex", f"[1:a]volume=0.20,afade=t=in:st=0:d=1.5,afade=t=out:st={total_duration - 2.0}:d=2.0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", str(total_duration),
        str(final_v2)
    ]
    subprocess.run(cmd_audio, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    for tc in temp_clips:
        tc.unlink(missing_ok=True)
    concat_txt.unlink(missing_ok=True)
    v2_no_audio.unlink(missing_ok=True)
    print(f"==> Video 2 Complete: {final_v2}")
    return final_v2

def install_capcut_project(project_name, video_file, title_text):
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
    
    install_capcut_project("snow_demo_walkthrough", v1, "Snow VAR - Step-by-Step Walkthrough")
    install_capcut_project("snow_demo_biomechanics", v2, "Snow VAR - Biomechanics Deep Dive")
    
    print("\n==========================================")
    print("ALL DEMO VIDEOS SUCCESSFULLY GENERATED & INSTALLED!")
    print(f"Video 1: {v1}")
    print(f"Video 2: {v2}")
    print("CapCut Draft 1: ~/Movies/CapCut/User Data/Projects/com.lveditor.draft/snow_demo_walkthrough")
    print("CapCut Draft 2: ~/Movies/CapCut/User Data/Projects/com.lveditor.draft/snow_demo_biomechanics")
    print("==========================================")

if __name__ == "__main__":
    main()
