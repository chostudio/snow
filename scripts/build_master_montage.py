#!/usr/bin/env python3
"""Build Clip 21: SNOW Master Showcase Montage.
Assembles the top highlights into a 26-second 1080p master showcase reel.
"""

from pathlib import Path
import subprocess

ROOT = Path("/Users/chrisho/snow")
CLIPS_DIR = ROOT / "docs" / "media" / "showcase_clips"
OUT_MONTAGE = CLIPS_DIR / "clip21_snow_master_showcase_montage.mp4"
DEMO_LOOP = ROOT / "docs" / "media" / "snow_project_demo_loop.mp4"

# We select segments from the best clips:
# 1. clip01: 0s to 4s (Dashboard UI overview)
# 2. clip05: 0s to 3.5s (Snowboarder skeleton double cork 900)
# 3. clip09: 0.5s to 4.0s (Omniverse broadcast side-angle digital twin)
# 4. clip12: 0s to 3.5s (Split screen: Real vs Omniverse)
# 5. clip13: 0s to 4.0s (3D Kinematics 360 spatial orbit)
# 6. clip18: 0s to 3.5s (Model B factored heads & Model C FIS score)
# 7. clip19: 4s to 8s (Interactive 3D pose tab scrubbing & orbiting)

SEGMENTS = [
    (CLIPS_DIR / "clip01_dashboard_scroll_run1.mp4", 0.0, 4.0, "SNOW VAR - FREESTYLE BIOMECHANICS DASHBOARD"),
    (CLIPS_DIR / "clip05_snowboarder_skeleton_run1_air3_double_cork.mp4", 0.0, 3.5, "NVIDIA BODY POSE NIM - 77-JOINT 3D RECONSTRUCTION"),
    (CLIPS_DIR / "clip09_omniverse_side_broadcast_synthi0027.mp4", 0.5, 4.0, "NVIDIA OMNIVERSE REPLICATOR - SYNTHETIC DIGITAL TWIN"),
    (CLIPS_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", 0.0, 3.5, "DIGITAL TWIN COMPARISON - BROADCAST VS ISAAC SIM"),
    (CLIPS_DIR / "clip13_3d_kinematics_extended_orbit.mp4", 0.0, 4.0, "3D KINEMATICS - SPATIAL ORBIT & CAMERA COMPENSATION"),
    (CLIPS_DIR / "clip18_model_b_classifier_live_hud.mp4", 0.0, 3.5, "MODEL B & C - FACTORED HEADS & FIS SCORE REGRESSION"),
    (CLIPS_DIR / "clip19_3d_pose_inspector_tab_interactive.mp4", 4.0, 8.0, "INTERACTIVE WEB VAR - 3D SKELETON POSE INSPECTOR"),
]

def build_montage():
    temp_files = []
    concat_list = CLIPS_DIR / "montage_concat.txt"
    
    try:
        lines = []
        for i, (clip_path, start, end, label) in enumerate(SEGMENTS):
            cut_path = CLIPS_DIR / f"temp_cut_{i:02d}.mp4"
            duration = end - start
            
            # Extract and add lower-third title card
            vf = (
                f"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,"
                f"drawbox=y=990:color=black@0.75:width=iw:height=90:t=fill,"
                f"drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial\\ Bold.ttf:text='{label}':"
                f"fontcolor=white:fontsize=26:x=50:y=1020"
            )
            
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-i", str(clip_path),
                "-t", str(duration),
                "-vf", vf,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-preset", "fast", "-crf", "18",
                str(cut_path)
            ]
            print(f"Rendering segment {i+1}/{len(SEGMENTS)}: {label}")
            subprocess.run(cmd, check=True)
            temp_files.append(cut_path)
            lines.append(f"file '{cut_path.resolve()}'")
            
        with open(concat_list, "w") as f:
            f.write("\n".join(lines) + "\n")
            
        # Concatenate segments
        print("Concatenating into Master Showcase Montage...")
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
            str(OUT_MONTAGE)
        ]
        subprocess.run(concat_cmd, check=True)
        print(f"Master Montage created: {OUT_MONTAGE}")
        
        # Also copy to demo loop
        import shutil
        shutil.copy(OUT_MONTAGE, DEMO_LOOP)
        print(f"Updated Demo Loop: {DEMO_LOOP}")
        
    finally:
        for t in temp_files:
            t.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)

if __name__ == "__main__":
    build_montage()
