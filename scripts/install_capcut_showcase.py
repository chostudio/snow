#!/usr/bin/env python3
"""Build and install CapCut project for Snow Showcase Master."""

import json
import os
import subprocess
import requests
from pathlib import Path

ROOT = Path("/Users/chrisho/snow")
CLIPS_DIR = ROOT / "docs" / "media" / "showcase_clips"
INSTALL_SCRIPT = Path("/Users/chrisho/Desktop/video-editing/pipeline/pipeline/install_draft.py")
CAPCUT_DIR = Path(os.path.expanduser("~/Movies/CapCut/User Data/Projects/com.lveditor.draft"))

SELECTED_CLIPS = [
    (CLIPS_DIR / "clip01_dashboard_scroll_run1.mp4", 0.0, 5.0),
    (CLIPS_DIR / "clip05_snowboarder_skeleton_run1_air3_double_cork.mp4", 0.0, 3.5),
    (CLIPS_DIR / "clip09_omniverse_side_broadcast_synthi0027.mp4", 0.5, 4.0),
    (CLIPS_DIR / "clip12_omniverse_vs_real_splitscreen.mp4", 0.0, 3.5),
    (CLIPS_DIR / "clip13_3d_kinematics_extended_orbit.mp4", 0.0, 4.0),
    (CLIPS_DIR / "clip18_model_b_classifier_live_hud.mp4", 0.0, 3.5),
    (CLIPS_DIR / "clip19_3d_pose_inspector_tab_interactive.mp4", 3.0, 8.0),
    (CLIPS_DIR / "clip20_upload_modal_and_pipeline.mp4", 0.0, 5.0),
    (CLIPS_DIR / "clip21_snow_master_showcase_montage.mp4", 0.0, 10.0),
]

def main():
    print("Creating draft on VectCutAPI...")
    resp = requests.post("http://localhost:9001/create_draft", json={"width": 1920, "height": 1080})
    resp_data = resp.json()
    url = resp_data["output"]["draft_url"]
    if "draft_id=" in url:
        draft_id = url.split("draft_id=")[1].split("&")[0]
    else:
        draft_id = url.split("/")[-1]
    print(f"Created draft ID: {draft_id}")
    
    target_start = 0.0
    for clip_path, start, end in SELECTED_CLIPS:
        dur = end - start
        payload = {
            "draft_id": draft_id,
            "video_url": str(clip_path.resolve()),
            "start": start,
            "end": end,
            "target_start": target_start,
            "width": 1920,
            "height": 1080
        }
        res = requests.post("http://localhost:9001/add_video", json=payload)
        res_data = res.json()
        print(f"Added {clip_path.name} ({dur}s) at {target_start}s -> success={res_data.get('success')}")
        target_start += dur
        
    print("Saving draft...")
    save_resp = requests.post("http://localhost:9001/save_draft", json={"draft_id": draft_id})
    print(f"Saved: {save_resp.json().get('success')}")
    
    project_name = "snow_showcase_master"
    existing_dest = CAPCUT_DIR / project_name
    if existing_dest.exists():
        import shutil
        shutil.rmtree(existing_dest)
        print(f"Removed previous {project_name}")
        
    print(f"Installing draft into CapCut as '{project_name}'...")
    env = os.environ.copy()
    env["CAPCUT_VENV"] = "/Users/chrisho/.capcut-pipeline-venv"
    cmd = [
        "python3", str(INSTALL_SCRIPT),
        draft_id,
        "--name", project_name,
        "--no-quit"
    ]
    subprocess.run(cmd, env=env, check=True)
    print(f"Installed into CapCut draft: {existing_dest}")

if __name__ == "__main__":
    main()
