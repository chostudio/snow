"""Synthetic ground truth: parametric tricks rendered with exact labels.

Nothing here touches the measurement pipeline. Renders are fed to the same
ingest, track, NIM and features stages as real footage, and only the labels
written next to them are new. See docs/synthetic.md.

``nova`` and ``tricks`` import nothing beyond numpy, so Blender's bundled
Python can load them; everything that needs bpy lives in ``blender/``.
"""
