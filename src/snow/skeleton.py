"""The Nova-77 skeleton the 3D Body Pose NIM returns poses on.

Joint 0 is the root (pelvis). ``PARENTS[i]`` is the index of joint ``i``'s
parent, or -1 for the root. This hierarchy is what a later USD/Omniverse
``SkelAnimation`` export needs, so it lives here rather than in the feature code.
"""

# Mirrors NOVA77_PARENTS in the upstream client's constants.py.
PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17,
    14, 19, 20, 21, 22, 14, 24, 25, 26, 27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37,
    3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42, 52, 53, 54, 55, 42, 57,
    58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75,
]  # fmt: skip

NUM_JOINTS = len(PARENTS)
LINKS = [(i, p) for i, p in enumerate(PARENTS) if p >= 0]

ROOT = 0
