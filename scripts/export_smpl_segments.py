"""Write the SMPL vertex -> body segment map used by the Body Avatar view.

SMPL topology is fixed (6890 vertices), so each vertex gets the segment of the joint that skins it
most (argmax of the LBS weights). Segments follow the usual 5-segment body split (arms, legs, trunk)
plus the head. Output: frontend/src/components/smplSegments.json (one small int per vertex).

Run: .venv\\Scripts\\python.exe scripts/export_smpl_segments.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEGMENTS = ["head", "trunk", "left_arm", "right_arm", "left_leg", "right_leg"]
JOINT_TO_SEGMENT = {
    0: 1, 3: 1, 6: 1, 9: 1, 13: 1, 14: 1,      # pelvis, spine1-3, collars -> trunk
    12: 0, 15: 0,                               # neck, head
    16: 2, 18: 2, 20: 2, 22: 2,                 # left shoulder/elbow/wrist/hand
    17: 3, 19: 3, 21: 3, 23: 3,
    1: 4, 4: 4, 7: 4, 10: 4,                    # left hip/knee/ankle/foot
    2: 5, 5: 5, 8: 5, 11: 5,
}


class _Loose(pickle.Unpickler):
    """The SMPL pickle references chumpy; load it as plain numpy without that dependency."""

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Stub
        return super().find_class(module, name)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})


def main(model=ROOT / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
         out=ROOT / "frontend" / "src" / "components" / "smplSegments.json"):
    with open(model, "rb") as fh:
        data = _Loose(fh, encoding="latin1").load()
    weights = np.asarray(data["weights"] if not isinstance(data["weights"], _Stub) else data["weights"].x)
    if weights.shape != (6890, 24):
        sys.exit(f"unexpected weights shape {weights.shape}")
    # Sum the weights per segment first: the pelvis joint alone outweighs each single leg joint on the
    # upper thigh, but the leg chain together does not.
    seg_w = np.zeros((len(weights), len(SEGMENTS)))
    for j, s in JOINT_TO_SEGMENT.items():
        seg_w[:, s] += weights[:, j]
    labels = seg_w.argmax(1).astype(np.int64)
    out.write_text(json.dumps({"segments": SEGMENTS, "vertex_count": 6890,
                               "source": "argmax of SMPL neutral LBS weights (scripts/export_smpl_segments.py)",
                               "labels": "".join(map(str, labels))}), encoding="utf-8")
    print("wrote", out, {s: int((labels == i).sum()) for i, s in enumerate(SEGMENTS)})


if __name__ == "__main__":
    main()
