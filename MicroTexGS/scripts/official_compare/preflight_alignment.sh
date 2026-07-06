#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

SRC="$(scene_data_path "${SCENE}")"
assert_dataset "${SRC}"

RNG_MAX_RESO="${RNG_MAX_RESO:-0}"
RNG_MAX_TRAINING_IMAGES="${RNG_MAX_TRAINING_IMAGES:-${VIEW_NUM}}"
RNG_DATA_DEVICE="${RNG_DATA_DEVICE:-cpu}"
require_rng_train_view_alignment "${RNG_MAX_TRAINING_IMAGES}"

case "${SCENE}" in
  LightStage/*|Synthetic/*|RenderCapture/*) EXPECT_HDR=1 ;;
  *) EXPECT_HDR=0 ;;
esac
case "${SCENE}" in
  Synthetic/*|RenderCapture/*) EXPECT_WHITE_BG=1 ;;
  *) EXPECT_WHITE_BG=0 ;;
esac

echo "=== official_compare preflight ==="
echo "SCENE=${SCENE}"
echo "SRC=${SRC}"
echo "VIEW_NUM=${VIEW_NUM}"
echo "RNG_MAX_TRAINING_IMAGES=${RNG_MAX_TRAINING_IMAGES}"
echo "RNG_MAX_RESO=${RNG_MAX_RESO}"
echo "RNG_DATA_DEVICE=${RNG_DATA_DEVICE}"
echo "EXPECT_HDR=${EXPECT_HDR}"
echo "EXPECT_WHITE_BACKGROUND=${EXPECT_WHITE_BG}"
echo "ALIGN_TRAIN_VIEWS=${ALIGN_TRAIN_VIEWS}"

"${PYTHON}" - "${SRC}" "${VIEW_NUM}" "${RNG_MAX_TRAINING_IMAGES}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
view_num = int(sys.argv[2])
rng_views = int(sys.argv[3])

def count_frames(name):
    path = src / name
    if not path.exists():
        return 0, None, None
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    first = frames[0].get("file_path") if frames else None
    last = frames[-1].get("file_path") if frames else None
    return len(frames), first, last

train_n, train_first, train_last = count_frames("transforms_train.json")
test_n, test_first, test_last = count_frames("transforms_test.json")
valid_n, valid_first, valid_last = count_frames("transforms_valid.json")

print(f"raw_train_frames={train_n} first={train_first} last={train_last}")
print(f"raw_test_frames={test_n} first={test_first} last={test_last}")
print(f"raw_valid_frames={valid_n} first={valid_first} last={valid_last}")

if train_n and train_n < view_num:
    print(f"warning: VIEW_NUM={view_num} exceeds raw train frames={train_n}; dataset loader will cap to available frames.")
if train_n and train_n < rng_views:
    print(f"warning: RNG_MAX_TRAINING_IMAGES={rng_views} exceeds raw train frames={train_n}; dataset loader will cap to available frames.")
PY

echo "preflight OK"
