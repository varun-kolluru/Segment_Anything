import os
import torch
import numpy as np
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

# ================= CONFIG =================
BASE_DIR = "storage"
FRAME_DIR = "storage/frames/28e794b4-63c5-4761-9e35-218418228185"
MASK_DIR = os.path.join(BASE_DIR, "masks", "28e794b4-63c5-4761-9e35-218418228185")

os.makedirs(MASK_DIR, exist_ok=True)

# ================= DEVICE =================
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)

# ================= LOAD MODEL =================
predictor = SAM2VideoPredictor.from_pretrained(
    "facebook/sam2-hiera-tiny",
    device=device,
    vos_optimized=False  # safer on Mac
)

# ================= INIT STATE =================
with torch.inference_mode():
    inference_state = predictor.init_state(FRAME_DIR)

# ================= RESET STATE (IMPORTANT) =================
predictor.reset_state(inference_state)

# ================= ADD PROMPT =================
ann_frame_idx = 0
ann_obj_id = 1

points = np.array([[210, 350]], dtype=np.float32)
labels = np.array([1], np.int32)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

print("Prompt added for object:", out_obj_ids)

# ================= PROPAGATE & SAVE MASKS =================
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    for i, out_obj_id in enumerate(out_obj_ids):
        # convert logits -> binary mask
        mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype("uint8") * 255
        mask = np.squeeze(mask)  # remove extra dims (1,H,W) -> (H,W)

        mask_path = os.path.join(
            MASK_DIR,
            f"mask_{out_frame_idx:05d}_obj{out_obj_id}.png"
        )

        Image.fromarray(mask).save(mask_path)

    print(f"Saved masks for frame {out_frame_idx}")


print("✅ All masks saved to:", MASK_DIR)


