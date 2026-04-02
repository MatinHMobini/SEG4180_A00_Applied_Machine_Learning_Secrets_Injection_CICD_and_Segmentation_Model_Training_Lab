import os
import argparse
import numpy as np
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def bbox_to_mask(bbox, width, height):
    """
    bbox is [x_min, y_min, width, height] like in the Week 7 notebook.
    Returns a boolean mask of shape (H, W).
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)

    mask = np.zeros((height, width), dtype=bool)
    x2 = min(x + w, width)
    y2 = min(y + h, height)
    mask[y:y2, x:x2] = True
    return mask


def iou(a, b, eps=1e-9):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + eps)


def ensure_dirs(out_root):
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(out_root, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(out_root, split, "masks"), exist_ok=True)


def save_pair(out_root, split, idx, image_pil, mask_bool):
    img_path = os.path.join(out_root, split, "images", f"{idx:06d}.png")
    msk_path = os.path.join(out_root, split, "masks", f"{idx:06d}.png")

    image_pil.save(img_path)

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(mask_uint8).save(msk_path)


def build_sam(mask_checkpoint, model_type="vit_b", device="cpu"):
    sam = sam_model_registry[model_type](checkpoint=mask_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(sam)


def generate_pixel_mask_with_sam(image_pil, bboxes, mask_generator, iou_threshold):
    """
    Correct Week 7 logic:
    - build a mask for EACH bbox
    - for each SAM proposal, compute IoU vs each bbox mask
    - keep it if it matches ANY bbox with IoU >= threshold
    - union kept SAM masks into final pixel mask
    """
    w, h = image_pil.size

    # Build per-bbox masks (NOT a single union mask)
    bbox_masks = [bbox_to_mask(b, w, h) for b in bboxes]

    masks = mask_generator.generate(np.array(image_pil))
    final_mask = np.zeros((h, w), dtype=bool)

    for m in masks:
        sam_mask = m["segmentation"].astype(bool)

        # max IoU across all bbox masks (Week7 compares each bbox) :contentReference[oaicite:2]{index=2}
        max_iou = 0.0
        for bm in bbox_masks:
            val = iou(sam_mask, bm)
            if val > max_iou:
                max_iou = val
            if max_iou >= iou_threshold:
                break

        if max_iou >= iou_threshold:
            final_mask |= sam_mask

    # Fallback: if nothing matched, at least use weak bbox masks (so masks aren’t empty)
    if final_mask.sum() == 0:
        weak = np.zeros((h, w), dtype=bool)
        for bm in bbox_masks:
            weak |= bm
        final_mask = weak

    return final_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="data/processed")
    parser.add_argument("--max_samples", type=int, default=300)   # start small on CPU
    parser.add_argument("--iou_threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sam_checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--sam_device", default="cpu")  # Mac: CPU

    args = parser.parse_args()

    ensure_dirs(args.out_root)

    # Week 7 uses this dataset
    ds = load_dataset("keremberke/satellite-building-segmentation", name="full")
    base = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    # Limit for speed
    n = min(args.max_samples, len(base))
    indices = np.arange(n)

    # Split 70/15/15
    train_idx, temp_idx = train_test_split(indices, test_size=0.30, random_state=args.seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=args.seed)

    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    # Build SAM once
    if not os.path.exists(args.sam_checkpoint):
        raise FileNotFoundError(
            f"SAM checkpoint not found at {args.sam_checkpoint}. "
            "Download it into checkpoints/ first."
        )

    mask_generator = build_sam(
        mask_checkpoint=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=args.sam_device
    )

    # Export processed dataset
    global_counter = 0
    for split_name, idxs in split_map.items():
        for i in idxs:
            example = base[int(i)]
            image = example["image"].convert("RGB")
            bboxes = example["objects"]["bbox"]  # Week7 uses objects['bbox'] :contentReference[oaicite:8]{index=8}

            mask = generate_pixel_mask_with_sam(
                image_pil=image,
                bboxes=bboxes,
                mask_generator=mask_generator,
                iou_threshold=args.iou_threshold
            )

            save_pair(args.out_root, split_name, global_counter, image, mask)
            global_counter += 1

    print("Done.")
    print(f"Wrote processed dataset to: {args.out_root}")
    print("Example expected outputs:")
    print("  data/processed/train/images/000000.png")
    print("  data/processed/train/masks/000000.png")


if __name__ == "__main__":
    main()