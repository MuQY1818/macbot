"""``lerobot-train`` with openpi-style image augmentation applied on the GPU. Same arguments as lerobot-train:

    python train_augmented.py --policy.path=lerobot/pi05_base --dataset.root=... [other lerobot-train args]
    accelerate launch --num_processes=8 train_augmented.py ...

Augmentation, following ``preprocess_observation`` in the public openpi repository:
  scene cameras   random crop to 95% then resize back, and rotate within +-5 degrees (every frame)
  all cameras     color jitter (brightness 0.7-1.3, contrast 0.51-1.96, hue +-0.1) with probability 0.5
  wrist cameras   no geometric change, except an optional small shift (WRIST_SHIFT, e.g. 0.03 = +-3%)
                  that helps when a wrist camera mount moved between recording sessions

Augmentation runs on the training batch on the GPU. LeRobot's dataset-side transforms are turned off so
images are not augmented twice. Camera keys containing "wrist" count as wrist cameras.
"""
import os
import sys

import lerobot.scripts.lerobot_train as LT
from lerobot.processor.image_augmentation_processor import ImageAugmentationProcessorStep
from lerobot.transforms.transforms import ImageTransformConfig, ImageTransformsConfig

WRIST_SHIFT = float(os.environ.get("WRIST_SHIFT", "0"))

GEOM = ImageTransformsConfig(enable=True, max_num_transforms=1, tfs={
    "crop_rotate": ImageTransformConfig(weight=1.0, type="RandomAffine", kwargs={
        "degrees": [-5.0, 5.0], "translate": [0.025, 0.025], "scale": [1 / 0.95, 1 / 0.95]}),
})
WRIST_GEOM = ImageTransformsConfig(enable=True, max_num_transforms=1, tfs={
    "shift": ImageTransformConfig(weight=1.0, type="RandomAffine", kwargs={
        "degrees": [0.0, 0.0], "translate": [WRIST_SHIFT, WRIST_SHIFT]}),
})
COLOR = ImageTransformsConfig(enable=True, max_num_transforms=1, tfs={       # Identity at equal weight -> p = 0.5
    "color": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={
        "brightness": [0.7, 1.3], "contrast": [0.51, 1.96], "hue": [-0.1, 0.1]}),
    "none": ImageTransformConfig(weight=1.0, type="Identity", kwargs={}),
})


class Augment:
    def __init__(self, image_keys, device, seed, chunk_size=32):
        def step(cfg, keys, offset):
            s = ImageAugmentationProcessorStep(config=cfg, device=device, chunk_size=chunk_size,
                                               seed=None if seed is None else seed + offset)
            s.image_keys = keys
            return s
        keys = list(image_keys)
        self.scene = [k for k in keys if "wrist" not in k]
        self.wrist = [k for k in keys if "wrist" in k]
        self.steps = [step(GEOM, self.scene, 7919)]
        if WRIST_SHIFT > 0 and self.wrist:
            self.steps.append(step(WRIST_GEOM, self.wrist, 104729))
        self.steps.append(step(COLOR, keys, 0))

    def observation(self, batch):
        for s in self.steps:
            batch = s.observation(batch)
        return batch


def _make_image_augmentation(cfg, dataset, device, process_index):
    keys = [k for k in dataset.meta.camera_keys if k not in dataset.meta.depth_keys]
    seed = None if cfg.seed is None else cfg.seed + process_index
    aug = Augment(keys, str(device), seed, cfg.dataset.image_transforms.gpu_chunk_size or 32)
    if process_index == 0:
        print(f"[augment] crop+rotate: {aug.scene}; wrist shift +-{WRIST_SHIFT:.0%}: {aug.wrist if WRIST_SHIFT else 'off'}; "
              f"color jitter p=0.5: {keys}", flush=True)
    return aug


LT._make_image_augmentation = _make_image_augmentation

if __name__ == "__main__":
    sys.argv += ["--dataset.image_transforms.enable=false"]
    LT.main()
