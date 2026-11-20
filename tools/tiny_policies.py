"""Generate offline random-weight LeRobot checkpoints for integration tests and benchmarks."""
from __future__ import annotations

import argparse
from pathlib import Path

CONFIGS = {
    "act": dict(chunk_size=4, n_action_steps=3, dim_model=32, n_heads=4, dim_feedforward=64,
                n_encoder_layers=1, use_vae=False, pretrained_backbone_weights=None),
    "diffusion": dict(horizon=8, n_action_steps=3, down_dims=(16, 32), spatial_softmax_num_keypoints=4,
                      diffusion_step_embed_dim=16, noise_scheduler_type="DDIM", num_inference_steps=4,
                      pretrained_backbone_weights=None),
    "vqbet": dict(n_obs_steps=2, n_action_pred_token=2, action_chunk_size=3, crop_shape=None,
                  pretrained_backbone_weights=None, spatial_softmax_num_keypoints=4, gpt_block_size=32,
                  gpt_input_dim=32, gpt_output_dim=32, gpt_hidden_dim=32, gpt_n_head=4, gpt_n_layer=1,
                  vqvae_n_embed=8, vqvae_embedding_dim=16, vqvae_enc_hidden_dim=16),
    "tdmpc": dict(image_encoder_hidden_dim=8, state_encoder_hidden_dim=16, latent_dim=8, mlp_dim=16,
                  q_ensemble_size=2, horizon=3, cem_iterations=2, n_gaussian_samples=16, n_pi_samples=2, n_elites=4),
    "gaussian_actor": dict(image_encoder_hidden_dim=8, state_encoder_hidden_dim=16, latent_dim=16),
    "smolvla": dict(chunk_size=4, n_action_steps=3, num_steps=2, num_vlm_layers=1,
                    resize_imgs_with_padding=(64, 64), load_vlm_weights=False),
    "multi_task_dit": dict(horizon=8, n_action_steps=3, hidden_dim=32, num_layers=1, num_heads=4,
                           timestep_embed_dim=32, image_crop_shape=None, tokenizer_max_length=16,
                           noise_scheduler_type="DDIM", num_inference_steps=4),
}


def _word_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["[UNK]", "[PAD]", "<image>", "<fake>", "<global>", "pick", "up", "the", "cube"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token=words[0]))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token=words[0], pad_token=words[1],
        extra_special_tokens={"fake_image_token": words[3], "global_image_token": words[4]},
    )


def _smol_backbone(path):
    from transformers import (
        SmolVLMConfig,
        SmolVLMImageProcessor,
        SmolVLMProcessor,
        SmolVLMVideoProcessor,
    )

    SmolVLMProcessor(SmolVLMImageProcessor(), _word_tokenizer(), SmolVLMVideoProcessor(),
                     image_seq_len=4).save_pretrained(path)
    SmolVLMConfig(
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=1, head_dim=8, vocab_size=32, pad_token_id=1),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                           image_size=64, patch_size=16),
        image_token_id=2, pad_token_id=1,
    ).save_pretrained(path)


def _clip_backbone(path):
    from transformers import CLIPConfig, CLIPModel

    common = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4)
    CLIPModel(CLIPConfig(
        text_config=dict(**common, vocab_size=32, max_position_embeddings=16, pad_token_id=1,
                         bos_token_id=2, eos_token_id=3),
        vision_config=dict(**common, image_size=64, patch_size=16),
        projection_dim=32,
    )).save_pretrained(path)
    _word_tokenizer().save_pretrained(path)


def build(path, kind):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors

    torch.manual_seed(124)
    names = ["top"] if kind in {"tdmpc", "vqbet"} else ["top", "wrist"]
    size = 64
    count = 6
    keys = ["observation.image"] if kind == "tdmpc" else [f"observation.images.{name}" for name in names]
    features = {key: PolicyFeature(FeatureType.VISUAL, (3, size, size)) for key in keys}
    features["observation.state"] = PolicyFeature(FeatureType.STATE, (count,))
    options = dict(CONFIGS[kind])
    if kind == "smolvla":
        backbone = path.resolve() / "backbone"
        _smol_backbone(backbone)
        options["vlm_model_name"] = str(backbone)
    elif kind == "multi_task_dit":
        backbone = path.resolve() / "clip-backbone"
        _clip_backbone(backbone)
        options.update(vision_encoder_name=str(backbone), text_encoder_name=str(backbone))
    config = make_policy_config(kind, device="cpu", input_features=features,
                                output_features={"action": PolicyFeature(FeatureType.ACTION, (count,))},
                                **options)
    stats = {k: {"min": torch.full(f.shape, -2.0), "max": torch.full(f.shape, 2.0),
                 "mean": torch.zeros(f.shape), "std": torch.ones(f.shape)}
             for k, f in (features | config.output_features).items()}
    stats["action"]["mean"] += 0.25
    model = get_policy_class(kind)(config)
    model.save_pretrained(path)
    for processor in make_pre_post_processors(config, dataset_stats=stats):
        processor.save_pretrained(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=CONFIGS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output, args.kind)
