import sys
sys.path.append("nanoVLM")

import torch
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_image_processor
from env_utils import (
    create_env, get_global_observation, randomize_positions,
    ACTION_NAMES, text_to_action,
)
from expert import bfs_path

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def make_image_string(tokenizer, n_h, n_w, mp_len):
    parts = []
    if hasattr(tokenizer, "global_image_token"):
        parts.append(tokenizer.global_image_token)
        parts.append(tokenizer.image_token * mp_len)
        if n_h == 1 and n_w == 1:
            return "".join(parts)
    for i in range(n_h):
        for j in range(n_w):
            parts.append(getattr(tokenizer, f"r{i+1}c{j+1}"))
            parts.append(tokenizer.image_token * mp_len)
    return "".join(parts)


def parse_action(text):
    text = text.strip().lower()
    for name in ACTION_NAMES:
        if name in text:
            return text_to_action(name)
    return None


@torch.inference_mode()
def evaluate(model, tokenizer, image_processor, num_episodes=100, max_steps=100):
    env = create_env()
    mp_len = model.cfg.mp_image_token_length
    prompt = (
        "What is the next action to reach the green goal? "
        "Choose from: turn left, turn right, move forward."
    )

    successes = 0
    total_steps = 0
    oracle_steps = 0

    for ep in range(num_episodes):
        env.reset()
        randomize_positions(env)
        oracle = bfs_path(env)
        oracle_len = len(oracle) if oracle else 0
        oracle_steps += oracle_len

        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < max_steps:
            # Process observation
            img = get_global_observation(env)
            processed_img, (n_h, n_w) = image_processor(img)

            # Build prompt with image tokens
            image_string = make_image_string(tokenizer, n_h, n_w, mp_len)
            messages = [
                {"role": "user", "content": image_string + prompt},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_special_tokens=False
            )
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

            # Generate
            generated_ids = model.generate(
                input_ids,
                images=[processed_img.to(dtype=dtype, device=device)],
                attention_mask=attention_mask,
                max_new_tokens=10,
                top_k=1,
                greedy=True,
            )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            action = parse_action(generated_text)

            if action is None:
                # Fallback to first valid action
                action = 2

            _, reward, done, truncated, _ = env.step(action)
            step += 1

        if done:
            successes += 1

        total_steps += step
        print(
            f"Ep {ep + 1}/{num_episodes}: "
            f"{'SUCCESS' if done else 'FAIL'} "
            f"({step}/{oracle_len} steps)"
        )

    success_rate = successes / num_episodes * 100
    avg_steps = total_steps / num_episodes
    avg_oracle = oracle_steps / num_episodes
    print(f"\nSuccess rate: {success_rate:.1f}%")
    print(f"Avg steps: {avg_steps:.1f} (oracle: {avg_oracle:.1f})")

    return success_rate, avg_steps


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_model.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-460M-8k")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(dtype=dtype, device=device)
    model.eval()

    tokenizer = model.tokenizer
    image_processor = get_image_processor(
        max_img_size=model.cfg.max_img_size,
        splitted_image_size=model.cfg.vit_img_size,
        resize_to_max_side_len=model.cfg.resize_to_max_side_len,
    )

    evaluate(model, tokenizer, image_processor, args.episodes, args.max_steps)
