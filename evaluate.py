import sys
sys.path.append("nanoVLM")

import json
import os
import glob
import argparse

import torch

from models.vision_language_model import VisionLanguageModel
from data.processors import get_image_processor
from env_utils import create_env, get_agent_view
from expert import bfs_path
from eval_utils import make_image_string, parse_action

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


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
        oracle = bfs_path(env)
        oracle_len = len(oracle) if oracle else 0
        oracle_steps += oracle_len

        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < max_steps:
            img = get_agent_view(env)
            processed_img, (n_h, n_w) = image_processor(img)

            image_string = make_image_string(tokenizer, n_h, n_w, mp_len)
            messages = [
                {"role": "user", "content": image_string + prompt},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_special_tokens=False
            )
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)

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

    return success_rate, avg_steps, avg_oracle


def evaluate_checkpoints(checkpoints_dir, model, tokenizer, image_processor,
                          episodes=50, max_steps=100):
    ckpt_pattern = os.path.join(checkpoints_dir, "sft_epoch_*.pt")
    ckpt_paths = sorted(glob.glob(ckpt_pattern),
                        key=lambda p: int(p.split("_epoch_")[1].split(".pt")[0]))

    results = []
    for ckpt_path in ckpt_paths:
        epoch = int(ckpt_path.split("_epoch_")[1].split(".pt")[0])
        print(f"\n{'='*60}")
        print(f"Evaluating epoch {epoch}...")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()
        sr, avg_s, avg_o = evaluate(model, tokenizer, image_processor, episodes, max_steps)
        results.append({"epoch": epoch, "success_rate": sr, "avg_steps": avg_s, "avg_oracle": avg_o})

    # Final model
    final_path = os.path.join(checkpoints_dir, "sft_model.pt")
    if os.path.exists(final_path):
        print(f"\n{'='*60}")
        print("Evaluating final model...")
        print(f"{'='*60}")
        model.load_state_dict(torch.load(final_path, map_location=device), strict=False)
        model.eval()
        sr, avg_s, avg_o = evaluate(model, tokenizer, image_processor, episodes, max_steps)
        results.append({"epoch": "final", "success_rate": sr, "avg_steps": avg_s, "avg_oracle": avg_o})

    # Print summary
    print("\n\n=== SUMMARY ===")
    print(f"{'Epoch':<8} {'Success Rate':<16} {'Avg Steps':<12} {'Oracle Steps':<14}")
    print("-" * 50)
    for r in results:
        epoch_str = str(r['epoch'])
        print(f"{epoch_str:<8} {r['success_rate']:<16.1f} {r['avg_steps']:<12.1f} {r['avg_oracle']:<14.1f}")

    # Save results
    out_path = os.path.join(checkpoints_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint .pt file")
    parser.add_argument("--checkpoints_dir", type=str, default=None,
                        help="Directory with sft_epoch_*.pt checkpoints")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print("Loading base model...")
    model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-460M-8k")
    model.to(dtype=dtype, device=device)
    model.eval()

    tokenizer = model.tokenizer
    image_processor = get_image_processor(
        max_img_size=model.cfg.max_img_size,
        splitted_image_size=model.cfg.vit_img_size,
        resize_to_max_side_len=model.cfg.resize_to_max_side_len,
    )

    if args.checkpoints_dir:
        evaluate_checkpoints(
            args.checkpoints_dir, model, tokenizer, image_processor,
            args.episodes, args.max_steps,
        )
    elif args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        evaluate(model, tokenizer, image_processor, args.episodes, args.max_steps)
    else:
        parser.print_help()
