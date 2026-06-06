import json
import os
import argparse
import matplotlib.pyplot as plt


def plot_results(results_path, output_path=None):
    with open(results_path) as f:
        results = json.load(f)

    epoch_results = [r for r in results if r["epoch"] != "final"]
    epochs = [r["epoch"] for r in epoch_results]
    final_result = [r for r in results if r["epoch"] == "final"]
    avg_oracle = results[0]["avg_oracle"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Success rate
    ax1.plot(epochs, [r["success_rate"] for r in epoch_results],
             marker="o", label="Model")
    ax1.axhline(y=100, color="green", linestyle="--", alpha=0.7, label="Oracle (100%)")
    if final_result:
        ax1.axhline(y=final_result[0]["success_rate"], color="blue", linestyle=":", alpha=0.5,
                    label=f"Final: {final_result[0]['success_rate']:.1f}%")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate vs Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Avg steps
    ax2.plot(epochs, [r["avg_steps"] for r in epoch_results],
             marker="o", label="Model")
    ax2.axhline(y=avg_oracle, color="green", linestyle="--", alpha=0.7,
                label=f"Oracle ({avg_oracle:.1f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Avg Steps to Goal")
    ax2.set_title("Avg Steps vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="checkpoints/eval_results.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.results), "eval_curves.png")

    plot_results(args.results, args.output)
