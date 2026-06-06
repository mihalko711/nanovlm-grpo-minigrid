import json
import os
import random
import sys
from collections import deque

from minigrid.core.world_object import Goal

from env_utils import (
    create_env, get_agent_view,
    action_to_text, randomize_positions,
)

DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def find_goal(env):
    ue = env.unwrapped
    for x in range(ue.grid.width):
        for y in range(ue.grid.height):
            cell = ue.grid.get(x, y)
            if isinstance(cell, Goal):
                return (x, y)
    return None


def bfs_path(env):
    ue = env.unwrapped
    goal = find_goal(env)
    if goal is None:
        return None

    ax, ay = ue.agent_pos
    ad = ue.agent_dir

    if (ax, ay) == goal:
        return []

    start = (ax, ay, ad)
    q = deque([start])
    parent = {start: (None, None)}

    while q:
        x, y, d = q.popleft()

        if (x, y) == goal:
            actions = []
            state = (x, y, d)
            _, act = parent[state]
            while act is not None:
                actions.append(act)
                state, _ = parent[state]
                _, act = parent[state]
            actions.reverse()
            return actions

        ns = (x, y, (d - 1) % 4)
        if ns not in parent:
            parent[ns] = ((x, y, d), 0)
            q.append(ns)

        ns = (x, y, (d + 1) % 4)
        if ns not in parent:
            parent[ns] = ((x, y, d), 1)
            q.append(ns)

        dx, dy = DIR_VEC[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < ue.width and 0 <= ny < ue.height:
            cell = ue.grid.get(nx, ny)
            if cell is None or isinstance(cell, Goal):
                ns = (nx, ny, d)
                if ns not in parent:
                    parent[ns] = ((x, y, d), 2)
                    q.append(ns)

    return None


def _run_episode(env, num_episodes, successful):
    env.reset(seed=random.randint(0, 2**31 - 1))
    randomize_positions(env)

    path = bfs_path(env)
    if path is None:
        return None, successful

    return path, successful + 1


def generate_dataset_global(num_episodes=100):
    env = create_env()
    os.makedirs("data/images", exist_ok=True)
    prompt = "What is the next action to reach the green goal? Choose from: turn left, turn right, move forward."

    successful = 0
    episode = 0
    total_steps = 0

    with open("data/sft_dataset.jsonl", "w") as f:
        while successful < num_episodes:
            env.reset(seed=random.randint(0, 2**31 - 1))
            randomize_positions(env)
            episode += 1

            path = bfs_path(env)
            if path is None:
                continue

            for step, action in enumerate(path):
                img = get_agent_view(env)
                img_path = f"data/images/ep_{episode:03d}_step_{step:02d}.png"
                img.save(img_path)

                record = {
                    "images": [img_path],
                    "texts": [{"user": prompt, "assistant": action_to_text(action)}],
                }
                f.write(json.dumps(record) + "\n")
                env.step(action)

            successful += 1
            total_steps += len(path)
            print(f"Episode {episode}: {len(path)} steps, goal reached ({successful}/{num_episodes})")

    print(f"Done! Collected {successful} trajectories, {total_steps} total steps.")


def generate_dataset_crop(num_episodes=100):
    env = create_env()
    os.makedirs("data/images_crop", exist_ok=True)
    prompt = "What is the next action to reach the green goal? Choose from: turn left, turn right, move forward."

    successful = 0
    episode = 0
    total_steps = 0

    with open("data/sft_dataset_crop.jsonl", "w") as f:
        while successful < num_episodes:
            env.reset(seed=random.randint(0, 2**31 - 1))
            randomize_positions(env)
            episode += 1

            path = bfs_path(env)
            if path is None:
                continue

            for step, action in enumerate(path):
                img = get_agent_view(env)
                img_path = f"data/images_crop/ep_{episode:03d}_step_{step:02d}.png"
                img.save(img_path)

                record = {
                    "images": [img_path],
                    "texts": [{"user": prompt, "assistant": action_to_text(action)}],
                }
                f.write(json.dumps(record) + "\n")
                env.step(action)

            successful += 1
            total_steps += len(path)
            print(f"Episode {episode}: {len(path)} steps, goal reached ({successful}/{num_episodes})")

    print(f"Done! Collected {successful} trajectories, {total_steps} total steps.")


def generate_all(num_episodes=100):
    env = create_env()
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/images_crop", exist_ok=True)
    prompt = "What is the next action to reach the green goal? Choose from: turn left, turn right, move forward."

    successful = 0
    episode = 0
    total_steps = 0

    f_global = open("data/sft_dataset.jsonl", "w")
    f_crop = open("data/sft_dataset_crop.jsonl", "w")

    while successful < num_episodes:
        env.reset(seed=random.randint(0, 2**31 - 1))
        randomize_positions(env)
        episode += 1

        path = bfs_path(env)
        if path is None:
            continue

        for step, action in enumerate(path):
            img_global = get_agent_view(env)
            img_crop = get_agent_view(env)

            img_path = f"data/images/ep_{episode:03d}_step_{step:02d}.png"
            img_global.save(img_path)
            f_global.write(json.dumps({
                "images": [img_path],
                "texts": [{"user": prompt, "assistant": action_to_text(action)}],
            }) + "\n")

            img_path_crop = f"data/images_crop/ep_{episode:03d}_step_{step:02d}.png"
            img_crop.save(img_path_crop)
            f_crop.write(json.dumps({
                "images": [img_path_crop],
                "texts": [{"user": prompt, "assistant": action_to_text(action)}],
            }) + "\n")

            env.step(action)

        successful += 1
        total_steps += len(path)
        print(f"Episode {episode}: {len(path)} steps, goal reached ({successful}/{num_episodes})")

    f_global.close()
    f_crop.close()
    print(f"Done! Collected {successful} trajectories, {total_steps} total steps.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "global":
        generate_dataset_global()
    elif mode == "crop":
        generate_dataset_crop()
    else:
        generate_all()
