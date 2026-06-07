import json
import os
import random
import sys
from collections import deque

import numpy as np

from minigrid.core.world_object import Goal

from env_utils import (
    create_env, get_agent_view, get_global_observation,
    action_to_text, is_goal_in_view,
)

DIR_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def plan_actions_to_goal(pos, direction, goal=(14, 14)):
    actions = []
    x, y = pos
    gx, gy = goal
    while (x, y) != (gx, gy):
        if x < gx:
            target = 0
        elif x > gx:
            target = 2
        elif y < gy:
            target = 1
        else:
            target = 3

        if direction == target:
            actions.append(2)
            dx, dy = DIR_VEC[direction]
            x += dx
            y += dy
        else:
            left_turns = (direction - target) % 4
            right_turns = (target - direction) % 4
            if left_turns <= right_turns:
                actions.append(0)
                direction = (direction - 1) % 4
            else:
                actions.append(1)
                direction = (direction + 1) % 4
    return actions


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


def _run_episode(env, prompt, image_dir_crop, image_dir_global, episode_num):
    ue = env.unwrapped
    actions = plan_actions_to_goal(ue.agent_pos, ue.agent_dir)

    records_crop = []
    records_global = []

    for step, action in enumerate(actions):
        img_crop = get_agent_view(env)
        img_global = get_global_observation(env)

        crop_path = f"{image_dir_crop}/ep_{episode_num:03d}_step_{step:02d}.png"
        global_path = f"{image_dir_global}/ep_{episode_num:03d}_step_{step:02d}.png"
        img_crop.save(crop_path)
        img_global.save(global_path)

        action_text = action_to_text(action)

        records_crop.append({
            "images": [crop_path],
            "texts": [{"user": prompt, "assistant": action_text}],
            "goal_visible": is_goal_in_view(env),
        })
        records_global.append({
            "images": [global_path],
            "texts": [{"user": prompt, "assistant": action_text}],
            "goal_visible": is_goal_in_view(env),
        })

        _, reward, done, truncated, _ = env.step(action)

    return records_crop, records_global


def _write_records(records, f_out):
    for r in records:
        f_out.write(json.dumps(r) + "\n")


def generate_dataset_both(num_episodes=20):
    env = create_env()
    os.makedirs("data/images_crop", exist_ok=True)
    os.makedirs("data/images_global", exist_ok=True)
    prompt = "What is the next action to reach the green goal? Choose from: turn left, turn right, move forward."

    total_steps = 0

    with open("data/sft_dataset_crop.jsonl", "w") as f_crop, \
         open("data/sft_dataset_global.jsonl", "w") as f_global:
        for episode in range(1, num_episodes + 1):
            env.reset(seed=random.randint(0, 2**31 - 1))

            records_crop, records_global = _run_episode(
                env, prompt, "data/images_crop", "data/images_global", episode,
            )

            _write_records(records_crop, f_crop)
            _write_records(records_global, f_global)
            total_steps += len(records_crop)
            print(f"Episode {episode}: {len(records_crop)} steps, goal reached ({episode}/{num_episodes})")

    print(f"Done! Collected {num_episodes} trajectories, {total_steps} total steps.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    if mode == "both":
        generate_dataset_both(num_episodes=n_eps)
    else:
        print(f"Usage: python expert.py both [num_episodes]")
