import random

from PIL import Image
import gymnasium as gym
from minigrid.core.world_object import Goal


def create_env():
    return gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")


def randomize_positions(env):
    ue = env.unwrapped
    w, h = ue.width, ue.height

    cells = [(x, y) for x in range(1, w - 1) for y in range(1, h - 1)]

    gx, gy = random.choice(cells)
    ax, ay = random.choice([c for c in cells if c != (gx, gy)])
    ad = random.randint(0, 3)

    for x in range(w):
        for y in range(h):
            cell = ue.grid.get(x, y)
            if isinstance(cell, Goal):
                ue.grid.set(x, y, None)
                break

    ue.put_obj(Goal(), gx, gy)
    ue.agent_pos = (ax, ay)
    ue.agent_dir = ad


def get_global_observation(env):
    arr = env.unwrapped.render()
    return Image.fromarray(arr)


def get_agent_view(env):
    arr = env.unwrapped.get_pov_render(32)
    return Image.fromarray(arr)


ACTION_NAMES = ["turn left", "turn right", "move forward"]
ACTION_IDS = {name: i for i, name in enumerate(ACTION_NAMES)}


def action_to_text(action):
    return ACTION_NAMES[action]


def text_to_action(text):
    return ACTION_IDS[text]
