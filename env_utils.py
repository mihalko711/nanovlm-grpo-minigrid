from PIL import Image
from minigrid.envs import EmptyEnv


def create_env():
    return EmptyEnv(size=16, agent_start_pos=None, render_mode="rgb_array")


def get_agent_view(env):
    ue = env.unwrapped
    arr = ue.get_pov_render(32)
    img = Image.fromarray(arr)
    img = img.rotate(-(ue.agent_dir + 1) * 90, expand=True)
    return img


def get_global_observation(env):
    arr = env.unwrapped.render()
    return Image.fromarray(arr)


def is_goal_in_view(env):
    ue = env.unwrapped
    obs = ue.gen_obs()
    return bool((obs['image'][:, :, 0] == 8).any())


ACTION_NAMES = ["turn left", "turn right", "move forward"]
ACTION_IDS = {name: i for i, name in enumerate(ACTION_NAMES)}


def action_to_text(action):
    return ACTION_NAMES[action]


def text_to_action(text):
    return ACTION_IDS[text]
