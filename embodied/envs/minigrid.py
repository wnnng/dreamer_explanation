from collections import deque
from typing import cast, TypeVar
import gymnasium
import copy

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.wrappers import (
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    ObservationWrapper
)

from embodied.core.wrappers import ResizeImage, UnifyDtypes
from embodied.envs.from_gymnasium import FromGymnasium

V = TypeVar("V")

class HideMission(ObservationWrapper):
    """Remove the 'mission' string from the observation."""

    def __init__(self, env):
        super().__init__(env)
        obs_space = cast(gymnasium.spaces.Dict, self.observation_space)
        obs_space.spaces.pop("mission")

    def observation(self, observation: dict):
        observation.pop("mission")
        return observation

class WrappedMinigrid(FromGymnasium):
    def __init__(self,
                 task: str,
                 fully_observable: bool,
                 hide_mission: bool,
                 seed: int,
                 **kwargs):
        env = gymnasium.make(f"MiniGrid-{task}-v0", render_mode="rgb_array", **kwargs)
        if fully_observable:
            env = RGBImgObsWrapper(env)
        else:
            env = RGBImgPartialObsWrapper(env)
        if hide_mission:
            env = HideMission(env)

        super().__init__(env=env, seed=seed)
        self.task = task

# also wrap in ResizeImage so that we can handle size kwarg
class Minigrid(ResizeImage):
    def __init__(self, *args, size, **kwargs):
        super().__init__(WrappedMinigrid(*args, **kwargs), size=size)
