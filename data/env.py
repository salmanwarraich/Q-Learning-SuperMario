from . import setup, tools
from .states import main_menu, load_screen, level1
from . import constants as c
import pygame as pg
import numpy as np
import scipy.misc
import random
import pdb
import time

class Env:
    action_idx = {
        0: 273,
        1: 274,
        2: 276,  # left
        3: 275,  # right
        4: 97,   # jump
        5: 115   # speed
    }

    mapping = {
        0: [0, 0, 0, 0, 0, 0],  # NO
        1: [0, 0, 1, 0, 0, 0],  # Left
        2: [0, 0, 1, 0, 1, 0],  # Left + A
        3: [0, 0, 0, 1, 0, 0],  # Right
        4: [0, 0, 0, 1, 1, 0],  # Right + A
        5: [0, 0, 0, 0, 1, 0],  # A - Jumpfire
    }

    controller = None

    action_n = len(mapping.keys())

    resize_x = 90
    resize_y = 90
    color_chanel = 1
    state_n = resize_x * resize_y * color_chanel

    def __init__(self):

        self.run_it = tools.Control(setup.ORIGINAL_CAPTION, self)
        self.state_dict = {
            c.MAIN_MENU: main_menu.Menu(),
            c.LOAD_SCREEN: load_screen.LoadScreen(),
            c.TIME_OUT: load_screen.TimeOut(),
            c.GAME_OVER: load_screen.GameOver(),
            c.LEVEL1: level1.Level1()
        }
        self.run_it.ml_done = False
        self.run_it.setup_states(self.state_dict, c.LEVEL1)


    def get_random_actions(self):
        return random.randint(0, 13)

    def rgb2gray(self, image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def reset(self, start_position=0):
        self.state_dict = {
            c.MAIN_MENU: main_menu.Menu(),
            c.LOAD_SCREEN: load_screen.LoadScreen(),
            c.TIME_OUT: load_screen.TimeOut(),
            c.GAME_OVER: load_screen.GameOver(),
            c.LEVEL1: level1.Level1()
        }
        self.run_it.ml_done = False
        self.run_it.setup_states(self.state_dict, c.LEVEL1)
        self.run_it.max_posision_x = 200
        state, _, _, _, _, _, _ = self.run_it.get_step()

        return state


    def step(self, action):
        input_action = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action = self.mapping[action]
        for idx in range(len(action)):
            if action[idx] == 1:
                input_action[self.action_idx[idx]] = 1
        manual_keys = self.run_it.event_loop(tuple(input_action))
        self.run_it.update()
        pg.display.update()
        next_state, reward, gameover, clear, max_x, timeout, now_x = self.run_it.get_step()
        reward_d, state_d = self.run_it.get_discrete_state()

        return (next_state, reward, gameover, clear, max_x, timeout, now_x, reward_d, state_d, manual_keys)