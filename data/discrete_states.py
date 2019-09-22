class discrete_states(object):
    """Control class for entire project. Contains the game loop, and contains
    the event_loop which passes events to States as needed. Logic for flipping
    states is also found here."""
    def __init__(self, reward, score):
        self.reward = 0
        self.score = 0