import random


class ChannelSLAM:
    def __init__(self, config_path):
        pass

    def initialize(self):
        pass

    def update_with_ranges(self, ranges):
        pass

    def get_current_pose(self):
        # Random-walk pose for demo
        return [
            random.uniform(0, 10),
            random.uniform(0, 10),
            random.uniform(-3.14, 3.14),
        ]

    def shutdown(self):
        pass
