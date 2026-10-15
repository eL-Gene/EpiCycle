import os
import json

class StateManager:
    def __init__(self, filename="hitl_state.json"):
        self.filename = filename

    # save the current state of the workflow
    def save_state(self, directory, state_dict):
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, self.filename)
        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

    # load the state for the current directory
    def load_state(self, directory):
        path = os.path.join(directory, self.filename)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
