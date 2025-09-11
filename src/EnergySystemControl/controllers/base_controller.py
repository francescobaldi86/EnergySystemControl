class GenericController():
    name: str
    def __init__(self, name):
        self.name = name

    def get_action(self):
        return None