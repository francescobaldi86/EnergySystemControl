

class IOClient:
    """
    Base class for all IO clients
    """
    name: str

    def __init__(self, name: str):
        self.name = name