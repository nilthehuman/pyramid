"""The core class that implements most of the business logic: a generalized two-dimensional morphological paradigm."""

class Paradigm(list):
    """An m x n table of two orthogonal, multivalued morpho(phono)logical features that jointly determine the binary value of a third feature."""

    def __init__(self, iterable=None):
        if iterable:
            super().__init__(iterable)
        else:
            super().__init__([])

    def step(self):
        pass # TODO

    def simulate(self, iterations):
        pass # TODO

