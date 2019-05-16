class LDA:
    def __init__(self):
        self.X = None
        self.y = None
        self.m = None

    def fit(self, X, y):
        # store the data
        self.X = X
        self.y = y
        self.m = len(y)
        pass

    def predict(self, x):
        pass
        # predicts x's label according to the magority of his k nn


pass
