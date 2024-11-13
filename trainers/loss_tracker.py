class LossMetric:
    def __init__(self):
        self.losses = {
            'train': [],
            'val': [],
            'test': []
        }
        self.accuracies = {
            'train': [],
            'val': [],
            'test': []
        }
        self.mode = 'train'
    
    def train(self):
        self.mode = 'train'
    
    def val(self):
        self.mode = 'val'
    
    def test(self):
        self.mode = 'test'