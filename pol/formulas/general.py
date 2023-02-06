'''
A simple class used for lazy initialization.
'''
class GeneralFormula:
    def __init__(self, cls, conf):
        self.cls = cls
        self.conf = conf

    def create_instance(self, *args):
        return self.cls(*args, **self.conf)
