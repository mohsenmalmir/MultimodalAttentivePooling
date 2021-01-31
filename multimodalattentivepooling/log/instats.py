import numpy as np

class BinCount:
    """
    calculates and logs the BinCount of the specified set of input.
    """
    def __init__(self, in_names, ignore_val, bin_sizes, print_interval):
        """
        initialize the bincount class
        :param in_names[list]: list of names of target fields
        :param ignore_val[int]: value to ignore in bincount
        :param bin_sizes[list]: number of bins in each target
        """
        self.in_names = in_names
        self.ignore_val = ignore_val
        self.bin_sizes = bin_sizes
        self.bin_counts = dict()
        for name,binsize in zip(self.in_names,self.bin_sizes):
            self.bin_counts[name] = np.zeros(binsize)
        self.print_interval = print_interval
        self.log_cntr = 0


    def __call__(self, data):
        """
        for the given data, iterate over target fields and perform bincounting
        :param data:
        :return:
        """
        for name,binsize in zip(self.in_names,self.bin_sizes):
            D = data[name].reshape(-1)
            D = D[np.where(D != self.ignore_val)]
            self.bin_counts[name] = self.bin_counts[name] + np.bincount(D,minlength=binsize)
        self.log_cntr += 1
        if self.log_cntr % self.print_interval == 0:
            print(self)

    def __str__(self):
        return str(self.bin_counts)