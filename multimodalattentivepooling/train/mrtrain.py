
class Train:
    """
    This class implements a wrapper around training procedure for moment retrieval task.
    It initiates dataset object, loss functions and optimizer. In the training loop, these
    objects are connected to each other by passing data between them.
    """
    def __init__(self, ds, ds_args):
        """
        initialize dataset, dataloader, loss, objective and optimizer.
        Each component has a list of args in the form of list[str]
        """
        # initialize the dataset object
        ds = ds(**ds_args)