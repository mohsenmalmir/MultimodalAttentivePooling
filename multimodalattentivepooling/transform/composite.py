from multimodalattentivepooling.utils.moduleload import load_comp, load_args


class CompositeTransform:
    """
    This class implements the general composite transform. Given a list of data transforms, this class will
    initialize them using YAML arg files. It then invokes them in the sequence specified.
    """
    def __init__(self, **composite_args):
        """
        initialize the transform components by reading the modules
        :param composite_args (dict): specification of composite stages in desired order. This should contain
        an 'order' compoent that specifies which transforms should be initialized and what is the order of invocation.
        """
        self.transforms = []
        for transform_name in composite_args["order"]:
            mod = composite_args[transform_name]["comp"]
            self.transforms.append(load_comp(mod)(**composite_args[transform_name]["args"]))

    def __call__(self, data):
        """
        class the transforms in the specified order. Each transform receives the data, performs modification and returns
        the result.
        :param data (dict): containing the target data.
        :return: modified data that is passed through the transforms
        """
        for t in self.transforms:
            data = t(data)
        return data