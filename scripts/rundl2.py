import argparse
from multimodalattentivepooling.utils.moduleload import load_comp, load_args

def create_parse():
    parser = argparse.ArgumentParser(description='Prepare TVR dataset.')
    parser.add_argument('--config', '-cfg', type=str, required=True, help='rundl config file')
    return parser


def run(dataset, dataset_args, dataloader, dataloader_args, transforms, transforms_args, logger, logger_args):
    """
    :param dataset (str): module specification of the dataset
    :param dataset_args(str): YAML file containing dataset arguments
    :param dataloader(str): module specification of the dataloader, for example torch.utils.data.DataLoader
    :param dataloader_args(str): YAML file containing dataloader arguments. The dataset will also be passed to this class.
    :param logger(str): module specification of the dataloader, for example torch.utils.data.DataLoader
    :param logger_args(str): YAML file containing logger arguments.
    :return:
    """
    # load components dynamically, initialize them
    dataset, dataset_args = load_comp(dataset), load_args(dataset_args)
    dataset = dataset(**dataset_args)
    dataloader, dataloader_args = load_comp(dataloader), load_args(dataloader_args)
    dataloader = dataloader(dataset, **dataloader_args)
    print("data loader:",len(dataloader))
    transforms, transforms_args= load_comp(transforms), load_args(transforms_args)
    transforms = transforms(**transforms_args)
    logger, logger_args= load_comp(logger), load_args(logger_args)
    logger = logger(**logger_args)
    # extract data
    for epoch_index, data in enumerate(dataloader):
        if epoch_index%2000==1999:
            print(epoch_index)
        data = transforms(data)
        logger(data)
    logger.conclude()



if __name__=="__main__":
    parser = create_parse()
    args = parser.parse_args()
    args = load_args(args.config)
    run(**args)