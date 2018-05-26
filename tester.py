import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.models import *
from utils.dataset import ChestXrayDataSet
from utils.loss import *


class Tester(object):
    def __init__(self, args):
        self.args = args
        self.loader = self.__init_loader()
        self.model = self.__load_model()

    def test(self):
        self.model.eval()

        progress_bar = tqdm(self.loader, desc='Testing')
        y_true = np.array([])
        y_pred = np.array([])
        for data, target in progress_bar:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            output = self.model(data)

            if len(y_true):
                y_true = np.concatenate([y_true, np.array(target.data)])
                y_pred = np.concatenate([y_pred, np.array(output.data)])
            else:
                y_true = np.array(target.data)
                y_pred = np.array(output.data)

        self.__save_array(y_true, './results/y_true_{}_{}'.format(self.args.weight_dir, self.args.mode))
        self.__save_array(y_pred, './results/y_pred_{}_{}'.format(self.args.weight_dir, self.args.mode))

    def __load_model(self):
        model_factory = ModelFactory(model_name=self.args.model,
                                     pretrained=False,
                                     classes=self.args.classes)
        model = model_factory.create_model()

        if self.args.cuda:
            model = model.cuda()

        model.load_state_dict(torch.load('./models/{}'.format(self.args.weight_dir))['state_dict'])
        return model

    def __init_transform(self):
        transform_list = [transforms.Resize(224),
                          transforms.RandomCrop(224),
                          transforms.ToTensor()]
        return transforms.Compose(transform_list)

    def __init_loader(self):
        train_loader = DataLoader(
            ChestXrayDataSet(data_dir=self.args.data_dir,
                             file_list=self.args.test_csv,
                             transforms=self.__init_transform()),
            batch_size=self.args.batch_size,
            shuffle=False
        )
        return train_loader

    def __save_array(self, array, name):
        np.savez('{}.npz'.format(name), array)
