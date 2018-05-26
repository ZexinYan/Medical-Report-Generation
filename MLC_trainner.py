# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         self.model = self.__load_model()
#         self.train_loader, self.val_loader = self.__init_loader()
#         self.optimizer = self.__init_optimizer()
#         self.scheduler = self.__init_scheduler()
#         self.loss = self.__init_loss()
#         self.loss_val = self.__init_min_loss()
#
#     def train(self):
#         for epoch in range(1, self.args.epochs + 1):
#             print("Training Epoch-{}".format(epoch))
#             loss_train = self.__epoch_train(epoch)
#             val_loss = self.__epoch_val(epoch)
#             self.scheduler.step(loss_train)
#
#     def __load_model(self):
#         model_factory = ModelFactory(model_name=self.args.model,
#                                      pretrained=self.args.pretrained,
#                                      classes=self.args.classes)
#         model = model_factory.create_model()
#
#         if self.args.weight_dir:
#             model.load_state_dict(torch.load(self.args.weight_dir)['state_dict'])
#
#         if self.args.cuda:
#             model = model.cuda()
#
#         print("Load {} Model".format(self.args.model))
#         return model
#
#     def __init_loss(self):
#         return LossFactory(self.args.loss_function, num_labels=self.args.classes)
#
#     def __epoch_train(self, epoch_id):
#         self.model.train()
#         loss_train = 0
#         for batch_idx, (data, target) in enumerate(self.train_loader):
#             if self.args.cuda:
#                 data, target = data.cuda(), target.cuda()
#             data, target = Variable(data), Variable(target)
#             self.optimizer.zero_grad()
#             output = self.model(data)
#
#             loss = self.__compute_loss(output, target)
#             loss_train += loss.data[0]
#             loss.backward()
#             self.optimizer.step()
#
#         self.loss_val = self.__save_model(loss_train, self.loss_val, epoch_id)
#         print('Epoch: {} - train results - Total train loss: {}'.format(epoch_id, loss_train))
#         return loss_train
#
#     def __epoch_val(self, epoch_id):
#         self.model.eval()
#         loss_val = 0
#
#         for batch_idx, (data, target) in enumerate(self.val_loader):
#             if self.args.cuda:
#                 data, target = data.cuda(), target.cuda()
#             data, target = Variable(data, volatile=True), Variable(target)
#             output = self.model(data)
#             loss_val += self.__compute_loss(output, target)
#
#         # self.loss_val = self.__save_model(loss_val.images[0], self.loss_val, epoch_id)
#         print('Epoch: {} - validation results - '
#               'Total val_loss: {:.4f} '
#               '- Min val_loss: {} '
#               '- Learning rate: {}'.format(epoch_id,
#                                            loss_val.data[0],
#                                            self.loss_val,
#                                            self.optimizer.param_groups[
#                                                0]['lr']))
#         return loss_val.data[0]
#
#     def __compute_loss(self, output, target):
#         return self.loss.compute_loss(output, target)
#
#     def __init_loader(self):
#         train_loader = DataLoader(
#             ChestXrayDataSet(data_dir=self.args.data_dir,
#                              file_list=self.args.train_csv,
#                              transforms=self.__init_transform()),
#             batch_size=self.args.batch_size,
#             shuffle=True
#         )
#         val_loader = DataLoader(
#             ChestXrayDataSet(data_dir=self.args.data_dir,
#                              file_list=self.args.val_csv,
#                              transforms=self.__init_transform()),
#             batch_size=self.args.val_batch_size,
#             shuffle=True
#         )
#         return train_loader, val_loader
#
#     # def __init_callback(self):
#     #     callback_params = {'epochs': self.args.epochs,
#     #                        'samples': len(self.train_loader) * self.args.batch_size,
#     #                        'steps': len(self.train_loader),
#     #                        'metrics': {'acc': np.array([]),
#     #                                    'loss': np.array([]),
#     #                                    'val_acc': np.array([]),
#     #                                    'val_loss': np.array([])}}
#     #     callback_list = callbacks.CallbackList(
#     #         [callbacks.BaseLogger(),
#     #          ])
#     #     callback_list.set_params(callback_params)
#     #     callback_list.set_model(self.model)
#     #     return callback_list
#
#     def __init_optimizer(self):
#         optimizer = optim.Adam(self.model.parameters(),
#                                lr=self.args.lr)
#
#         if self.args.weight_dir:
#             optimizer = torch.load(self.args.weight_dir)['optimizer']
#         return optimizer
#
#     def __init_transform(self):
#         transform_list = [transforms.Resize(self.args.reshape_size),
#                           transforms.RandomCrop(self.args.crop_size),
#                           transforms.RandomHorizontalFlip(),
#                           transforms.RandomRotation(30),
#                           transforms.ToTensor(),
#                           transforms.Normalize([0.485, 0.456, 0.406],
#                                                [0.229, 0.224, 0.225])
#                           ]
#         return transforms.Compose(transform_list)
#
#     def __init_scheduler(self):
#         scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=10)
#         return scheduler
#
#     def __save_model(self, val_loss, min_loss, epoch_id):
#         if val_loss < min_loss:
#             print("Saved Model")
#             min_loss = val_loss
#             if self.args.weights:
#                 file_name = './models/m-{}-{}-{}-{}_{}.pth.tar'.format(self.__get_date(),
#                                                                        self.args.model,
#                                                                        self.args.loss_function,
#                                                                        'weight',
#                                                                        self.args.times)
#             else:
#                 file_name = './models/m-{}-{}-{}_{}.pth.tar'.format(self.__get_date(),
#                                                                     self.args.model,
#                                                                     self.args.loss_function,
#                                                                     self.args.times)
#             torch.save({'epoch': epoch_id + 1,
#                         'state_dict': self.model.state_dict(),
#                         'best_loss': min_loss,
#                         'optimizer': self.optimizer.state_dict()},
#                        file_name)
#         return min_loss
#
#     def __get_date(self):
#         return str(time.strftime('%Y%m%d', time.gmtime()))
#
#     def __init_min_loss(self):
#         loss_val = 100000
#         if self.args.weight_dir:
#             loss_val = torch.load(self.args.weight_dir)['best_loss']
#         return loss_val
