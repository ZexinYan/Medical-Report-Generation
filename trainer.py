import time
import pickle
import argparse

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.logger import Logger


class TrainerBase:
    def __init__(self, args):
        self.args = args
        self.min_loss = 100000000
        self.min_tag_loss = 1000000
        self.min_stop_loss = 1000000
        self.min_word_loss = 10000000
        self._init_model_path()
        self.model_dir = self._init_model_dir()
        self.transform = self._init_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_mode_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list)
        self.val_data_loader = self._init_data_loader(self.args.val_file_list)

        self.extractor = self._init_visual_extractor()
        self.mlc = self._init_mlc()
        self.co_attention = self._init_co_attention()
        self.sentence_model = self._init_sentence_model()
        self.word_model = self._init_word_model()

        self.encoder = self._init_encoder()
        self.decoder = self._init_decoder()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()

    def train(self):
        for epoch_id in range(self.args.epochs):
            train_tag_loss, train_stop_loss, train_word_loss, train_loss = self._epoch_train()
            val_tag_loss, val_stop_loss, val_word_loss, val_loss = self._epoch_val()

            self.scheduler.step(val_loss)
            print(
                "[{} - Epoch {}] train loss:{} - val_loss:{} - lr:{}".format(self._get_now(),
                                                                             epoch_id,
                                                                             train_loss,
                                                                             val_loss,
                                                                             self.optimizer.param_groups[0]['lr']))
            self._save_model(epoch_id, train_loss, train_tag_loss, train_stop_loss, train_stop_loss)
            self._log(train_tags_loss=train_tag_loss,
                      train_stop_loss=train_stop_loss,
                      train_word_loss=train_word_loss,
                      train_loss=train_loss,
                      val_tags_loss=val_tag_loss,
                      val_stop_loss=val_stop_loss,
                      val_word_loss=val_word_loss,
                      val_loss=val_loss,
                      epoch=epoch_id)

    def _epoch_train(self):
        raise NotImplementedError

    def _epoch_val(self):
        raise NotImplementedError

    def _init_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def _init_model_dir(self):
        model_dir = os.path.join(self.args.model_path, self.args.saved_model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_dir = os.path.join(model_dir, self._get_now())
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def _init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print("Vocab Size:{}".format(len(vocab)))
        return vocab

    def _load_mode_state_dict(self):
        try:
            return torch.load(self.args.load_model_path)
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            return None

    def _init_visual_extractor(self):
        model = VisualFeatureExtractor()

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_mlc(self):
        model = MLC(classes=self.args.classes,
                    sementic_features_dim=self.args.sementic_features_dim,
                    kernel_size=self.args.kernel_size,
                    fc_in_features=self.args.fc_in_features,
                    k=self.args.k)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['mlc'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_co_attention(self):
        model = CoAttention(embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.args.visual_size)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['co_attention'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_sentence_model(self):
        raise NotImplementedError

    def _init_word_model(self):
        raise NotImplementedError

    # Debugging
    def _init_encoder(self):
        model = EncoderCNN(embed_size=256)

        if self.args.cuda:
            model = model.cuda()
        return model

    # Debugging
    def _init_decoder(self):
        model = DecoderRNN(embed_size=256,
                           hidden_size=512,
                           vocab_size=len(self.vocab),
                           num_layers=1)

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_data_loader(self, file_list):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 s_max=self.args.s_max,
                                 n_max=self.args.n_max,
                                 shuffle=True)
        return data_loader

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def _init_optimizer(self):
        params = list(self.extractor.parameters()) \
                 + list(self.mlc.parameters()) \
                 + list(self.co_attention.parameters()) \
                 + list(self.sentence_model.parameters()) \
                 + list(self.word_model.parameters())
        return torch.optim.Adam(params=params, lr=self.args.learning_rate)

    def _log(self,
             train_tags_loss,
             train_stop_loss,
             train_word_loss,
             train_loss,
             val_tags_loss,
             val_stop_loss,
             val_word_loss,
             val_loss,
             epoch):
        info = {
            'train tags loss': train_tags_loss,
            'train stop loss': train_stop_loss,
            'train word loss': train_word_loss,
            'train loss': train_loss,
            'val tags loss': val_tags_loss,
            'val stop loss': val_stop_loss,
            'val word loss': val_word_loss,
            'val loss': val_loss,
            'learning rate': self.optimizer.param_groups[0]['lr']
        }

        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, epoch + 1)

    def _init_logger(self):
        logger = Logger(os.path.join(self.model_dir, 'logs'))
        return logger

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def _get_date(self):
        return str(time.strftime('%Y%m%d', time.gmtime()))

    def _get_now(self):
        return str(time.strftime('%Y%m%d-%H:%M:%S', time.gmtime()))

    def _init_scheduler(self):
        scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=20, factor=0.5)
        return scheduler

    def _init_model_path(self):
        if not os.path.exists(self.args.model_path):
            os.makedirs(self.args.model_path)

    def _init_log_path(self):
        if not os.path.exists(self.args.log_path):
            os.makedirs(self.args.log_path)

    def _save_model(self, epoch_id, loss, tag_loss, stop_loss, word_loss):
        def save_model(_filename):
            print("Saved Model in {}".format(_filename))
            torch.save({'extractor': self.extractor.state_dict(),
                        'mlc': self.mlc.state_dict(),
                        'co_attention': self.co_attention.state_dict(),
                        'sentence_model': self.sentence_model.state_dict(),
                        'word_model': self.word_model.state_dict(),
                        'epoch': epoch_id},
                       os.path.join(self.model_dir, "{}".format(_filename)))

        if loss < self.min_loss:
            file_name = "best_loss.pth.tar"
            save_model(file_name)
            self.min_loss = loss
        # if tag_loss < self.min_tag_loss:
        #     file_name = "best_tag.pth.tar"
        #     save_model(file_name)
        #     self.min_tag_loss = tag_loss
        # if stop_loss < self.min_stop_loss:
        #     file_name = "best_stop.pth.tar"
        #     save_model(file_name)
        #     self.min_stop_loss = stop_loss
        # if word_loss < self.min_word_loss:
        #     file_name = "best_word.pth.tar"
        #     save_model(file_name)
        #     self.min_word_loss = word_loss


class LSTMTrainer(TrainerBase):
    def _init_(self, args):
        TrainerBase.__init__(self, args)
        self.args = args

    def _epoch_train(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.train()
        self.mlc.train()
        self.co_attention.train()
        self.sentence_model.train()
        self.word_model.train()

        for i, (images, _, label, captions, prob) in enumerate(self.train_data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images)
            visual_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(visual_features)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, 512))
            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            for sentence_index in range(captions.shape[1]):
                ctx = self.co_attention.forward(visual_features, semantic_features, prev_hidden_states)
                topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                                                                                           prev_hidden_states,
                                                                                           sentence_states)
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                # Debugging...
                # print("Real Stop: {}".format(prob[:, sentence_index]))
                # print("Pred Stop: {}".format(p_stop))
                # print("")

                for word_index in range(1, captions.shape[2] - 1):
                    # context = self._to_var(torch.Tensor(captions[:, sentence_index, :word_index]).long(),
                    #                        requires_grad=False)
                    # real_word = self._to_var(torch.Tensor(captions[:, sentence_index, word_index]).long(),
                    #                          requires_grad=False)
                    word_mask = self._to_var((torch.Tensor(captions[:, sentence_index, word_index]) > 0).float(),
                                             requires_grad=True)
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])

                    # Debugging...
                    # print("Context:{}".format(context))
                    # print("Pred:{}".format(torch.max(words.squeeze(1), 1)[1]))
                    # print("Real:{}".format(context[:, sentence_index, word_index]))
                    # print("")
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index]) * word_mask).sum()
                prev_hidden_states = hidden_state

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            self.extractor.zero_grad()
            self.mlc.zero_grad()
            self.co_attention.zero_grad()
            self.sentence_model.zero_grad()
            self.word_model.zero_grad()

            batch_loss.backward()
            self.optimizer.step()

            tag_loss += batch_tag_loss.data
            stop_loss += batch_stop_loss.data
            word_loss += batch_word_loss.data
            loss += batch_loss.data

            print("tag loss:{}".format(batch_tag_loss))
            print("stop loss:{}".format(batch_stop_loss))
            print("word loss:{}".format(batch_word_loss))

        return tag_loss, stop_loss, word_loss, loss

    def _epoch_val(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.eval()
        self.mlc.eval()
        self.co_attention.eval()
        self.sentence_model.eval()
        self.word_model.eval()

        for i, (images, _, label, captions, prob) in enumerate(self.val_data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images, requires_grad=False)
            visual_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(visual_features)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, 512))
            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            for sentence_index in range(captions.shape[1]):
                ctx = self.co_attention.forward(visual_features, semantic_features, prev_hidden_states)
                topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                                                                                           prev_hidden_states,
                                                                                           sentence_states)
                batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                # Debugging...
                # print("Real Stop: {}".format(prob[:, sentence_index]))
                # print("Pred Stop: {}".format(p_stop))
                # print("")

                for word_index in range(1, captions.shape[2] - 1):
                    # context = self._to_var(torch.Tensor(captions[:, sentence_index, :word_index]).long(),
                    #                        requires_grad=False)
                    # real_word = self._to_var(torch.Tensor(captions[:, sentence_index, word_index]).long(),
                    #                          requires_grad=False)
                    word_mask = self._to_var((torch.Tensor(captions[:, sentence_index, word_index]) > 0).float(),
                                             requires_grad=False)
                    words = self.word_model.forward(topic, context[:, sentence_index, :word_index])

                    # Debugging...
                    # print("Context:{}".format(context))
                    # print("Pred:{}".format(torch.max(words.squeeze(1), 1)[1]))
                    # print("Real:{}".format(real_word))
                    # print("")
                    batch_word_loss += (self.ce_criterion(words, context[:, sentence_index, word_index]) * word_mask).sum()
                prev_hidden_states = hidden_state
            #
            batch_loss = self.args.lambda_tag * batch_tag_loss.data \
                         + self.args.lambda_stop * batch_stop_loss.data \
                         + self.args.lambda_word * batch_word_loss.data

            tag_loss += batch_tag_loss.data
            stop_loss += batch_stop_loss.data
            word_loss += batch_word_loss.data
            loss += batch_loss.data

            # print("tag loss:{}".format(batch_tag_loss))
            # print("stop loss:{}".format(batch_stop_loss))
            # print("word loss:{}".format(batch_word_loss))

        return tag_loss, stop_loss, word_loss, loss

    def _init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['word_lstm'])

        if self.args.cuda:
            model = model.cuda()
        return model


class TCNTrainer(TrainerBase):
    def _init_(self, args):
        TrainerBase.__init__(self, args)
        self.args = args

    def _epoch_train(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.extractor.train()
        self.mlc.train()
        self.co_attention.train()
        self.sentence_model.train()
        self.word_model.train()

        for i, (images, _, label, targets, prob) in enumerate(self.train_data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images)
            visual_features = self.extractor.forward(images)
            tags, semantic_features = self.mlc.forward(visual_features)

            batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            sentence_states = None
            prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, 512))

            for sentence_index in range(targets.shape[1]):
                ctx = self.co_attention.forward(visual_features, semantic_features, prev_hidden_states)
                topic, p_stop, prev_hidden_states, sentence_states = self.sentence_model.forward(ctx,
                                                                                                 prev_hidden_states,
                                                                                                 sentence_states)

                batch_stop_loss += self.ce_criterion(p_stop.squeeze(),
                                                     self._to_var(torch.Tensor(prob[:, sentence_index]),
                                                                  requires_grad=False).long()).sum()

                # Debugging...
                # print("Real Stop: {}".format(prob[:, sentence_index]))
                # print("Pred Stop: {}".format(p_stop))
                # print("")

                for word_index in range(0, targets.shape[2] - 1):
                    real_word = self._to_var(torch.Tensor(targets[:, sentence_index, word_index + 1]).long(),
                                             requires_grad=False)
                    # word_mask = self._to_var((torch.Tensor(captions[:, sentence_index, word_index]) > 1).float(),
                    #                           requires_grad=True)
                    captions = self._to_var(torch.zeros(images.shape[0], self.args.n_max), requires_grad=False)
                    captions[:, word_index] = self._to_var(torch.Tensor(targets[:, sentence_index, word_index]),
                                                           requires_grad=False)
                    words = self.word_model.forward(topic, captions.long())
                    #
                    # Debugging...
                    # print("Context:{}".format(captions))
                    print("Pred:{}".format(torch.max(words.squeeze(1), 1)[1]))
                    print("Real:{}".format(real_word))
                    print("")
                    batch_word_loss += (self.ce_criterion(words, real_word)).sum()

            batch_loss = self.args.lambda_tag * batch_tag_loss \
                         + self.args.lambda_stop * batch_stop_loss \
                         + self.args.lambda_word * batch_word_loss

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            tag_loss += batch_tag_loss
            stop_loss += batch_stop_loss
            word_loss += batch_word_loss
            loss += batch_loss

            print("tag loss:{}".format(batch_tag_loss))
            print("stop loss:{}".format(batch_stop_loss))
            print("word loss:{}".format(batch_word_loss))

        return tag_loss, stop_loss, word_loss, loss

    def _epoch_val(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0

        return tag_loss, stop_loss, word_loss, loss

    def _init_sentence_model(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def _init_word_model(self):
        model = WordTCN(input_channel=self.args.n_max + 1,
                        vocab_size=len(self.vocab),
                        embed_size=self.args.embed_size,
                        output_size=self.args.hidden_size,
                        nhid=self.args.word_nhid,
                        levels=self.args.word_levels,
                        kernel_size=self.args.word_kernel_size,
                        dropout=self.args.word_dropout,
                        n_max=self.args.n_max)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['word_model'])

        if self.args.cuda:
            model = model.cuda()
        return model


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./report_models/',
                        help='path for saving trained models')
    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/debugging_vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/debugging_captions.json',
                        help='path for captions')
    parser.add_argument('--train_file_list', type=str, default='./data/debugging.txt',
                        help='the train array')
    parser.add_argument('--val_file_list', type=str, default='./data/debugging.txt',
                        help='the val array')
    parser.add_argument('--load_model_path', type=str, default='.',
                        help='The path of loaded model')
    parser.add_argument('--saved_model_name', type=str, default='debugging_word_less_vocab',
                        help='The name of saved model')

    parser.add_argument('--classes', type=int, default=156)
    parser.add_argument('--sementic_features_dim', type=int, default=512)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--fc_in_features', type=int, default=2048)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--visual_size', type=int, default=49)
    parser.add_argument('--sentence_num_layers', type=int, default=2)
    parser.add_argument('--word_num_layers', type=int, default=2)

    parser.add_argument('--sentence_nhid', type=int, default=512)
    parser.add_argument('--sentence_levels', type=int, default=10)
    parser.add_argument('--sentence_kernel_size', type=int, default=2)
    parser.add_argument('--sentence_dropout', type=int, default=0)

    parser.add_argument('--word_nhid', type=int, default=512)
    parser.add_argument('--word_levels', type=int, default=8)
    parser.add_argument('--word_kernel_size', type=int, default=7)
    parser.add_argument('--word_dropout', type=int, default=0)

    parser.add_argument('--s_max', type=int, default=1)
    parser.add_argument('--n_max', type=int, default=30)

    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=10)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print(args)

    # trainer = TCNTrainer(args)
    # trainer.train()

    trainer = LSTMTrainer(args)
    trainer.train()
