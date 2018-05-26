import time
import pickle
import argparse
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.models import *
from utils.dataset import *
from utils.loss import *
from utils.build_tag import *


class DebuggerSampler(object):
    def __init__(self, args):
        self.args = args

        self.vocab = self.__init_vocab()
        self.tagger = self.__init_tagger()
        self.transform = self._init_transform()
        self.data_loader = self.__init_data_loader(self.args.test_file_lits)
        self.model_state_dict = self.__load_mode_state_dict()

        # self.extractor = self.__init_visual_extractor()
        # self.mlc = self.__init_mlc()
        # self.co_attention = self.__init_co_attention()
        # self.sentence_model = self.__init_sentence_lstm()
        # self.word_model = self.__init_word_lstm()

        self.encoder = self._init_encoder()
        self.decoder = self._init_decoder()

        self.ce_criterion = self._init_ce_criterion()
        self.mse_criterion = self._init_mse_criterion()

    @staticmethod
    def _init_ce_criterion():
        return nn.CrossEntropyLoss(size_average=False, reduce=False)

    @staticmethod
    def _init_mse_criterion():
        return nn.MSELoss()

    def _to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def test(self):
        tag_loss, stop_loss, word_loss, loss = 0, 0, 0, 0
        self.encoder.eval()
        self.decoder.eval()

        for i, (images, _, label, captions, prob) in enumerate(self.data_loader):
            batch_tag_loss, batch_stop_loss, batch_word_loss, batch_loss = 0, 0, 0, 0
            images = self._to_var(images)

            visual_features = self.encoder.forward(images)
            # visual_features = self.extractor.forward(images)
            # tags, semantic_features = self.mlc.forward(visual_features)

            # batch_tag_loss = self.mse_criterion(tags, self._to_var(label, requires_grad=False)).sum()

            # sentence_states = None
            # prev_hidden_states = self._to_var(torch.zeros(images.shape[0], 1, 512))
            context = self._to_var(torch.Tensor(captions).long(), requires_grad=False)
            # prob_real = self._to_var(torch.Tensor(prob).long(), requires_grad=False)

            # for sentence_index in range(captions.shape[1]):
            #     ctx = self.co_attention.forward(visual_features, semantic_features, prev_hidden_states)
            #     topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
            #                                                                                prev_hidden_states,
            #                                                                                sentence_states)
            #     batch_stop_loss += self.ce_criterion(p_stop.squeeze(), prob_real[:, sentence_index]).sum()

                # Debugging...
                # print("Real Stop: {}".format(prob[:, sentence_index]))
                # print("Pred Stop: {}".format(p_stop))
                # print("")

            for word_index in range(1, captions.shape[2] - 1):
                    # context = self._to_var(torch.Tensor(captions[:, sentence_index, :word_index]).long(),
                    #                        requires_grad=False)
                    # real_word = self._to_var(torch.Tensor(captions[:, sentence_index, word_index]).long(),
                    #                          requires_grad=False)
                # word_mask = self._to_var((torch.Tensor(captions[:, sentence_index, word_index]) > 0).float(),
                #                              requires_grad=True)
                words = self.decoder.forward(visual_features, context[:, 0, :word_index])

                # Debugging...
                # print("Context:{}".format(context[:, 0, :word_index]))
                # print("Pred:{}".format(torch.max(words.squeeze(1), 1)[1]))
                # print("Real:{}".format(context[:, 0, word_index]))
                # print("")
                batch_loss += (self.ce_criterion(words, context[:, 0, word_index])).sum()

            # batch_loss = self.args.lambda_tag * batch_tag_loss \
            #              + self.args.lambda_stop * batch_stop_loss \
            #              + self.args.lambda_word * batch_word_loss

            # self.extractor.zero_grad()
            # self.mlc.zero_grad()
            # self.co_attention.zero_grad()
            # self.sentence_model.zero_grad()
            # self.word_model.zero_grad()

            # tag_loss += batch_tag_loss.data
            # stop_loss += batch_stop_loss.data
            # word_loss += batch_word_loss.data
            loss += batch_loss.data

            # print("tag loss:{}".format(batch_tag_loss))
            # print("stop loss:{}".format(batch_stop_loss))
            # print("word loss:{}".format(batch_word_loss))
        return tag_loss, stop_loss, word_loss, loss

    def sample(self):
        progress_bar = tqdm(self.data_loader, desc='Sampling')
        results = {}

        self.encoder.eval()
        self.decoder.eval()

        for images, image_id, label, captions, _ in progress_bar:
            images = self.__to_var(images, requires_grad=False)

            visual_features = self.encoder.forward(images)
            # tags, semantic_features = self.mlc.forward(visual_features)

            # sentence_states = None
            # prev_hidden_states = self.__to_var(torch.zeros(images.shape[0], 1, 512))
            pred_sentences = {}
            real_sentences = {}
            for i in image_id:
                pred_sentences[i] = {}
                real_sentences[i] = {}

            for i in range(1):
                # ctx = self.co_attention.forward(visual_features, semantic_features, prev_hidden_states)
                # topic, p_stop, hidden_state, sentence_states = self.sentence_model.forward(ctx,
                #                                                                           prev_hidden_states,
                #                                                                           sentence_states)
                # p_stop = p_stop.squeeze(1)
                # p_stop = torch.max(p_stop, 1)[1].unsqueeze(1)

                start_tokens = np.zeros((images.shape[0], 1))
                start_tokens[:, 0] = self.vocab('<start>')
                start_tokens = self.__to_var(torch.Tensor(start_tokens).long(), requires_grad=False)

                sampled_ids = self.decoder.sample(visual_features, start_tokens)
                # prev_hidden_states = hidden_state
                # sampled_ids = sampled_ids * p_stop

                for id, array in zip(image_id, sampled_ids):
                    pred_sentences[id][i] = self.__vec2sent(array)
                break

            for id, array in zip(image_id, captions):
                for i, sent in enumerate(array):
                    real_sentences[id][i] = self.__vec2sent(sent)

            for id in image_id:
                results[id] = {
                    # 'Real Tags': self.tagger.inv_tags2array(real_tag),
                    # 'Pred Tags': self.tagger.array2tags(torch.topk(pred_tag, self.args.k)[1].cpu().detach().numpy()),
                    'Pred Sent': pred_sentences[id],
                    'Real Sent': real_sentences[id]
                }

        self.__save_json(results)

    def __save_json(self, result):
        if not os.path.exists(self.args.result_path):
            os.makedirs(self.args.result_path)
        with open(os.path.join(self.args.result_path, '{}.json'.format(self.args.result_name)), 'w') as f:
            json.dump(result, f)

    def __load_mode_state_dict(self):
        try:
            model_state_dict = torch.load(self.args.load_model_path)
            print("[Load Model-{} Succeed!]".format(self.args.load_model_path))
            return model_state_dict
        except Exception as err:
            print("[Load Model Failed] {}".format(err))
            raise err

    def __init_tagger(self):
        return Tag()

    def __vec2sent(self, array):
        sampled_caption = []
        for word_id in array:
            word = self.vocab.get_word_by_id(word_id)
            if word == '<start>':
                continue
            if word == '<end>':
                break
            sampled_caption.append(word)
        return ' '.join(sampled_caption)

    def __init_vocab(self):
        with open(self.args.vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    def __init_data_loader(self, file_list):
        data_loader = get_loader(image_dir=self.args.image_dir,
                                 caption_json=self.args.caption_json,
                                 file_list=file_list,
                                 vocabulary=self.vocab,
                                 transform=self.transform,
                                 batch_size=self.args.batch_size,
                                 shuffle=False)
        return data_loader

    def _init_transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.resize),
            transforms.RandomCrop(self.args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        return transform

    def __to_var(self, x, requires_grad=True):
        if self.args.cuda:
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)

    def __init_visual_extractor(self):
        model = VisualFeatureExtractor()

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['extractor'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_mlc(self):
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

    def __init_co_attention(self):
        model = CoAttention(embed_size=self.args.embed_size,
                            hidden_size=self.args.hidden_size,
                            visual_size=self.args.visual_size)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['co_attention'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_sentence_lstm(self):
        model = SentenceLSTM(embed_size=self.args.embed_size,
                             hidden_size=self.args.hidden_size,
                             num_layers=self.args.sentence_num_layers)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['sentence_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    def __init_word_lstm(self):
        model = WordLSTM(vocab_size=len(self.vocab),
                         embed_size=self.args.embed_size,
                         hidden_size=self.args.hidden_size,
                         num_layers=self.args.word_num_layers)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['word_model'])

        if self.args.cuda:
            model = model.cuda()
        return model

    # Debugging
    def _init_encoder(self):
        model = EncoderCNN(embed_size=256)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['encoder'])

        if self.args.cuda:
            model = model.cuda()
        return model

    # Debugging
    def _init_decoder(self):
        model = DecoderRNN(embed_size=256,
                           hidden_size=512,
                           vocab_size=len(self.vocab),
                           num_layers=1)

        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict['decoder'])

        if self.args.cuda:
            model = model.cuda()
        return model

if __name__ == '__main__':
    model_dir = './report_models/NLC_debugger/20180526-07:46:59'

    import warnings
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--resize', type=int, default=256,
                        help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='not using pretrained model when training')
    parser.add_argument('--vocab_path', type=str, default='./data/debugging_vocab.pkl',
                        help='the path for vocabulary object')
    parser.add_argument('--image_dir', type=str, default='./data/images',
                        help='the path for images')
    parser.add_argument('--caption_json', type=str, default='./data/debugging_captions.json',
                        help='path for captions')
    parser.add_argument('--test_file_lits', type=str, default='./data/debugging.txt',
                        help='the path for test file list')
    parser.add_argument('--load_model_path', type=str, default=os.path.join(model_dir, 'best_loss.pth.tar'),
                        help='The path of loaded model')
    parser.add_argument('--result_path', type=str, default=os.path.join(model_dir, 'results'),
                        help='the path for storing results')
    parser.add_argument('--result_name', type=str, default='word_test',
                        help='the name of results')

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

    parser.add_argument('--s_max', type=int, default=2)
    parser.add_argument('--n_max', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--lambda_tag', type=float, default=10000)
    parser.add_argument('--lambda_stop', type=float, default=10)
    parser.add_argument('--lambda_word', type=float, default=1)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    sampler = DebuggerSampler(args)
    tag_loss, stop_loss, word_loss, loss = sampler.test()

    print("tag loss:{}".format(tag_loss))
    print("stop loss:{}".format(stop_loss))
    print("word loss:{}".format(word_loss))
    print("loss:{}".format(loss))

    sampler.sample()
