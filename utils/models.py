import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
import torchvision.models as models


class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.model, self.out_features, self.avg_func, self.bn, self.linear = self.__get_model()
        self.activation = nn.ReLU()

    def __get_model(self):
        model = None
        out_features = None
        func = None
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            modules = list(resnet.children())[:-2]
            model = nn.Sequential(*modules)
            out_features = resnet.fc.in_features
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        linear = nn.Linear(in_features=out_features, out_features=out_features)
        bn = nn.BatchNorm1d(num_features=out_features, momentum=0.1)
        return model, out_features, func, bn, linear

    def forward(self, images):
        """
        :param images:
        :return:
        """
        visual_features = self.model(images)
        avg_features = self.avg_func(visual_features).squeeze()
        # avg_features = self.activation(self.bn(self.linear(avg_features)))
        return visual_features, avg_features


class MLC(nn.Module):
    def __init__(self,
                 classes=156,
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.softmax(self.classifier(avg_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 visual_size=2048,
                 k=10,
                 momentum=0.1):
        super(CoAttention, self).__init__()
        self.version = version
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=momentum)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=k, momentum=momentum)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_att = nn.BatchNorm1d(num_features=k, momentum=momentum)

        # self.W_fc = nn.Linear(in_features=visual_size, out_features=embed_size)  # for v3
        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=momentum)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.__init_weight()

    def __init_weight(self):
        self.W_v.weight.data.uniform_(-0.1, 0.1)
        self.W_v.bias.data.fill_(0)

        self.W_v_h.weight.data.uniform_(-0.1, 0.1)
        self.W_v_h.bias.data.fill_(0)

        self.W_v_att.weight.data.uniform_(-0.1, 0.1)
        self.W_v_att.bias.data.fill_(0)

        self.W_a.weight.data.uniform_(-0.1, 0.1)
        self.W_a.bias.data.fill_(0)

        self.W_a_h.weight.data.uniform_(-0.1, 0.1)
        self.W_a_h.bias.data.fill_(0)

        self.W_a_att.weight.data.uniform_(-0.1, 0.1)
        self.W_a_att.bias.data.fill_(0)

        self.W_fc.weight.data.uniform_(-0.1, 0.1)
        self.W_fc.bias.data.fill_(0)

    def forward(self, avg_features, semantic_features, h_sent):
        if self.version == 'v1':
            return self.v1(avg_features, semantic_features, h_sent)
        elif self.version == 'v2':
            return self.v2(avg_features, semantic_features, h_sent)
        elif self.version == 'v3':
            return self.v3(avg_features, semantic_features, h_sent)
        elif self.version == 'v4':
            return self.v4(avg_features, semantic_features, h_sent)
        elif self.version == 'v5':
            return self.v5(avg_features, semantic_features, h_sent)

    def v1(self, avg_features, semantic_features, h_sent) -> object:
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v2(self, avg_features, semantic_features, h_sent) -> object:
        """
        no bn
        :rtype: object
        """
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v3(self, avg_features, semantic_features, h_sent) -> object:
        """

        :rtype: object
        """
        W_v = self.bn_v(self.W_v(avg_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.W_v_att(self.tanh(W_v + W_v_h)))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v4(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(torch.add(W_v, W_v_h))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(torch.add(W_a_h, W_a))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a

    def v5(self, avg_features, semantic_features, h_sent):
        W_v = self.W_v(avg_features)
        W_v_h = self.W_v_h(h_sent.squeeze(1))

        alpha_v = self.softmax(self.W_v_att(self.tanh(self.bn_v(torch.add(W_v, W_v_h)))))
        v_att = torch.mul(alpha_v, avg_features)

        W_a_h = self.W_a_h(h_sent)
        W_a = self.W_a(semantic_features)
        alpha_a = self.softmax(self.W_a_att(self.tanh(self.bn_a(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)

        ctx = self.W_fc(torch.cat([v_att, a_att], dim=1))

        return ctx, alpha_v, alpha_a


class SentenceLSTM(nn.Module):
    def __init__(self,
                 version='v1',
                 embed_size=512,
                 hidden_size=512,
                 num_layers=1,
                 dropout=0.3,
                 momentum=0.1):
        super(SentenceLSTM, self).__init__()
        self.version = version

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout)

        self.W_t_h = nn.Linear(in_features=hidden_size,
                               out_features=embed_size,
                               bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_t_ctx = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s_1 = nn.Linear(in_features=hidden_size,
                                    out_features=embed_size,
                                    bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop_s = nn.Linear(in_features=hidden_size,
                                  out_features=embed_size,
                                  bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_stop = nn.Linear(in_features=embed_size,
                                out_features=2,
                                bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.W_topic = nn.Linear(in_features=embed_size,
                                 out_features=embed_size,
                                 bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=momentum)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.__init_weight()

    def __init_weight(self):
        self.W_t_h.weight.data.uniform_(-0.1, 0.1)
        self.W_t_h.bias.data.fill_(0)

        self.W_t_ctx.weight.data.uniform_(-0.1, 0.1)
        self.W_t_ctx.bias.data.fill_(0)

        self.W_stop_s_1.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s_1.bias.data.fill_(0)

        self.W_stop_s.weight.data.uniform_(-0.1, 0.1)
        self.W_stop_s.bias.data.fill_(0)

        self.W_stop.weight.data.uniform_(-0.1, 0.1)
        self.W_stop.bias.data.fill_(0)

        self.W_topic.weight.data.uniform_(-0.1, 0.1)
        self.W_topic.bias.data.fill_(0)

    def forward(self, ctx, prev_hidden_state, states=None) -> object:
        """
        :rtype: object
        """
        if self.version == 'v1':
            return self.v1(ctx, prev_hidden_state, states)
        elif self.version == 'v2':
            return self.v2(ctx, prev_hidden_state, states)
        elif self.version == 'v3':
            return self.v3(ctx, prev_hidden_state, states)

    def v1(self, ctx, prev_hidden_state, states=None):
        """
        v1 (only training)
        :param ctx:
        :param prev_hidden_state:
        :param states:
        :return:
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.sigmoid(self.bn_t_h(self.W_t_h(hidden_state))
                                          + self.bn_t_ctx(self.W_t_ctx(ctx))))
        p_stop = self.W_stop(self.sigmoid(self.bn_stop_s_1(self.W_stop_s_1(prev_hidden_state))
                                          + self.bn_stop_s(self.W_stop_s(hidden_state))))
        return topic, p_stop, hidden_state, states

    def v2(self, ctx, prev_hidden_state, states=None):
        """
        v2
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.bn_topic(self.W_topic(self.tanh(self.bn_t_h(self.W_t_h(hidden_state)
                                                                 + self.W_t_ctx(ctx)))))
        p_stop = self.bn_stop(self.W_stop(self.tanh(self.bn_stop_s(self.W_stop_s_1(prev_hidden_state)
                                                                   + self.W_stop_s(hidden_state)))))
        return topic, p_stop, hidden_state, states

    def v3(self, ctx, prev_hidden_state, states=None):
        """
        v3
        :rtype: object
        """
        ctx = ctx.unsqueeze(1)
        hidden_state, states = self.lstm(ctx, states)
        topic = self.W_topic(self.tanh(self.W_t_h(hidden_state) + self.W_t_ctx(ctx)))
        p_stop = self.W_stop(self.tanh(self.W_stop_s_1(prev_hidden_state) + self.W_stop_s(hidden_state)))
        return topic, p_stop, hidden_state, states


class WordLSTM(nn.Module):
    def __init__(self,
                 embed_size,
                 hidden_size,
                 vocab_size,
                 num_layers,
                 n_max=50):
        super(WordLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max
        self.vocab_size = vocab_size

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, topic_vec, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs

    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        sampled_ids[:, 0] = start_tokens.view(-1, )
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings

        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return sampled_ids


if __name__ == '__main__':
    import torchvision.transforms as transforms

    import warnings
    warnings.filterwarnings("ignore")
#
    extractor = VisualFeatureExtractor(model_name='resnet152')
    mlc = MLC(fc_in_features=extractor.out_features)
    co_att = CoAttention(visual_size=extractor.out_features)
    sent_lstm = SentenceLSTM()
    word_lstm = WordLSTM(embed_size=512, hidden_size=512, vocab_size=100, num_layers=1)

    images = torch.randn((4, 3, 224, 224))
    captions = torch.ones((4, 10)).long()
    hidden_state = torch.randn((4, 1, 512))

    # # image_file = '../data/images/CXR2814_IM-1239-1001.png'
#     # # images = Image.open(image_file).convert('RGB')
#     # # captions = torch.ones((1, 10)).long()
#     # # hidden_state = torch.randn((10, 512))
# #
# norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.TenCrop(224),
#     transforms.Lambda(lambda crops: torch.stack([norm(transforms.ToTensor()(crop)) for crop in crops])),
# ])

# images = transform(images)
# images.unsqueeze_(0)
#
# # bs, ncrops, c, h, w = images.size()
# # images = images.view(-1, c, h, w)
#
    print("images:{}".format(images.shape))
    print("captions:{}".format(captions.shape))
    print("hidden_states:{}".format(hidden_state.shape))

    visual_features, avg_features = extractor.forward(images)

    print("visual_features:{}".format(visual_features.shape))
    print("avg features:{}".format(avg_features.shape))

    tags, semantic_features = mlc.forward(avg_features)

    print("tags:{}".format(tags.shape))
    print("semantic_features:{}".format(semantic_features.shape))

    ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features, hidden_state)

    print("ctx:{}".format(ctx.shape))
    print("alpht_v:{}".format(alpht_v.shape))
    print("alpht_a:{}".format(alpht_a.shape))

    topic, p_stop, hidden_state, states = sent_lstm.forward(ctx, hidden_state)
    # p_stop_avg = p_stop.view(bs, ncrops, -1).mean(1)

    print("Topic:{}".format(topic.shape))
    print("P_STOP:{}".format(p_stop.shape))
    # print("P_stop_avg:{}".format(p_stop_avg.shape))

    words = word_lstm.forward(topic, captions)
    print("words:{}".format(words.shape))

    cam = torch.mul(visual_features, alpht_v.view(alpht_v.shape[0], alpht_v.shape[1], 1, 1)).sum(1)
    cam.squeeze_()
    cam = cam.cpu().data.numpy()
    for i in range(cam.shape[0]):
        heatmap = cam[i]
        heatmap = heatmap / np.max(heatmap)
        print(heatmap.shape)
