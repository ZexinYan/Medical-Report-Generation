import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.autograd import Variable
from torchvision.models.vgg import model_urls as vgg_model_urls
import torchvision.models as models

from utils.tcn import *


class DenseNet121(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=classes, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class DenseNet161(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet161, self).__init__()
        self.model = torchvision.models.densenet161(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class DenseNet169(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet169, self).__init__()
        self.model = torchvision.models.densenet169(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class DenseNet201(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(DenseNet201, self).__init__()
        self.model = torchvision.models.densenet201(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet34, self).__init__()
        self.model = torchvision.models.resnet34(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet101, self).__init__()
        self.model = torchvision.models.resnet101(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class ResNet152(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(ResNet152, self).__init__()
        self.model = torchvision.models.resnet152(pretrained=pretrained)
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class VGG19(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(VGG19, self).__init__()
        self.model = torchvision.models.vgg19_bn(pretrained=pretrained)
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            self.__init_linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            self.__init_linear(in_features=4096, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class VGG(nn.Module):
    def __init__(self, tags_num):
        super(VGG, self).__init__()
        vgg_model_urls['vgg19'] = vgg_model_urls['vgg19'].replace('https://', 'http://')
        self.vgg19 = models.vgg19(pretrained=True)
        vgg19_classifier = list(self.vgg19.classifier.children())[:-1]
        self.classifier = nn.Sequential(*vgg19_classifier)
        self.fc = nn.Linear(4096, tags_num)
        self.fc.apply(self.init_weights)
        self.bn = nn.BatchNorm1d(tags_num, momentum=0.1)
#        self.init_weights()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            self.fc.weight.data.normal_(0, 0.1)
            self.fc.bias.data.fill_(0)

    def forward(self, images) -> object:
        """

        :rtype: object
        """
        visual_feats = self.vgg19.features(images)
        tags_classifier = visual_feats.view(visual_feats.size(0), -1)
        tags_classifier = self.bn(self.fc(self.classifier(tags_classifier)))
        return tags_classifier


class InceptionV3(nn.Module):
    def __init__(self, classes=156, pretrained=True):
        super(InceptionV3, self).__init__()
        self.model = torchvision.models.inception_v3(pretrained=pretrained)
        num_in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            self.__init_linear(in_features=num_in_features, out_features=classes),
            # nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.model(x)
        return x


class CheXNetDenseNet121(nn.Module):
    def __init__(self, classes=14, pretrained=True):
        super(CheXNetDenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        num_in_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(in_features=num_in_features, out_features=classes, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class CheXNet(nn.Module):
    def __init__(self, classes=156):
        super(CheXNet, self).__init__()
        self.densenet121 = CheXNetDenseNet121(classes=14)
        self.densenet121 = torch.nn.DataParallel(self.densenet121).cuda()
        self.densenet121.load_state_dict(torch.load('./models/CheXNet.pth.tar')['state_dict'])
        self.densenet121.module.densenet121.classifier = nn.Sequential(
            self.__init_linear(1024, classes),
            nn.Sigmoid()
        )

    def __init_linear(self, in_features, out_features):
        func = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        func.weight.data.normal_(0, 0.1)
        return func

    def forward(self, x) -> object:
        """

        :rtype: object
        """
        x = self.densenet121(x)
        return x


class ModelFactory(object):
    def __init__(self, model_name, pretrained, classes):
        self.model_name = model_name
        self.pretrained = pretrained
        self.classes = classes

    def create_model(self):
        if self.model_name == 'VGG19':
            _model = VGG19(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet121':
            _model = DenseNet121(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet161':
            _model = DenseNet161(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet169':
            _model = DenseNet169(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'DenseNet201':
            _model = DenseNet201(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'CheXNet':
            _model = CheXNet(classes=self.classes)
        elif self.model_name == 'ResNet18':
            _model = ResNet18(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet34':
            _model = ResNet34(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet50':
            _model = ResNet50(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet101':
            _model = ResNet101(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'ResNet152':
            _model = ResNet152(pretrained=self.pretrained, classes=self.classes)
        elif self.model_name == 'VGG':
            _model = VGG(tags_num=self.classes)
        else:
            _model = CheXNet(classes=self.classes)

        return _model


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, pretrained=True):
        super(EncoderCNN, self).__init__()
        # TODO Extract Image features from CNN based on other models
        resnet = models.resnet152(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.1)
        self.__init_weights()

    def __init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, images) -> object:
        """

        :rtype: object
        """
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, n_max=50):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.__init_weights()
        self.n_max = n_max

    def __init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions) -> object:
        """

        :rtype: object
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs

    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings.unsqueeze(1)

        for i in range(self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            predicted = torch.max(outputs, 1)[1]
            sampled_ids[:, i] = predicted
            predicted = predicted.unsqueeze(1)
        return sampled_ids


class VisualFeatureExtractor(nn.Module):
    def __init__(self, pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        resnet = models.resnet152(pretrained=pretrained)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.out_features = resnet.fc.in_features

    def forward(self, images) -> object:
        """

        :rtype: object
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features


class MLC(nn.Module):
    def __init__(self, classes=156, sementic_features_dim=512, fc_in_features=2048, k=10):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(in_features=fc_in_features, out_features=classes)
        self.embed = nn.Embedding(classes, sementic_features_dim)
        self.k = k
        self.softmax = nn.Softmax()

    def forward(self, visual_features) -> object:
        """

        :rtype: object
        """
        tags = self.softmax(self.classifier(visual_features))
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features


class CoAttention(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, visual_size=2048):
        super(CoAttention, self).__init__()
        self.W_v = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v = nn.BatchNorm1d(num_features=visual_size, momentum=0.1)

        self.W_v_h = nn.Linear(in_features=hidden_size, out_features=visual_size)
        self.bn_v_h = nn.BatchNorm1d(num_features=visual_size, momentum=0.1)

        self.W_v_att = nn.Linear(in_features=visual_size, out_features=visual_size)
        self.bn_v_att = nn.BatchNorm1d(num_features=visual_size, momentum=0.1)

        self.W_a = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a = nn.BatchNorm1d(num_features=10, momentum=0.1)

        self.W_a_h = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.bn_a_h = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_a_att = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        self.bn_a_att = nn.BatchNorm1d(num_features=10, momentum=0.1)

        self.W_fc = nn.Linear(in_features=visual_size + hidden_size, out_features=embed_size)
        self.bn_fc = nn.BatchNorm1d(num_features=embed_size, momentum=0.1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, visual_features, semantic_features, h_sent) -> object:
        """
        only training
        :rtype: object
        """
        W_v = self.bn_v(self.W_v(visual_features))
        W_v_h = self.bn_v_h(self.W_v_h(h_sent.squeeze(1)))

        alpha_v = self.softmax(self.bn_v_att(self.W_v_att(self.tanh(W_v + W_v_h))))
        v_att = torch.mul(alpha_v, visual_features)
        # v_att = torch.mul(alpha_v, visual_features).sum(1).unsqueeze(1)

        W_a_h = self.bn_a_h(self.W_a_h(h_sent))
        W_a = self.bn_a(self.W_a(semantic_features))
        alpha_a = self.softmax(self.bn_a_att(self.W_a_att(self.tanh(torch.add(W_a_h, W_a)))))
        a_att = torch.mul(alpha_a, semantic_features).sum(1)
        # a_att = (alpha_a * semantic_features).sum(1)
        ctx = self.bn_fc(self.W_fc(torch.cat([v_att, a_att], dim=1)))
        # return self.W_fc(self.bn_fc(torch.cat([v_att, a_att], dim=1)))
        return ctx, v_att


class SentenceLSTM(nn.Module):
    def __init__(self, embed_size=512, hidden_size=512, num_layers=1):
        super(SentenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        self.W_t_h = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_t_h = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_t_ctx = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.bn_t_ctx = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_stop_s_1 = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_stop_s_1 = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_stop_s = nn.Linear(in_features=hidden_size, out_features=embed_size, bias=True)
        self.bn_stop_s = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_stop = nn.Linear(in_features=embed_size, out_features=2, bias=True)
        self.bn_stop = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_topic = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.bn_topic = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.W_topic_2 = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.bn_topic_2 = nn.BatchNorm1d(num_features=1, momentum=0.1)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    # def forward(self, ctx, prev_hidden_state, states=None) -> object:
    #     """
    #     Only training
    #     :rtype: object
    #     """
    #     ctx = ctx.unsqueeze(1)
    #     hidden_state, states = self.lstm(ctx, states)
    #     topic = self.bn_topic(self.W_topic(self.sigmoid(self.bn_t_h(self.W_t_h(hidden_state))
    #                                                     + self.bn_t_ctx(self.W_t_ctx(ctx)))))
    #     p_stop = self.bn_stop(self.W_stop(self.sigmoid(self.bn_stop_s_1(self.W_stop_s_1(prev_hidden_state))
    #                                       + self.bn_stop_s(self.W_stop_s(hidden_state)))))
    #     return topic, p_stop, hidden_state, states

    def forward(self, ctx, prev_hidden_state, states=None) -> object:
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


class SentenceTCN(nn.Module):
    def __init__(self,
                 input_channel=10,
                 embed_size=512,
                 output_size=512,
                 nhid=512,
                 levels=8,
                 kernel_size=2,
                 dropout=0):
        super(SentenceTCN, self).__init__()
        channel_sizes = [nhid] * levels
        self.tcn = TCN(input_size=input_channel,
                       output_size=output_size,
                       num_channels=channel_sizes,
                       kernel_size=kernel_size,
                       dropout=dropout)
        self.W_t_h = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_t_ctx = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_stop_s_1 = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_stop_s = nn.Linear(in_features=output_size, out_features=embed_size, bias=True)
        self.W_stop = nn.Linear(in_features=embed_size, out_features=2, bias=True)
        self.t_w = nn.Linear(in_features=5120, out_features=2, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, ctx, prev_output) -> object:
        """

        :rtype: object
        """
        output = self.tcn.forward(ctx)
        topic = self.tanh(self.W_t_h(output) + self.W_t_ctx(ctx[:, -1, :]).squeeze(1))
        p_stop = self.W_stop(self.tanh(self.W_stop_s_1(prev_output) + self.W_stop_s(output)))
        return topic, p_stop, output


class WordLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, n_max=50):
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

    def forward(self, topic_vec, captions) -> object:
        """

        :rtype: object
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((topic_vec, embeddings), 1)
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden[:, -1, :])
        return outputs

    def val(self, features, start_tokens):
        samples = torch.zeros((np.shape(features)[0], self.n_max, self.vocab_size))
        samples[:, 0, start_tokens[0]] = 1
        predicted = start_tokens
        embeddings = features
        embeddings = embeddings

        for i in range(1, self.n_max):
            predicted = self.embed(predicted)
            embeddings = torch.cat([embeddings, predicted], dim=1)
            hidden_states, _ = self.lstm(embeddings)
            hidden_states = hidden_states[:, -1, :]
            outputs = self.linear(hidden_states)
            samples[:, i, :] = outputs
            predicted = torch.max(outputs, 1)[1]
            predicted = predicted.unsqueeze(1)
        return samples

    def sample(self, features, start_tokens):
        sampled_ids = np.zeros((np.shape(features)[0], self.n_max))
        sampled_ids[:, 0] = start_tokens.view(-1,)
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


class WordTCN(nn.Module):
    def __init__(self,
                 input_channel=11,
                 vocab_size=1000,
                 embed_size=512,
                 output_size=512,
                 nhid=512,
                 levels=8,
                 kernel_size=2,
                 dropout=0,
                 n_max=50):
        super(WordTCN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.output_size = output_size
        channel_sizes = [nhid] * levels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_max = n_max
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.W_out = nn.Linear(in_features=output_size, out_features=vocab_size, bias=True)
        self.tcn = TCN(input_size=input_channel,
                       output_size=output_size,
                       num_channels=channel_sizes,
                       kernel_size=kernel_size,
                       dropout=dropout)

    def forward(self, topic_vec, captions) -> object:
        """

        :rtype: object
        """
        captions = self.embed(captions)
        embeddings = torch.cat([topic_vec, captions], dim=1)
        output = self.tcn.forward(embeddings)
        words = self.W_out(output)
        return words


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    images = torch.randn((4, 3, 224, 224))
    captions = torch.ones((4, 10)).long()
    hidden_state = torch.randn((4, 1, 512))

    print("images:{}".format(images.shape))
    print("captions:{}".format(captions.shape))
    print("hidden_states:{}".format(hidden_state.shape))

    extractor = VisualFeatureExtractor()
    visual_features = extractor.forward(images)
    print("visual_features:{}".format(visual_features.shape))

    mlc = MLC()
    tags, semantic_features = mlc.forward(visual_features)
    print("tags:{}".format(tags.shape))
    print("semantic_features:{}".format(semantic_features.shape))

    co_att = CoAttention()
    ctx, v_att = co_att.forward(visual_features, semantic_features, hidden_state)
    print("ctx:{}".format(ctx.shape))
    print("v_att:{}".format(v_att.shape))

    sent_lstm = SentenceLSTM()
    topic, p_stop, hidden_state, states = sent_lstm.forward(ctx, hidden_state)
    print("Topic:{}".format(topic.shape))
    print("P_STOP:{}".format(p_stop.shape))

    word_lstm = WordLSTM(embed_size=512, hidden_size=512, vocab_size=100, num_layers=1)
    words = word_lstm.forward(topic, captions)
    print("words:{}".format(words.shape))

    # Expected Output
    # images: torch.Size([4, 3, 224, 224])
    # captions: torch.Size([4, 1, 10])
    # hidden_states: torch.Size([4, 1, 512])
    # visual_features: torch.Size([4, 2048, 7, 7])
    # tags: torch.Size([4, 156])
    # semantic_features: torch.Size([4, 10, 512])
    # ctx: torch.Size([4, 512])
    # Topic: torch.Size([4, 1, 512])
    # P_STOP: torch.Size([4, 1, 2])
    # words: torch.Size([4, 1000])

    # images = torch.randn((4, 3, 224, 224))
    # captions = torch.ones((4, 3, 10)).long()
    # prev_outputs = torch.randn((4, 512))
    # now_words = torch.ones((4, 1))
    #
    # ctx_records = torch.zeros((4, 10, 512))
    # captions = torch.zeros((4, 10)).long()
    #
    # print("images:{}".format(images.shape))
    # print("captions:{}".format(captions.shape))
    # print("hidden_states:{}".format(prev_outputs.shape))
    #
    # extractor = VisualFeatureExtractor()
    # visual_features = extractor.forward(images)
    # print("visual_features:{}".format(visual_features.shape))
    #
    # mlc = MLC()
    # tags, semantic_features = mlc.forward(visual_features)
    # print("tags:{}".format(tags.shape))
    # print("semantic_features:{}".format(semantic_features.shape))
    #
    # co_att = CoAttention()
    # ctx = co_att.forward(visual_features, semantic_features, prev_outputs)
    # print("ctx:{}".format(ctx.shape))
    #
    # ctx_records[:, 0, :] = ctx
    #
    # sent_tcn = SentenceTCN()
    # topic, p_stop, prev_outputs = sent_tcn.forward(ctx_records, prev_outputs)
    # print("Topic:{}".format(topic.shape))
    # print("P_STOP:{}".format(p_stop.shape))
    # print("Prev_Outputs:{}".format(prev_outputs.shape))
    #
    # captions[:, 0] = now_words.view(-1,)
    #
    # word_tcn = WordTCN()
    # words = word_tcn.forward(topic, captions)
    # print("words:{}".format(words.shape))

