import torch
from torch import nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.autograd import Variable

import numpy as np



class TransRec(nn.Module):
    def __init__(self, embedding_dim, user2id, id2user, poi2id, id2poi, cuda=False):
        super(TransRec, self).__init__()

        self.user2id = user2id
        self.id2user = id2user
        self.poi2id = poi2id
        self.id2poi = id2poi
        self.n_users = len(id2user)
        self.n_poi = len(id2poi)

        self.embedding_dim = embedding_dim
        self.poi_embedding = nn.Embedding(self.n_poi, self.embedding_dim)           # 1
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)        # 2
        self.user_global_embedding = nn.Embedding(1, self.embedding_dim)            # 3
        self.poi_bias = nn.Embedding(self.n_poi, 1)                                 # 4
        self.init_weights()
        self.first_prediction = True

        self.cuda = cuda
        if self.cuda:
            self.float_one = Variable(torch.cuda.FloatTensor([1.]))
            self.int_zero = Variable(torch.cuda.LongTensor([0]))
            self.float_zero = Variable(torch.cuda.FloatTensor([0.]))
        else:
            self.float_one = torch.FloatTensor([1.])
            self.int_zero = torch.LongTensor([0])
            self.float_zero = torch.FloatTensor([0.])

    def init_weights(self):
        self.poi_embedding.weight.data = F.normalize(self.poi_embedding.weight.data)                        # 1
        self.user_embedding.weight.data.zero_()                                                             # 2
        self.user_global_embedding.weight.data = F.normalize(self.user_global_embedding.weight.data)        # 3
        self.poi_bias.weight.data.zero_()                                                                   # 4

    def forward(self, user_id, prev_id, pos_id, neg_id):


        user_personal = self.user_embedding(user_id)                        # [batch_size x embedding_dim]
        user_global = self.user_global_embedding(torch.LongTensor([0]))     # [1          x embedding_dim]
        prev_poi = self.poi_embedding(prev_id)                              # [batch_size x embedding_dim]
        pos_poi = self.poi_embedding(pos_id)                                # [batch_size x embedding_dim]
        neg_poi = self.poi_embedding(neg_id)                                # [batch_size x embedding_dim]
        pos_poi_bias = self.poi_bias(pos_id)                                # [batch_size x             1]
        neg_poi_bias = self.poi_bias(neg_id)                                # [batch_size x             1]

        pos_probability = (pos_poi_bias -                                   # [batch_size x             1]
                           ((prev_poi + user_personal + user_global).sub_(pos_poi)).norm(dim=1, keepdim=True))
        neg_probability = (neg_poi_bias -                                   # [batch_size x             1]
                           ((prev_poi + user_personal + user_global).sub_(neg_poi)).norm(dim=1, keepdim=True))

        objective = (pos_probability - neg_probability).squeeze()           # [batch_size]
        # normalizing =====================
        self.poi_embedding.weight.data[prev_id]/= torch.maximum(self.float_one,
                                                self.poi_embedding.weight.data[prev_id].norm(dim=1, keepdim=True))
        self.poi_embedding.weight.data[pos_id]/= torch.maximum(self.float_one,
                                                self.poi_embedding.weight.data[pos_id].norm(dim=1, keepdim=True))
        self.poi_embedding.weight.data[neg_id]/= torch.maximum(self.float_one,
                                                self.poi_embedding.weight.data[neg_id].norm(dim=1, keepdim=True))
        # [batch_size x embedding_dim]
        # =================================
        return objective

    def predict(self, user_id, pre_poi):
        user_id = torch.LongTensor([user_id])
        pre_poi = torch.LongTensor([pre_poi])
        if  self.first_prediction == True:
            poi_biases = (self.poi_bias.weight.data.max() -  self.poi_bias.weight.data).sqrt()
            self.poi_vectors = torch.cat((self.poi_embedding.weight.data,
                       poi_biases), 1).detach().numpy()

            self.KNN = NearestNeighbors(n_neighbors=50, algorithm='ball_tree')
            self.KNN.fit(self.poi_vectors)
            self.first_prediction = False
        translation = (self.poi_embedding(pre_poi) +
                       self.user_embedding(user_id) +
                       self.user_global_embedding(self.int_zero))
        translation = torch.cat((translation,self.float_zero), 1).detach().numpy()

        _, indices = self.KNN.kneighbors(translation)
        indices = indices[0]
        indices = indices[indices!=int(pre_poi[0])]
        return indices










