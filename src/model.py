import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import time

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(1)

class NewsAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(NewsAttention, self).__init__()
        self.W = nn.Linear(embed_dim, num_heads)

    def forward(self, news_embeddings):
        attn_raw = self.W(news_embeddings)
        attn_weights = F.softmax(attn_raw, dim=0)
        headwise_user_emb = torch.einsum('nh,nd->hd', attn_weights, news_embeddings)
        user_embedding = headwise_user_emb.mean(dim=0)
        return user_embedding, attn_weights

class NeighborAggregation(nn.Module):
    def __init__(self, embed_dim):
        super(NeighborAggregation, self).__init__()
        self.W = nn.Linear(embed_dim, 1)

    def forward(self, neighbor_embeddings):
        attention_scores = self.W(neighbor_embeddings).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=0)
        neighbor_embedding = torch.sum(neighbor_embeddings * attention_weights.unsqueeze(-1), dim=0)
        return neighbor_embedding, attention_weights

class UserMLP(nn.Module):
    def __init__(self, embed_dim, dropout_rate=0.2):
        super(UserMLP, self).__init__()
        self.fc1 = nn.Linear(2 * embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.fc4 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, news_embedding, neighbor_embedding):
        combined = torch.cat((news_embedding, neighbor_embedding), dim=0)
        g1 = F.relu(self.fc1(combined))
        g1 = self.dropout(g1)
        g2 = F.relu(self.fc2(g1))
        g2 = self.dropout(g2)
        g3 = F.relu(self.fc3(g2))
        g3 = self.dropout(g3)
        h_i = self.fc4(g3)
        return h_i


class ScoringLayer(nn.Module):
    def __init__(self, embed_dim):
        super(ScoringLayer, self).__init__()
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, user_embedding, news_embedding):
        x = user_embedding * news_embedding
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze()


class EndToEndRecommendationModel(nn.Module):
    def __init__(self, embed_dim, news_embeddings, user_interactions, trusted_neighbors_dict):
        super(EndToEndRecommendationModel, self).__init__()
        self.news_embeddings = news_embeddings
        self.user_interactions = user_interactions
        self.trusted_neighbors_dict = trusted_neighbors_dict

        self.news_attention = NewsAttention(embed_dim)
        self.neighbor_attention = NeighborAggregation(embed_dim)
        self.user_mlp = UserMLP(embed_dim)
        self.scoring = ScoringLayer(embed_dim)

        self.cached_hiN = {}
        self.cached_hiS = {}
        self.cached_user_embeddings = {}

    def get_user_news_embeddings(self, user_id):
        news_ids = [x['news_id'] for x in self.user_interactions[user_id]]
        return torch.stack([self.news_embeddings[nid] for nid in news_ids[:50]])

    def get_neighbor_embeddings(self, user_id):
        neighbors = self.trusted_neighbors_dict.get(user_id, [])
        embeddings = []
        for nid in neighbors:
            if nid in self.cached_hiN:
                embeddings.append(self.cached_hiN[nid])
            elif nid in self.user_interactions:
                news_embs = self.get_user_news_embeddings(nid)
                emb = self.news_attention(news_embs)[0]
                embeddings.append(emb)
        if embeddings:
            return torch.stack(embeddings)
        else:
            return torch.zeros((1, next(iter(self.news_embeddings.values())).shape[0]))

    def forward(self, user_id, candidate_news_id):
        if user_id not in self.cached_user_embeddings:
            user_news_embs = self.get_user_news_embeddings(user_id)
            hiN, _ = self.news_attention(user_news_embs)
            self.cached_hiN[user_id] = hiN

            neighbor_embs = self.get_neighbor_embeddings(user_id)
            hiS, _ = self.neighbor_attention(neighbor_embs)
            self.cached_hiS[user_id] = hiS

            user_emb = self.user_mlp(hiN, hiS)
            self.cached_user_embeddings[user_id] = user_emb
        else:
            user_emb = self.cached_user_embeddings[user_id]

        news_emb = self.news_embeddings[candidate_news_id]
        return self.scoring(user_emb, news_emb)
