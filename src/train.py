def train_model(model, train_data, news_embeddings, labels_dict, epochs=10, K=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for i, (user_id, interactions) in enumerate(train_data.items()):
            model.cached_hiN.clear()
            model.cached_hiS.clear()
            model.cached_user_embeddings.clear()

            pos_news_ids = [x['news_id'] for x in interactions]
            pos_scores = torch.stack([model(user_id, nid) for nid in pos_news_ids])

            neg_news_ids = negative_sampling(
                train_data, news_embeddings, labels_dict,
                user_id, pos_news_ids, K
            )

            neg_scores = torch.stack([model(user_id, nid) for nid in neg_news_ids])


            pos_scores_expanded = pos_scores.unsqueeze(1)
            neg_scores_expanded = neg_scores.unsqueeze(0).expand(len(pos_scores), -1)
            all_scores = torch.cat([pos_scores_expanded, neg_scores_expanded], dim=1)
            exp_scores = torch.exp(all_scores)
            pi = exp_scores[:, 0] / torch.sum(exp_scores, dim=1)

            user_loss = -torch.sum(torch.log(pi)) / len(pos_news_ids)

            optimizer.zero_grad()
            user_loss.backward()
            optimizer.step()
            total_loss += user_loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_data):.4f}")



model = EndToEndRecommendationModel(
    embed_dim=embed_dim,
    news_embeddings=news_embeddings,
    user_interactions=user_interactions,
    trusted_neighbors_dict=trusted_neighbors_dict
)




subset_train_data = dict(list(train_data.items())[:5])   # sample data
     

train_data = dict(list(train_data.items())[:])   # full data


train_model(model, train_data, news_embeddings, labels_dict, epochs=20)
