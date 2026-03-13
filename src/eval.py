def evaluate_model(model, test_data, news_embeddings, labels_dict, K=5):
    model.eval()
    model.cached_hiN.clear()
    model.cached_hiS.clear()
    model.cached_user_embeddings.clear()

    hits, reciprocal_ranks, ndcg_scores, tf_at_k = 0, [], [], 0
    y_true, y_scores = [], []

    with torch.no_grad():
        for user_id, test_interaction in test_data.items():
            test_news_id = test_interaction['news_id']

            news_scores = {nid: model(user_id, nid).item() for nid in news_embeddings}
            sorted_news = sorted(news_scores, key=news_scores.get, reverse=True)

            if test_news_id in sorted_news[:K]:
                hits += 1

            y_true.append(1)
            y_scores.append(news_scores[test_news_id])
            y_true.extend([0] * (len(news_embeddings) - 1))
            y_scores.extend([news_scores[nid] for nid in news_embeddings if nid != test_news_id])

            rank = sorted_news.index(test_news_id) + 1
            reciprocal_ranks.append(1 / rank)
            dcg = 1 / np.log2(rank + 1)
            idcg = 1 / np.log2(2)
            ndcg_scores.append(dcg / idcg)

            top_k_news = sorted_news[:K]
            fake_news_count = sum(1 for nid in top_k_news if labels_dict[nid] == 'fake')
            tf_at_k += fake_news_count / K

    hit_rate = hits / len(test_data)
    rocauc = roc_auc_score(y_true, y_scores)
    return hit_rate, rocauc, np.mean(reciprocal_ranks), np.mean(ndcg_scores), tf_at_k / len(test_data)


  #subset_test_data = dict(list(test_data.items())[:5]) # sample data
     

test_data = dict(list(test_data.items())[:]) # full data
     

hit_rate, rocauc, mrr, ndcg, tf_at_k = evaluate_model(
    model=model,
    test_data=test_data,
    news_embeddings=news_embeddings,
    labels_dict=labels_dict,
    K=10
)
print(f"Hit Rate : {hit_rate * 100:.2f}%")
print(f"ROCAUC: {rocauc:.4f}")
print(f"MRR: {mrr:.4f}")
print(f"nDCG: {ndcg:.4f}")
print(f"TF: {tf_at_k:.4f}")
     
     
