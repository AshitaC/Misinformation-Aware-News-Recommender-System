# Misinformation-Aware-News-Recommender-System
Implementation and validation of Fake News Aware Recommendation System

A comprehensive implementation of the Fake News Aware Recommender (FANAR) system that mitigates misinformation spread while maintaining high-quality news recommendations through hybrid user modeling and attention mechanisms.

## Overview
FANAR addresses the critical challenge of misinformation in news recommendation systems by implementing a hybrid user modeling approach that combines:

News-space modeling: User preferences based on direct news interactions
Neighbor-space modeling: Social network analysis with reliability filtering
Attention mechanisms: Intelligent aggregation of user preferences
BERT embeddings: Semantic understanding of news content

## Key Achievements

48% improvement in ranking quality (MRR) over traditional collaborative filtering

45% reduction in fake news recommendations

Successfully validated on benchmark datasets (PolitiFact, FakeHealth)

Ablation studies to evaluate impact of individual components

## Features

-  Semantic News Understanding: BERT-based news embeddings for rich content representation
- Social Network Integration: Neighbor-based filtering with reliability scoring
- Attention Mechanisms: Learnable attention weights for optimal user preference aggregation
- Comprehensive Evaluation: Multiple metrics including MRR, Hit Rate, nDCG, and Fake News Ratio
- Ablation Studies: Systematic analysis of model components
- Baseline Comparisons: Evaluation against 7 traditional recommendation algorithms

## Architecture
<img width="737" height="451" alt="image" src="https://github.com/user-attachments/assets/7d1541c8-6f98-4ea8-b4d5-8b4ffea19d8c" />


# Datasets
Supported Datasets
1. PolitiFact (FakeNewsNet)

Users: 1,028 Twitter users
News Articles: 542 (322 fake, 220 real)
Interactions: 20,265 user-news interactions
Social Connections: 3,021 following relationships

2. FakeHealth HealthStory

Users: 5,406 Twitter users
News Articles: 1,690 (472 fake, 1,218 real)
Interactions: 120,124 user-news interactions
Social Connections: 4,102 following relationships

# Results

PolitiFact Dataset Performance


<img width="571" height="227" alt="image" src="https://github.com/user-attachments/assets/d1039c37-de53-4aed-a503-9012c584fc9c" />


Key Improvements

48% improvement in MRR over strongest baseline (User-User CF)

45% reduction in fake news recommendations compared to traditional methods

Consistent performance across multiple evaluation metrics

Ablation Study Results

<img width="536" height="164" alt="image" src="https://github.com/user-attachments/assets/0fe0131c-5179-44e0-891e-b4fe257595c3" />


# Areas for Contribution

Model Improvements: Enhanced attention mechanisms, new aggregation strategies

Evaluation Metrics: Additional fake news detection metrics

Scalability: Optimization for larger datasets

Applications: Integration with real-world news platforms

Research: Novel approaches to misinformation detection
