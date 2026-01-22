#!/usr/bin/env python3
"""
Discourse Analysis - Word2Vec + LDA Topic Modeling

Analyzes lemmatized Finnish text using Word2Vec for semantic relationships
and LDA for topic modeling. Outputs visualizations and HTML report.
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

from report_builder import build_html


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Discourse analysis with Word2Vec and LDA')
    parser.add_argument('-i', '--input', default='lemmatized.txt', help='Input lemmatized text file')
    parser.add_argument('-t', '--target', default='rokotevastaisuus', help='Target word to analyze')
    parser.add_argument('-s', '--stopwords', default='stopwords.txt', help='Stop words file')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('--topics', type=int, default=5, help='Number of LDA topics')
    parser.add_argument('--min-count', type=int, default=10, help='Min word count for vocab')
    parser.add_argument('--window', type=int, default=5, help='Context window size')
    parser.add_argument('--top-words', type=int, default=200, help='Top N words for t-SNE viz')
    return parser.parse_args()


def load_data(input_file, stopwords_file):
    """Load sentences and stopwords from files."""
    # Load stopwords if available
    stops = set()
    if os.path.exists(stopwords_file):
        with open(stopwords_file, 'r') as f:
            stops = set(line.strip().lower() for line in f if line.strip())
        print(f"Loaded {len(stops)} stop words")
    
    # Load and filter sentences
    sents = []
    with open(input_file, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if stops:
                tokens = [t for t in tokens if t.lower() not in stops]
            if len(tokens) > 3:
                sents.append(tokens)
    
    return sents, stops


def train_models(sents, n_topics, window, min_count):
    """Train Word2Vec and LDA models on sentence corpus."""
    print(f"\nTraining Word2Vec...")
    w2v = Word2Vec(sents, vector_size=100, window=window, min_count=min_count,
                   workers=4, sg=1, epochs=10, seed=42)
    print(f"W2V vocab size: {len(w2v.wv)}")
    
    print(f"Training LDA with {n_topics} topics...")
    dictionary = Dictionary(sents)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(s) for s in sents]
    
    # Using symmetric priors to avoid dtype issues with 'auto'
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topics,
                   random_state=42, passes=10, alpha='symmetric', 
                   eta='symmetric', per_word_topics=True)
    
    return w2v, lda, dictionary, corpus


def extract_collocations(sents, target, window):
    """Find words that co-occur with target word within context window."""
    colls = Counter()
    for s in sents:
        if target in s:
            idx = s.index(target)
            start = max(0, idx - window)
            end = min(len(s), idx + window + 1)
            context = s[start:idx] + s[idx+1:end]
            colls.update(context)
    return colls


def calc_keyness(sents, doc_topics, n_topics):
    """Calculate log-likelihood keyness scores for distinctive words per topic."""
    topic_words = defaultdict(list)
    for i, s in enumerate(sents):
        if i < len(doc_topics) and doc_topics[i] >= 0:
            topic_words[doc_topics[i]].extend(s)
    
    results = []
    total = sum(len(words) for words in topic_words.values())
    all_words = Counter()
    for words in topic_words.values():
        all_words.update(words)
    
    for tid, words in topic_words.items():
        topic_counter = Counter(words)
        n_topic = len(words)
        
        for word in topic_counter.most_common(50):
            w = word[0]
            obs = topic_counter[w]
            exp = (n_topic / total) * all_words[w]
            
            if obs > 0 and exp > 0:
                ll = 2 * (obs * np.log(obs / exp))
                results.append({'topic': tid, 'word': w, 'keyness': ll, 'freq': obs})
    
    return pd.DataFrame(results)


def make_plots(w2v, lda, dictionary, sents, target, similar, colls, keyness_df, 
               doc_topics, args, out_dir):
    """Generate all visualization plots (t-SNE, topics, collocations, keyness, distribution)."""
    n_topics = args.topics
    
    # 1. t-SNE word embedding plot
    print("Creating t-SNE plot...")
    vocab_freq = [(w, w2v.wv.get_vecattr(w, "count")) for w in w2v.wv.index_to_key]
    top_words = [w for w, _ in sorted(vocab_freq, key=lambda x: x[1], reverse=True)[:args.top_words]]
    
    vecs = np.array([w2v.wv[w] for w in top_words])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(vecs)
    
    # Color by topic
    word_topics = {}
    for w in top_words:
        wid = dictionary.token2id.get(w)
        if wid:
            tprobs = lda.get_term_topics(wid, minimum_probability=0)
            word_topics[w] = max(tprobs, key=lambda x: x[1])[0] if tprobs else -1
        else:
            word_topics[w] = -1
    
    colors = [word_topics.get(w, -1) for w in top_words]
    
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Show only top 50 words for maximum clarity
    show_n = min(50, len(top_words))
    scatter = ax.scatter(coords[:show_n, 0], coords[:show_n, 1], 
                        c=colors[:show_n], cmap='tab10', 
                        alpha=0.4, s=120, edgecolors='black', linewidths=1)
    
    # Annotate target word very prominently
    if target in top_words[:show_n]:
        idx = top_words.index(target)
        ax.scatter(coords[idx, 0], coords[idx, 1], c='red', s=600, marker='*', 
                  edgecolors='black', linewidths=4, zorder=10)
        ax.annotate(target, (coords[idx, 0], coords[idx, 1]),
                   fontsize=22, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', 
                            alpha=0.95, edgecolor='red', linewidth=3),
                   zorder=11, ha='center')
    
    # Annotate top similar words clearly
    for i, (w, score) in enumerate(similar[:12], 1):
        if w in top_words[:show_n]:
            idx = top_words.index(w)
            ax.annotate(w, (coords[idx, 0], coords[idx, 1]),
                       fontsize=14, fontweight='normal', 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                                alpha=0.85, edgecolor='navy', linewidth=1.5),
                       zorder=9)
    
    # Label remaining words
    annotated = [target] + [w for w, _ in similar[:12]]
    for i, w in enumerate(top_words[:show_n]):
        if w not in annotated:
            ax.text(coords[i, 0], coords[i, 1], w, 
                   fontsize=11, alpha=0.7, ha='center', va='center',
                   fontweight='normal')
    
    ax.set_title(f'Word2Vec Semantic Space: "{target}" and Similar Words\n' + 
                f'Top {show_n} most frequent words (colored by topic)',
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16)
    ax.grid(True, alpha=0.4, linewidth=1.5, linestyle='-')
    plt.colorbar(scatter, ax=ax, label='Topic ID', shrink=0.8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/word2vec_tsne.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. LDA topics
    print("Creating topic plots...")
    topics_data = []
    for i in range(n_topics):
        words = lda.show_topic(i, topn=15)
        topics_data.append({
            'id': i,
            'words': [w for w, _ in words],
            'weights': [p for _, p in words]
        })
    
    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 3*n_topics))
    if n_topics == 1:
        axes = [axes]
    
    for i, t in enumerate(topics_data):
        axes[i].barh(t['words'][::-1], t['weights'][::-1], color='steelblue', alpha=0.7)
        axes[i].set_xlabel('Probability')
        axes[i].set_title(f'Topic {i}: {", ".join(t["words"][:5])}', fontweight='bold')
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{out_dir}/lda_topics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Collocations
    print("Creating collocation plot...")
    top20 = colls.most_common(20)
    words = [w for w, _ in top20]
    freqs = [f for _, f in top20]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(words[::-1], freqs[::-1], color='coral', alpha=0.7)
    ax.set_xlabel('Co-occurrence frequency')
    ax.set_title(f'Top 20 Collocations with "{target}"', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/collocations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Keyness heatmap
    print("Creating keyness heatmap...")
    top_key = keyness_df.groupby('topic').apply(lambda x: x.nlargest(15, 'keyness')).reset_index(drop=True)
    pivot = top_key.pivot_table(index='word', columns='topic', values='keyness', fill_value=0)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(pivot, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Log-likelihood'}, ax=ax)
    ax.set_title('Keyness: Distinctive Words per Topic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/keyness_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Topic distribution
    print("Creating topic distribution...")
    counts = Counter(doc_topics)
    topics = sorted([t for t in counts.keys() if t >= 0])
    vals = [counts[t] for t in topics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(topics, vals, color='mediumpurple', alpha=0.7)
    ax.set_xlabel('Topic ID')
    ax.set_ylabel('Number of sentences')
    ax.set_title('Topic Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/topic_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return topics_data


def main():
    """Main analysis pipeline."""
    args = parse_args()
    
    # Setup output directory
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}")
    
    # Load data
    print("Loading data...")
    sents, stops = load_data(args.input, args.stopwords)
    print(f"Loaded {len(sents)} sentences")
    
    vocab = Counter()
    for s in sents:
        vocab.update(s)
    print(f"Vocab: {len(vocab)} unique lemmas")
    print(f"Avg sentence length: {np.mean([len(s) for s in sents]):.1f}")
    
    # Train models
    w2v, lda, dictionary, corpus = train_models(sents, args.topics, args.window, args.min_count)
    w2v.save(f'{out_dir}/{args.target}_w2v.model')
    lda.save(f'{out_dir}/{args.target}_lda.model')
    
    # Get similar words
    similar = []
    if args.target in w2v.wv:
        similar = w2v.wv.most_similar(args.target, topn=30)
        print(f"\nTop similar to '{args.target}':")
        for w, score in similar[:10]:
            print(f"  {w}: {score:.3f}")
    else:
        print(f"Warning: '{args.target}' not in vocabulary")
    
    # Get document topics
    doc_topics = []
    for bow in corpus:
        td = lda.get_document_topics(bow)
        doc_topics.append(max(td, key=lambda x: x[1])[0] if td else -1)
    
    # Collocation analysis
    print("\nCollocation analysis...")
    colls = extract_collocations(sents, args.target, args.window)
    
    # Keyness analysis
    print("Keyness analysis...")
    keyness_df = calc_keyness(sents, doc_topics, args.topics)
    
    # Create plots
    print("\nGenerating plots...")
    topics_data = make_plots(w2v, lda, dictionary, sents, args.target, similar, 
                             colls, keyness_df, doc_topics, args, out_dir)
    
    # Build HTML report
    print("\nBuilding HTML report...")
    build_html(f'{out_dir}/analysis_report.html', args.target, sents, len(vocab), 
               w2v, similar, topics_data, colls, keyness_df, args)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"Files in '{out_dir}':")
    print("  - analysis_report.html (open in browser)")
    print("  - word2vec_tsne.png")
    print("  - lda_topics.png")
    print("  - collocations.png")
    print("  - keyness_heatmap.png")
    print("  - topic_distribution.png")
    print(f"  - {args.target}_w2v.model")
    print(f"  - {args.target}_lda.model")
    print("="*60)


if __name__ == '__main__':
    main()
