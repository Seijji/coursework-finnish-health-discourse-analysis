"""
HTML report builder for discourse analysis results.
"""


def build_html(out_file, target, sents, vocab, w2v, similar, topics_data, 
               colls, keyness_df, args):
    """Build HTML report with all analysis results and visualizations."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Discourse Analysis: {target}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .word-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 15px 0;
        }}
        .word-badge {{
            background: #3498db;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Discourse Analysis: "{target}"</h1>

    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value">{len(sents):,}</div>
                <div class="stat-label">Sentences</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{vocab:,}</div>
                <div class="stat-label">Unique Lemmas</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(w2v.wv):,}</div>
                <div class="stat-label">W2V Vocabulary</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{args.topics}</div>
                <div class="stat-label">Topics</div>
            </div>
        </div>
        <h3>Methods</h3>
        <p><strong>Word2Vec:</strong> Skip-gram, dim=100, window={args.window}, min_count={args.min_count}</p>
        <p><strong>LDA:</strong> {args.topics} topics, symmetric priors, 10 passes</p>
        <p><strong>Collocation window:</strong> ±{args.window} words</p>
    </div>

    <div class="section">
        <h2>1. Semantic Network (Word2Vec)</h2>
        <p>t-SNE projection of word embeddings. Closer words share similar contexts.</p>
        <img src="word2vec_tsne.png" alt="Word2Vec t-SNE">
        
        <h3>Top 20 Similar Words</h3>
        <table>
            <tr><th>Rank</th><th>Word</th><th>Similarity</th></tr>
"""
    
    for i, (w, score) in enumerate(similar[:20], 1):
        html += f"            <tr><td>{i}</td><td><strong>{w}</strong></td><td>{score:.4f}</td></tr>\n"
    
    html += """        </table>
    </div>

    <div class="section">
        <h2>2. Topic Modeling (LDA)</h2>
        <img src="lda_topics.png" alt="LDA Topics">
        <img src="topic_distribution.png" alt="Distribution">
"""
    
    for t in topics_data:
        html += f"        <h3>Topic {t['id']}</h3>\n        <div class=\"word-list\">\n"
        for w in t['words']:
            html += f"            <span class=\"word-badge\">{w}</span>\n"
        html += "        </div>\n"
    
    html += """    </div>

    <div class="section">
        <h2>3. Collocations</h2>
        <p>Words frequently co-occurring with target term.</p>
        <img src="collocations.png" alt="Collocations">
        
        <table>
            <tr><th>Rank</th><th>Word</th><th>Frequency</th></tr>
"""
    
    for i, (w, freq) in enumerate(colls.most_common(20), 1):
        html += f"            <tr><td>{i}</td><td><strong>{w}</strong></td><td>{freq}</td></tr>\n"
    
    html += """        </table>
    </div>

    <div class="section">
        <h2>4. Keyness Analysis</h2>
        <p>Log-likelihood scores for distinctive words per topic.</p>
        <img src="keyness_heatmap.png" alt="Keyness">
        
        <h3>Top Distinctive Words</h3>
"""
    
    for tid in range(args.topics):
        top = keyness_df[keyness_df['topic'] == tid].nlargest(10, 'keyness')
        html += f"        <h4>Topic {tid}</h4>\n        <div class=\"word-list\">\n"
        for _, row in top.iterrows():
            html += f"            <span class=\"word-badge\">{row['word']} ({row['keyness']:.1f})</span>\n"
        html += "        </div>\n"
    
    html += """    </div>
</body>
</html>
"""
    
    with open(out_file, 'w') as f:
        f.write(html)
    print(f"HTML report: {out_file}")
