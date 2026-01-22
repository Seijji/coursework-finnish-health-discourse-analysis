#!/usr/bin/env python3
"""
Lemmatization Script for Finnish Text

Lemmatizes Finnish concordance data using Stanza NLP library.
Reduces words to their base/dictionary forms.
"""

import argparse
import sys
import stanza


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Lemmatize Finnish text with Stanza')
    parser.add_argument('-i', '--input', default='concordance.txt', help='Input concordance file')
    parser.add_argument('-o', '--output', default='lemmatized.txt', help='Output file')
    parser.add_argument('-d', '--dedup', action='store_true', help='Remove duplicate lines')
    parser.add_argument('--download', action='store_true', help='Download Finnish model first')
    return parser.parse_args()


def load_data(input_file, dedup=False):
    """Load concordance lines from file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(lines)} lines")
        
        if dedup:
            orig = len(lines)
            lines = list(dict.fromkeys(lines))  # preserve order
            print(f"Removed {orig - len(lines)} duplicates, {len(lines)} unique")
        
        return lines
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found", file=sys.stderr)
        sys.exit(1)


def lemmatize(texts, nlp):
    """Lemmatize texts using Stanza pipeline."""
    results = []
    
    for i, text in enumerate(texts, 1):
        doc = nlp(text)
        lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
        results.append(' '.join(lemmas))
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(texts)}...")
    
    return results


def save_results(output_file, texts):
    """Save lemmatized texts to file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"Saved {len(texts)} lines to {output_file}")
    except IOError as e:
        print(f"Error writing file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main lemmatization pipeline."""
    args = parse_args()
    
    if args.download:
        print("Downloading Finnish model...")
        stanza.download('fi')
    
    print("Initializing Stanza pipeline...")
    nlp = stanza.Pipeline('fi', verbose=False)
    
    texts = load_data(args.input, dedup=args.dedup)
    
    print("Lemmatizing...")
    lemmatized = lemmatize(texts, nlp)
    
    save_results(args.output, lemmatized)
    print("Done!")


if __name__ == '__main__':
    main()
