import random
import torch
import torch.nn as nn

from utils import *
from model import Struc2Vec

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import logging

def main(args):

    if args.mode == 'train':
        
        if args.graph == 'karate_club_graph':
            nodes, edges = karate_club_graph()
        elif args.graph == 'barbell_graph':
            nodes, edges = barbell_graph()
        else:
            nodes, edges = generate_random_graph(args.number_nodes, args.percent_edges, args.seed)

        G = list2dict(nodes, edges)

        # --- Generates the walks ---

        walks = Struc2Vec(G, args.k, args.walk_length, args.number_walks, args.q, nodes, edges).run()
        
        # --- SkipGram ---

        model = Word2Vec(walks, vector_size=args.embedding_size, window=args.window, hs=1, sg=1, min_count=1)
        
        model.train(walks, total_examples=len(walks), epochs=args.epochs)

        # --- Plots --- 

        if args.plot_graph:
            plot_graph(nodes, edges, args.seed)

        if args.save_model:

            if not os.path.exists('weights'):
                os.makedirs('weights')
            
            model.save(f'weights/weights_{args.graph}_{args.epochs}.pth')

        if args.plot_pca:

            vectors = model.wv
            labels = vectors.index_to_key
            word_vectors = vectors.vectors

            plot_pca(word_vectors, args, labels)

    elif args.mode == 'inference':

        model = Word2Vec.load(f'weights/{args.model_load_path}.pth')

        most_similar_nodes = model.wv.most_similar(args.target, topn=args.most_similar)

        for node, similarity in most_similar_nodes:
            print(f'Node {node}: {similarity:.4f}')

if __name__ == '__main__':

    args = get_args()

    main(args)