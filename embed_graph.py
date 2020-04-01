#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:06:41 2020

adapted from https://github.com/facebookresearch/PyTorch-BigGraph/blob/master/torchbiggraph/examples/livejournal.py

@author: dbeskow
"""


import os
import random
import json
import h5py
import attr
import pandas as pd
import networkx as nx
import twitter_col
import argparse

from torchbiggraph.converters.import_from_tsv import convert_input_data
from torchbiggraph.config import parse_config
from torchbiggraph.train import train
from torchbiggraph.eval import do_eval

PROJECT = 'project'

#%%
# My Edgelist Preprocess
def preprocess_twitter(file_name):
    print('Creating Edgelist...')
    edge = twitter_col.get_edgelist_file(file_name, to_csv = False, kind = 'id_str')
    df = twitter_col.parse_only_text(file_name)
    G=nx.from_pandas_edgelist(edge, 'from', 'to', create_using = nx.DiGraph(),edge_attr = 'type')
    to_remove = set(list(G.nodes)) - set(df['id_str'].tolist())
    G.remove_nodes_from(to_remove)
    G = max(nx.weakly_connected_component_subgraphs(G), key=len)
    edge = nx.to_pandas_edgelist(G)
    lookup = pd.DataFrame({'source': list(G.nodes)})
    lookup['from'] = list(range(len(G.nodes)))
    edge = pd.merge(edge, lookup)
    lookup = pd.DataFrame({'target': list(G.nodes)})
    lookup['to'] = list(range(len(G.nodes)))
    edge = pd.merge(edge, lookup)
    outfile = 'data/' + PROJECT + '/edge.csv'
    print('Saving Edgelist to ', outfile)
    edge[['from','to']].to_csv(outfile, sep = ' ',index = False, header = False)
    
# My Edgelist Preprocess
def preprocess_edge_file(file_name):
    '''
    edge_list file must have a 'from' and 'to' column

    '''
    print('Creating Edgelist...')
    edge = pd.read_csv(file_name, dtype = str)
    G=nx.from_pandas_edgelist(edge, 'from', 'to', create_using = nx.DiGraph(),edge_attr = 'type')
    G = max(nx.weakly_connected_component_subgraphs(G), key=len)
    edge = nx.to_pandas_edgelist(G)
    lookup = pd.DataFrame({'source': list(G.nodes)})
    lookup['from'] = list(range(len(G.nodes)))
    edge = pd.merge(edge, lookup)
    lookup = pd.DataFrame({'target': list(G.nodes)})
    lookup['to'] = list(range(len(G.nodes)))
    edge = pd.merge(edge, lookup)
    outfile = 'data/' + PROJECT + '/edge.csv'
    print('Saving Edgelist to ', outfile)
    edge[['from','to']].to_csv(outfile,, sep = ' ',index = False, header = False)
    
#%%
# Create Config File
def MakeFile(file_name, epochs):

    lines = []
    lines.append("entities_base = 'data/" + PROJECT + "'")
    lines.append("def get_torchbiggraph_config():")
    lines.append("    config = dict(")
    lines.append("        # I/O data")
    lines.append("        entity_path=entities_base,")
    lines.append("        edge_paths=[],")
    lines.append("        checkpoint_path='model/" + PROJECT + "',")
    lines.append("        # Graph structure")
    lines.append("        entities={")
    lines.append("            'user_id': {'num_partitions': 1},")
    lines.append("        },")
    lines.append("        relations=[{")
    lines.append("            'name': 'follow',")
    lines.append("            'lhs': 'user_id',")
    lines.append("            'rhs': 'user_id',")
    lines.append("            'operator': 'none',")
    lines.append("        }],")
    lines.append("        # Scoring model")
    lines.append("        dimension=1024,")
    lines.append("        global_emb=False,")
    lines.append("        # Training")
    lines.append("        num_epochs=" + str(epochs) + ",")
    lines.append("        lr=0.001,")
    lines.append('        # Misc')
    lines.append("        hogwild_delay=2,")
    lines.append("    )")
    lines.append('    return config')
    
    total_lines =     "\n".join(lines)
    
    with open(file_name, 'w') as f:
        f.write(total_lines)
# ----------------------------------------------------------------------------------------------------------------------
# Helper functions, and constants


def convert_path(fname):
    basename, _ = os.path.splitext(fname)
    out_dir = basename + '_partitioned'
    return out_dir

def random_split_file(fpath):
    root = os.path.dirname(fpath)

    output_paths = [
        os.path.join(root, FILENAMES['train']),
        os.path.join(root, FILENAMES['test']),
    ]
    if all(os.path.exists(path) for path in output_paths):
        print("Found some files that indicate that the input data "
              "has already been shuffled and split, not doing it again.")
        print("These files are: %s" % ", ".join(output_paths))
        return

    print('Shuffling and splitting train/test file. This may take a while.')
    train_file = os.path.join(root, FILENAMES['train'])
    test_file = os.path.join(root, FILENAMES['test'])

    print('Reading data from file: ', fpath)
    with open(fpath, "rt") as in_tf:
        lines = in_tf.readlines()

    # The first few lines are comments
    lines = lines[4:]
    print('Shuffling data')
    random.shuffle(lines)
    split_len = int(len(lines) * TRAIN_FRACTION)

    print('Splitting to train and test files')
    with open(train_file, "wt") as out_tf_train:
        for line in lines[:split_len]:
            out_tf_train.write(line)

    with open(test_file, "wt") as out_tf_test:
        for line in lines[split_len:]:
            out_tf_test.write(line)
#%%
DATA_PATH = "data/" + PROJECT + "/edge.csv"
DATA_DIR = "data/" + PROJECT
CONFIG_PATH = "config.py"
FILENAMES = {
    'train': 'train.txt',
    'test': 'test.txt',
}
TRAIN_FRACTION = 0.75

edge_paths = [os.path.join(DATA_DIR, name) for name in FILENAMES.values()]
train_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['train']))]
eval_path = [convert_path(os.path.join(DATA_DIR, FILENAMES['test']))]

# ----------------------------------------------------------------------------------------------------------------------
#

def run_train_eval():
    random_split_file(DATA_PATH)

    convert_input_data(
        CONFIG_PATH,
        edge_paths,
        lhs_col=0,
        rhs_col=1,
        rel_col=None,
    )

    train_config = parse_config(CONFIG_PATH)

    train_config = attr.evolve(train_config, edge_paths=train_path)

    train(train_config)

    eval_config = attr.evolve(train_config, edge_paths=eval_path)

    do_eval(eval_config)

def output_embedding(epochs):
    with open(os.path.join(DATA_DIR, "dictionary.json"), "rt") as tf:
        dictionary = json.load(tf)

    user_id = "0"
    offset = dictionary["entities"]["user_id"].index(user_id)
    print("our offset for user_id ", user_id, " is: ", offset)

    with h5py.File("model/" + PROJECT + "/embeddings_user_id_0.v" + str(epochs) + ".h5", "r") as hf:
        embedding = hf["embeddings"][offset, :]

    print(f" our embedding looks like this: {embedding}")
    print(f"and has a size of: {embedding.shape}")


# ----------------------------------------------------------------------------------------------------------------------
# Main method

def main():
    parser=argparse.ArgumentParser(description="This embeds a graph with Facebook BigGraph")
    def file_choices(choices,fname):
        if not any(s in fname for s in choices):
           parser.error("file doesn't end with one of {}".format(choices))
        return fname
    parser.add_argument("--file",help="File Path for Twitter or Edgefile'" , 
                         type=lambda s:file_choices(["json","json.gz",'csv'],s))
    parser.add_argument('--epochs', '--Number of Epochs', type=int,
                    default=8, help='Number of Epochs')
    args=parser.parse_args()
    print(args)
    
       
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists('data/' + PROJECT):
        os.makedirs('data/' + PROJECT)
        
    if not os.path.exists('model'):
        os.makedirs('model')
        
    if not os.path.exists('model/' + PROJECT):
        os.makedirs('model/' + PROJECT)
        
    MakeFile('config.py', epochs = args.epochs)
    
    if 'json' in args.file:
        preprocess_twitter(args.file)
    elif 'csv' in args.file:
        preprocess_edge_file(args.file)
    else:
        exit('File must be Twitter JSON or CSV Edgelist')
 
    
    run_train_eval()

    output_embedding(args.epochs)

if __name__ == "__main__":
    main()