# Embed Pandas Edgelist Graph with Pytorch Biggraph

This package takes a tutorial found at [here](https://github.com/facebookresearch/PyTorch-BigGraph/blob/master/torchbiggraph/examples/livejournal.py) and makes it easy to apply to a Graph that is represented in a Pandas Edgelist.  

## To Run

Run `embed_graph.py` from command line with either a tweet json.gz file or an edgelist.  For examples


```bash
#Example Command Line
python3 embed_graph.py --file mytweets.json.gz --epochs 8
```

Note that this will create/use subdirectories 'data' and 'model'.

When done, load embedding with command:

```python
with h5py.File("model/" + PROJECT + "/embeddings_user_id_0.v" + str(epochs) + ".h5", "r") as hf:
    embedding = hf["embeddings"][offset, :]
```

To cite Pytorch-BigGraph, use:

```
@article{lerer2019pytorch,
  title={Pytorch-biggraph: A large-scale graph embedding system},
  author={Lerer, Adam and Wu, Ledell and Shen, Jiajun and Lacroix, Timothee and Wehrstedt, Luca and Bose, Abhijit and Peysakhovich, Alex},
  journal={arXiv preprint arXiv:1903.12287},
  year={2019}
}
```

To cite `bot_match` methodology.
