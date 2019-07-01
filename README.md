# Graph-Attention-Networks
PyTorch implementation of Graph Attention Networks.

# Graph Attention Networks
This is a PyTorch implementation of Graph Attention Networks (GAT) from the paper [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf).

## Usage

In the `src` directory, edit the `config.json` file to specify arguments and
flags. Then run `python main.py`.

## Limitations
* Currently, only supports the Cora dataset. However, for a new dataset it should be fairly straightforward to write a Dataset class similar to `datasets.Cora`.

## References
* [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf), Velickovic et al., ICLR 2018.
* [Collective Classification in Network Data](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2157), Sen et al., AI Magazine 2008.