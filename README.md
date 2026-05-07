# VulSNExp

**Framework**

![Framework diagram](./framework.pdf)

VulSNExp introduces a structural optimization framework for vulnerability detection, which learns a differentiable edge mask to enforce factual sufficiency and counterfactual necessity, thereby extracting a compact subgraph that preserves the model’s prediction when retained and alters it when removed.See the framework figure above.

**Quick Guide**
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Train Detector](#train-detector)
- [Run Explainers](#run-explainers)

# Environment Setup

Use CUDA 11.7 and cuDNN 8.8.1, then create a Conda environment:
```shell
conda create -n vulsnexp python=3.9
conda activate vulsnexp
```

Install the main dependencies:
```shell
pip install https://download.pytorch.org/whl/cu117/torch-2.0.0%2Bcu117-cp39-cp39-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu117/torchvision-0.15.1%2Bcu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/pyg_lib-0.2.0%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_cluster-1.6.1%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_scatter-2.1.1%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_sparse-0.6.17%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu117/torch_spline_conv-1.2.2%2Bpt20cu117-cp39-cp39-linux_x86_64.whl
pip install torch_geometric
```

Other required Python packages are as follows:
```shell
pip install numpy==1.24.3
pip install pandas==2.0.1
pip install scikit-learn==1.2.2
pip install tensorboard==2.13.0
pip install transformers==4.29.1
pip install tqdm==4.65.0
pip install scipy==1.10.1
pip install graphviz==0.20.1
pip install unidiff==0.7.5
pip install dive-into-graphs==1.1.0
pip install captum==0.2.0
pip install matplotlib==3.7.1
pip install rdkit
```

Install Joern 1.1.260 for graph generation:
```shell
wget https://github.com/joernio/joern/releases/download/v1.1.260/joern-install.sh
chmod +x ./joern-install.sh
printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n'  | sudo ./joern-install.sh --interactive
```

# Data Preparation

1.Download `MSR_data_cleaned.csv`, then place it here:
```shell
unzip MSR_data_cleaned.zip
rm MSR_data_cleaned.zip
mv MSR_data_cleaned.csv vulsnexp/storage/external
```

2.Set the project path:
```shell
cd vulsnexp
export SINGSTORAGE=$(pwd)
```

3.Preprocess the dataset:

```shell
python data_pre.py
```

This creates the cached data under `storage/cache`.

4.Generate code graphs:
```shell
python code_graph_gen.py 1
python code_graph_gen.py 2
python code_graph_gen.py 3
python code_graph_gen.py 4
python code_graph_gen.py 5
```

5.Build train/val/test graph datasets:
```shell
python graph_dataset.py train
python graph_dataset.py val
python graph_dataset.py test
```
The processed files are saved under `storage/cache/vul_graph_feat` and `storage/processed/vul_graph_dataset`.

6.Extract changed lines if needed:
```shell
python line_extract.py
```
The output is saved to `storage/processed/bigvul/eval/statement_labels.pkl`.


# Train Detector

Train the four GNN models:
```shell
python main.py --do_train --do_test --gnn_model GCNConv --cuda_id 0
python main.py --do_train --do_test --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --cuda_id 0
python main.py --do_train --do_test --gnn_model GINConv --gin_eps 0.2 --cuda_id 0
python main.py --do_train --do_test --gnn_model GraphConv --gconv_aggr add --cuda_id 0
```

Checkpoints are saved in `storage/cache/saved_models`.

# Run Explainers

Run the explainers on a trained detector:
```shell
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method specific_explainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method specific_explainer --KM 8 --cuda_id 0
```

See `scripts.sh` for the full experiment commands.

# Notes

Experiments were run on  a Linux-based server equipped with two 2.1GHz Intel Xeon Silver-4310 CPU, 128GB of RAM, and dual NVIDIA RTX 3090 GPUs.
