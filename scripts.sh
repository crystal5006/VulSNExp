#!/bin/bash
cd vulfcexp

# Train GNN-based vulnerability detectors
python main.py --do_train --do_test --gnn_model GCNConv --cuda_id 0
python main.py --do_train --do_test --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --cuda_id 0
python main.py --do_train --do_test --gnn_model GINConv --gin_eps 0.2 --cuda_id 0
python main.py --do_train --do_test --gnn_model GraphConv --gconv_aggr add --cuda_id 0


# Generating explanations using various explainers
# Generating explanations for the GCN-based vulnerability detector
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method gnnexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method pgexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method subgraphx --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method gnn_lrp --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method deeplift --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method gradcam --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method cfexplainer --KM 8 --cuda_id 1
python main.py --do_test --do_explain --gnn_model GCNConv --ipt_method vulsnexp --KM 8 --cuda_id 0



# Generating explanations for the GGNN-based vulnerability detector
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method gnnexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method pgexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method subgraphx --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method gnn_lrp --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method deeplift --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method gradcam --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method cfexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GatedGraphConv --num_gnn_layers 1 --num_ggnn_steps 2 --ggnn_aggr mean --ipt_method vulsnexp --KM 8 --cuda_id 1

# Generating explanations for the GIN-based vulnerability detector
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method gnnexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method pgexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method subgraphx --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method gnn_lrp --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method deeplift --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method gradcam --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method cfexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GINConv --gin_eps 0.2 --ipt_method vulsnexp --KM 8 --cuda_id 1

# Generating explanations for the GraphConv-based vulnerability detector
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method gnnexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method pgexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method subgraphx --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method gnn_lrp --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method deeplift --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method gradcam --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method cfexplainer --KM 8 --cuda_id 0
python main.py --do_test --do_explain --gnn_model GraphConv --gconv_aggr add --ipt_method vulsnexp --KM 8 --cuda_id 0

# Generating explanations under different KM settings
bash explain.sh