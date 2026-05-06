import os
import gc
import json
import random
import argparse
import warnings
import time

import numpy as np
from tqdm import tqdm
from sklearn.metrics import *
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.data import DataLoader
from torch_geometric.utils import *
import torch_scatter
from transformers import AdamW, get_linear_schedule_with_warmup
from torch_geometric.utils import remove_self_loops, add_self_loops, coalesce, add_remaining_self_loops, to_dense_adj

from models.vul_detector import Detector
from helpers import utils
from line_extract import get_dep_add_lines_bigvul
from graph_dataset import VulGraphDataset, collate
from models.gnnexplainer import XGNNExplainer
from models.cfexplainer import CFExplainer
from models.vulsnexp import VulSNExp
from models.pgexplainer import XPGExplainer, PGExplainer_edges
from models.subgraphx import SubgraphX
from models.gnn_lrp import GNN_LRP
from models.deeplift import DeepLIFT
from models.gradcam import GradCAM

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(y_true, y_pred):
    results = {
        'binary_precision': round(precision_score(y_true, y_pred, average='binary'), 4),
        'binary_recall': round(recall_score(y_true, y_pred, average='binary'), 4),
        'binary_f1': round(f1_score(y_true, y_pred, average='binary'), 4),
    }
    return results


def set_seed(seed=42, deterministic=True, warn_only=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Some ops (e.g., scatter_reduce on CUDA used by GAT) have no deterministic impls.
    # Allow configuring determinism to avoid runtime errors.
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    except TypeError:
        # Fallback for older PyTorch without warn_only kwarg
        torch.use_deterministic_algorithms(deterministic)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def train(args, train_dataloader, valid_dataloader, test_dataloader, model):
    args.max_steps = args.num_train_epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)

    checkpoint_last = os.path.join(args.model_checkpoint_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location=args.device))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location=args.device))

    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataloader)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  Total optimization steps = {}".format(args.max_steps))
    print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0

    model.zero_grad()
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch_data in enumerate(bar):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            # labels = global_max_pool(batch_data._VULN, batch).long()
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1
            model.train()
            probs = model(x, edge_index, batch)
            labels = F.one_hot(1 - labels, 2)
            loss = F.binary_cross_entropy(probs, labels.float())

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:

                    results = evaluate(args, valid_dataloader, model)
                    print(f"  Valid acc:{results['eval_acc']}")

                    # Save model checkpoint
                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        print("  " + "*" * 20)
                        print("  Best acc:{}".format(round(best_acc, 4)))
                        print("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        print("Saving model checkpoint to {}".format(output_dir))

                        test_result = evaluate(args, test_dataloader, model)
                        for key, value in test_result.items():
                            print("  {} = {}".format(key, round(value, 4)))
        bar.close()


def evaluate(args, eval_dataloader, model):
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataloader)))
    print("  Batch size = {}".format(args.batch_size))

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for step, batch_data in enumerate(eval_dataloader):
            batch_data.to(args.device)
            x, edge_index, batch = batch_data.x, batch_data.edge_index.long(), batch_data.batch
            edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
            edge_index = coalesce(edge_index)
            # labels = global_max_pool(batch_data._VULN, batch).long()
            labels = torch_scatter.segment_csr(batch_data._VULN, batch_data.ptr).long()
            labels[labels != 0] = 1
            probs = model(x, edge_index, batch)
            probs = F.one_hot(torch.argmax(probs, dim=-1), 2)[:, 0]
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, 0)
    all_labels = np.concatenate(all_labels, 0)
    eval_acc = np.mean(all_labels == all_probs)

    result = {
        "eval_acc": round(eval_acc, 4),
    }

    eval_results = calculate_metrics(all_labels, all_probs)
    result.update(eval_results)

    return result


def gen_exp_lines(edge_index, edge_weight, index, num_nodes, lines):
    temp = torch.zeros_like(edge_weight).to(edge_index.device)
    temp[index] = edge_weight[index]

    adj_mask = torch.sparse_coo_tensor(edge_index, temp, [num_nodes, num_nodes])
    adj_mask_binary = to_dense_adj(edge_index[:, temp != 0], max_num_nodes=num_nodes).squeeze(0)

    out_degree = torch.sum(adj_mask_binary, dim=1)
    out_degree[out_degree == 0] = 1e-8
    in_degree = torch.sum(adj_mask_binary, dim=0)
    in_degree[in_degree == 0] = 1e-8

    line_importance_init = torch.ones(num_nodes).unsqueeze(-1).to(edge_index.device)
    line_importance_out = torch.spmm(adj_mask, line_importance_init) / out_degree.unsqueeze(-1)
    line_importance_in = torch.spmm(adj_mask.T, line_importance_init) / in_degree.unsqueeze(-1)
    line_importance = line_importance_out + line_importance_in

    ret = sorted(
        list(
            zip(
                line_importance.squeeze(-1).cpu().numpy(),
                lines,
            )
        ),
        reverse=True,
    )

    filtered_ret = []
    for i in ret:
        if i[0] > 0:
            filtered_ret.append(int(i[1]))

    return filtered_ret



def eval_cff_exp(exp_saved_path, model, correct_lines, args):
    # 加载已保存的反事实解释数据
    graph_exp_list = torch.load(exp_saved_path, map_location=args.device)
    print("Number of explanations:", len(graph_exp_list))

    # 初始化评估指标
    accuracy = 0
    precisions = []
    recalls = []
    F1s = []
    pn = []
    ps = []

    for graph in graph_exp_list:
        graph.to(args.device)
        # 提取图的各项信息
        x, edge_index, edge_weight, pred, batch = graph.x, graph.edge_index.long(), graph.edge_weight, graph.pred, graph.batch
        label = global_max_pool(graph._VULN, batch).long()[0]  # 获取标签
        sampleid = graph._SAMPLE.max().int().item()  # 获取样本 ID
        exp_label_lines = correct_lines[int(sampleid)]
        exp_label_lines = list(exp_label_lines["removed"])  # 仅使用 "removed" 作为 Ground Truth

        # 选择权重最高的边
        if len(edge_weight) > args.KM:
            value, index = torch.topk(edge_weight, k=args.KM)
        else:
            index = torch.arange(edge_weight.shape[0])
        temp = torch.ones_like(edge_weight)
        temp[index] = 0
        cf_index = temp != 0

        # 获取解释的代码行号（需确保 gen_exp_lines 函数已定义）
        lines = graph._LINE.cpu().numpy()
        exp_lines = gen_exp_lines(edge_index, edge_weight, index, x.shape[0], lines)

        # 计算 accuracy
        for l in exp_lines:
            if l in exp_label_lines:
                accuracy += 1
                break

        # 计算 precision, recall, F1-score
        hit = sum(1 for l in exp_lines if l in exp_label_lines)
        precision = hit / len(exp_lines) if hit != 0 else 0
        recall = hit / len(exp_label_lines) if hit != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if hit != 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        F1s.append(f1)

        # 计算因果必要性（PN）和充分性（PS）
        fac_edge_index = edge_index[:, index]
        fac_edge_index, _ = add_self_loops(fac_edge_index, num_nodes=x.shape[0])
        fac_logits = model(x, fac_edge_index, batch)
        fac_pred = F.one_hot(torch.argmax(fac_logits, dim=-1), 2)[0][0]

        cf_edge_index = edge_index[:, cf_index]
        cf_edge_index, _ = add_self_loops(cf_edge_index, num_nodes=x.shape[0])
        cf_logits = model(x, cf_edge_index, batch)
        cf_pred = F.one_hot(torch.argmax(cf_logits, dim=-1), 2)[0][0]

        pn.append(int(cf_pred != pred))
        ps.append(int(fac_pred == pred))

    # 计算最终的评估指标
    accuracy = round(accuracy / len(graph_exp_list), 4)
    precision = round(np.mean(precisions), 4)
    recall = round(np.mean(recalls), 4)
    f1 = round(np.mean(F1s), 4)
    PN = round(sum(pn) / len(pn), 4)
    PS = round(sum(ps) / len(ps), 4)
    FNS = round(2 * PN * PS / (PN + PS), 4) if (PN + PS) != 0 else 0

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Probability of Necessity (PN):", PN)
    print("Probability of Sufficiency (PS):", PS)
    print("FNS:", FNS)

    # 记录实验结果
    saving_dir = "parameter_analysis" if args.hyper_para else "results"
    saving_path = os.path.join(utils.cache_dir(), saving_dir, f"{args.ipt_method}.res")

    KM_index_map = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6, 16: 7, 18: 8, 20: 9}
    if os.path.isfile(saving_path):
        result = json.load(open(saving_path, "r"))
    else:
        result = {}
    # Ensure current model key and metrics exist even if older file didn't include it
    metrics = ["Accuracy", "Precision", "Recall", "$F_1$", "PN", "PS", "FNS"]
    if args.gnn_model not in result:
        result[args.gnn_model] = {metric: [0.0] * 10 for metric in metrics}
    else:
        for metric in metrics:
            if metric not in result[args.gnn_model]:
                result[args.gnn_model][metric] = [0.0] * 10

    result[args.gnn_model]["Accuracy"][KM_index_map[args.KM]] = accuracy
    result[args.gnn_model]["Precision"][KM_index_map[args.KM]] = precision
    result[args.gnn_model]["Recall"][KM_index_map[args.KM]] = recall
    result[args.gnn_model]["$F_1$"][KM_index_map[args.KM]] = f1
    result[args.gnn_model]["PN"][KM_index_map[args.KM]] = PN
    result[args.gnn_model]["PS"][KM_index_map[args.KM]] = PS
    result[args.gnn_model]["FNS"][KM_index_map[args.KM]] = FNS

    json.dump(result, open(saving_path, "w"))

def gnnexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []

    explainer = XGNNExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05,
        coff_edge_size=0.0001, coff_edge_ent=0.001
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(f"Explaining sample {sampleid} ...")
        start_time = time.time()

        # 统一为：除 VulSNExp 外，均无条件添加自环
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        # edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, False, None,
        #                                                                              num_classes=args.num_classes, batch=batch)
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(
            x, edge_index, batch, False, None, num_classes=args.num_classes
        )
        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)
        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list

def cfexplainer_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []

    explainer = CFExplainer(
        model=model, explain_graph=True, epochs=800, lr=0.05, alpha=args.cfexp_alpha, L1_dist=args.cfexp_L1
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(f"Explaining sample {sampleid} ...")

        # 统一为：除 VulSNExp 外，均无条件添加自环
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        start_time = time.time()
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = explainer(x, edge_index, batch, False, None,
                                                                                     num_classes=args.num_classes)
        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)

        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"⏱Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        edge_weight = 1 - edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list

def vulsnexp_run(args, model, test_dataset, correct_lines):

    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []

    explainer = VulSNExp(
        model=model,
        epochs=args.fcexp_epochs,
        lr=args.fcexp_lr,
        gam=args.fcexp_gam,
        lam=args.fcexp_lam,
        alp=args.fcexp_alp,
        sparse_coeff=args.sparse_coeff
    )
    explainer.device = args.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()  # 假设每个图有 _SAMPLE 属性
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        x_input = graph.x

        # 保持训练与解释的一致性：GIN 不添加自环
        if args.gnn_model == "GINConv":
            edge_index_proc = edge_index
        else:
            edge_index_proc, _ = add_self_loops(edge_index, num_nodes=x_input.shape[0])

        prob = model(x_input, edge_index_proc, batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(f"Explaining sample {sampleid} ...")

        # 设置图的边索引以供 VulSNExp 使用
        if args.gnn_model == "GINConv":
            edge_index_full = edge_index
        else:
            edge_index_full, _ = add_remaining_self_loops(edge_index, num_nodes=x_input.shape[0])
        graph.edge_index = edge_index_full
        # 调用 VulSNExp 得到扰动后的边权
        # final_edge_weights = explainer(graph)
        start_time = time.time()
        final_edge_weights = explainer(graph, target_label=label.item())

        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)
        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        final_edge_weights = final_edge_weights.detach().cpu()
        edge_index_final, final_edge_weights = remove_self_loops(graph.edge_index.detach().cpu(), final_edge_weights)
        graph.edge_index = edge_index_final

        graph.__setitem__("edge_weight", torch.Tensor(final_edge_weights))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def pgexplainer_run(args, model, eval_model, train_dataset, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []
    input_dim = args.gnn_hidden_size * 2

    pgexplainer = XPGExplainer(model=model, in_channels=input_dim, device=args.device, explain_graph=True, epochs=100,
                               lr=0.005,
                               coff_size=0.01, coff_ent=5e-4, sample_bias=0.0, t0=5.0, t1=1.0)
    pgexplainer_saving_path = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/pgexplainer.bin")
    if os.path.isfile(pgexplainer_saving_path) and not args.ipt_update:
        print("Load saved PGExplainer model...")
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))
    else:
        pgexplainer.train_explanation_network(train_dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        pgexplainer.load_state_dict(torch.load(pgexplainer_saving_path, map_location=args.device))

    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=eval_model)
    pgexplainer_edges.device = pgexplainer.device

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(f"Explaining sample {sampleid} ...")
        start_time = time.time()

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = pgexplainer_edges(
            x, edge_index, batch=batch, num_classes=args.num_classes, sparsity=0.5
        )
        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)
        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def subgraphx_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/subgraphx")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    subgraphx = SubgraphX(model, args.num_classes, args.device, explain_graph=True,
                          verbose=False, c_puct=10.0, rollout=5, high2low=False, min_atoms=5, expand_atoms=14,
                          reward_method='gnn_score', subgraph_building_method='zero_filling',
                          save_dir=explanation_saving_dir)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        if edge_index.shape[1] > 2900:  # 跳过边数过大的图,根据实际情况调整阈值
            print(f"Skipping large graph {sampleid}")
            continue
        print(f"Explaining sample {sampleid} ...")
        start_time = time.time()

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        saved_MCTSInfo_list = None
        prediction = prob.argmax(-1).item()
        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            saved_MCTSInfo_list = torch.load(
                os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device
            )
            print(f"load example {sampleid}.")
        explain_result = subgraphx.explain(
            x, edge_index, batch=batch, label=prediction, node_idx=0, saved_MCTSInfo_list=saved_MCTSInfo_list
        )
        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)
        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        torch.save(explain_result, os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))
        node_weight = torch.zeros(x.shape[0])
        for item in explain_result:
            node_weight[item['coalition']] += item['P']
        node_weight = node_weight / len(explain_result)
        edge_index, _ = remove_self_loops(edge_index.detach().cpu())
        edge_weight = node_weight[edge_index[0]] + node_weight[edge_index[1]]
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gnn_lrp_run(args, model, test_dataset, correct_lines):
    # for name, parameter in model.named_parameters():
    #     print(name)

    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []

    explanation_saving_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}/gnn_lrp")
    if not os.path.exists(explanation_saving_dir):
        os.makedirs(explanation_saving_dir)
    gnnlrp_explainer = GNN_LRP(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        if edge_index.shape[1] > 2900:  # 跳过边数过大的图,根据实际情况调整阈值
            print(f"Skipping large graph {sampleid}")
            continue
        print(sampleid)
        start_time = time.time()
        print("\n", edge_index.shape[1])

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])

        if os.path.isfile(os.path.join(explanation_saving_dir, f'example_{sampleid}.pt')):
            edge_masks, self_loop_edge_index = torch.load(
                os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'), map_location=args.device)
            print(f"load example {sampleid}.")
        else:
            torch.cuda.empty_cache()  # 清理缓存，避免显存泄漏
            walks, edge_masks, related_preds, self_loop_edge_index = gnnlrp_explainer(
                x, edge_index, batch=batch, sparsity=0.5, num_classes=args.num_classes
            )
            elapsed_time = time.time() - start_time
            explanation_times.append(elapsed_time)
            avg_time_so_far = sum(explanation_times) / len(explanation_times)
            print(
                f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

            torch.save((edge_masks, self_loop_edge_index),
                       os.path.join(explanation_saving_dir, f'example_{sampleid}.pt'))

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph.detach().clone().cpu())
        visited_sampleids.add(sampleid)

        del graph
        gc.collect()

    return graph_exp_list


def deeplift_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []
    deep_lift = DeepLIFT(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(f"Explaining sample {sampleid} ...")
        start_time = time.time()

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = deep_lift(
            x, edge_index, batch=batch, sparsity=0.5, num_classes=args.num_classes
        )
        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)
        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)].sigmoid()
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def gradcam_run(args, model, test_dataset, correct_lines):
    graph_exp_list = []
    visited_sampleids = set()
    explanation_times = []
    gc_explainer = GradCAM(model, explain_graph=True)

    for graph in test_dataset:
        graph.to(args.device)
        x, edge_index, batch = graph.x, graph.edge_index.long(), graph.batch
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        if edge_index.shape[1] == 0:
            continue
        label = global_max_pool(graph._VULN, batch).long()[0]
        sampleid = graph._SAMPLE.max().int().item()
        if sampleid not in correct_lines:
            continue
        if sampleid in visited_sampleids:
            continue
        lines = graph._LINE.cpu().numpy()
        prob = model(x, add_self_loops(edge_index, num_nodes=x.shape[0])[0], batch)
        exp_prob_label = F.one_hot(torch.argmax(prob, dim=-1), 2)
        if label != 1 or prob[0][0] < prob[0][1]:
            continue
        print(f"Explaining sample {sampleid} ...")
        start_time = time.time()

        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
        edge_masks, hard_edge_masks, related_preds, self_loop_edge_index = gc_explainer(x, edge_index, batch=batch,
                                                                                        sparsity=0.5,
                                                                                        num_classes=args.num_classes)
        elapsed_time = time.time() - start_time
        explanation_times.append(elapsed_time)
        avg_time_so_far = sum(explanation_times) / len(explanation_times)
        print(
            f"  ⏱️  Time for this graph: {elapsed_time:.4f}s | Running {len(explanation_times)}graphs avg: {avg_time_so_far:.4f}s")

        edge_weight = edge_masks[torch.argmax(exp_prob_label, dim=-1)]
        edge_index, edge_weight = remove_self_loops(self_loop_edge_index.detach().cpu(), edge_weight.detach().cpu())
        graph.edge_index = edge_index

        graph.__setitem__("edge_weight", torch.Tensor(edge_weight))
        graph.__setitem__("pred", exp_prob_label[0][0])
        graph_exp_list.append(graph)
        visited_sampleids.add(sampleid)

    return graph_exp_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0,
                        help='which gpu to use if any')
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for initialization")

    parser.add_argument("--cv_split", type=str, default="default",
                        help="Cross-validation split mode: 'default' or 'crossval-5'")
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold index for cross-validation (0-4)')

    # GNN Model
    parser.add_argument("--model_checkpoint_dir", default="saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--gnn_model", default="GCNConv", type=str,
                        help="GNN core.")
    parser.add_argument("--gnn_hidden_size", default=256, type=int,
                        help="hidden size of gnn.")
    parser.add_argument("--gnn_feature_dim_size", default=768, type=int,
                        help="feature dim size of gnn.")
    parser.add_argument("--residual", action='store_true',
                        help="Whether to obtain residual representations.")
    parser.add_argument("--graph_pooling", default="mean", type=str,
                        help="The operator of graph pooling.")
    parser.add_argument("--num_gnn_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_ggnn_steps", default=3, type=int,
                        help="The sequence length for GGNN.")
    parser.add_argument("--ggnn_aggr", default="add", type=str,
                        help="The aggregation scheme to use for GGNN.")
    parser.add_argument("--gin_eps", default=0.3, type=float,
                        help="Eps value for GIN.")
    parser.add_argument("--gin_train_eps", action='store_true',
                        help="If set to True, eps will be a trainable parameter.")
    parser.add_argument("--gconv_aggr", default="mean", type=str,
                        help="The aggregation scheme to use.")
    # SAGE specific
    parser.add_argument("--sage_aggr", default="mean", type=str,
                        help="Aggregator for SAGEConv. Options: 'mean', 'max', 'add', 'lstm'.")
    # GAT specific
    parser.add_argument("--gat_heads", default=4, type=int,
                        help="Number of attention heads for GATConv.")
    parser.add_argument("--gat_concat", action='store_true',
                        help="Whether to concatenate multi-head outputs in GATConv. If set, hidden_size must be divisible by heads.")
    parser.add_argument("--gat_dropout", default=0.0, type=float,
                        help="Dropout rate for the attention coefficients in GATConv.")
    parser.add_argument("--dropout_rate", default=0.1, type=float,
                        help="Dropout rate.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")

    # Training
    parser.add_argument("--num_train_epochs", default=50, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size.")
    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_explain", action='store_true',
                        help="Whether to run explaining.")

    # Explainer
    parser.add_argument("--ipt_method", default="vulsnexp", type=str,
                        help="The save path of interpretations.")
    parser.add_argument("--ipt_update", action='store_true',
                        help="Whether to update interpretations.")
    parser.add_argument("--KM", default=8, type=int,
                        help="The size of explanation subgraph.")
    parser.add_argument("--cfexp_L1", action='store_true',
                        help="Whether to use L1 distance item.")
    parser.add_argument("--cfexp_alpha", default=0.1, type=float,
                        help="CFExplainer.")
    parser.add_argument("--fcexp_epochs", type=int, default=1000,
                        help="VulSNExp optimization epochs.")
    parser.add_argument("--fcexp_lr", type=float, default=0.1,
                        help="VulSNExp learning rate.")
    parser.add_argument("--fcexp_gam", type=float, default=0.9,
                        help="VulSNExp gam parameter.")
    parser.add_argument("--fcexp_lam", type=float, default=1,
                        help="VulSNExp lam parameter.")
    parser.add_argument("--fcexp_alp", type=float, default=0.6,
                        help="VulSNExp alp parameter.")
    parser.add_argument("--sparse_coeff", type=float, default=0.02)
    parser.add_argument("--componentwise_eval", action='store_true',
                        help="When set, compute PS/PN by removing isolated nodes and evaluating each connected component separately. Default off uses legacy whole-graph keep/remove Top-K.")
    parser.add_argument("--hyper_para", action='store_true',
                        help="Whether to tune the hyper-parameters.")
    parser.add_argument("--case_sample_ids", nargs='+',
                        help="Ids of samples to extract for case study.")
    parser.add_argument("--case_sample_id", type=int, default=None, help="Specific sample ID to explain")

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    args.device = device
    args.model_checkpoint_dir = str(utils.cache_dir() / f"{args.model_checkpoint_dir}" / args.gnn_model)
    # Some CUDA scatter_reduce ops used by certain GNNs (e.g., GAT, SAGE) lack deterministic impls.
    # To prevent runtime errors, relax determinism for these models.
    set_seed(args.seed, deterministic=True)

    args.start_epoch = 0
    args.start_step = 0

    model = Detector(args)
    model.to(args.device)

    # ===== 动态构建分区名称 =====
    train_partition = "train"
    val_partition = "val"
    test_partition = "test"

    # ===== 加载数据集 =====
    train_dataset = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        partition=train_partition,
        splits=args.cv_split
    )
    valid_dataset = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        partition=val_partition,
        splits=args.cv_split
    )
    test_dataset = VulGraphDataset(
        root=str(utils.processed_dir() / "vul_graph_dataset"),
        partition=test_partition,
        splits=args.cv_split
    )

    # train_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    print(train_dataset)

    # valid_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='val')
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                                  pin_memory=True)
    print(valid_dataset)

    # test_dataset = VulGraphDataset(root=str(utils.processed_dir() / "vul_graph_dataset"), partition='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(test_dataset)

    if args.do_train:
        train(args, train_dataloader, valid_dataloader, test_dataloader, model)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'
        model_checkpoint_dir = os.path.join(args.model_checkpoint_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
        model.to(args.device)
        test_result = evaluate(args, test_dataloader, model)

        print("***** Test results *****")
        for key in sorted(test_result.keys()):
            print("  {} = {}".format(key, str(round(test_result[key], 4))))

        if args.do_explain:
            # correct_lines = get_dep_add_lines_bigvul()
            correct_lines = get_dep_add_lines_bigvul(splits=args.cv_split)
            ipt_save_dir = str(utils.cache_dir() / f"explainer_cache" / f"{args.gnn_model}")
            if not os.path.exists(ipt_save_dir):
                os.makedirs(ipt_save_dir)
            # 根据解释方法选择保存路径
            if args.hyper_para:
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}_params.pt")
            else:
                ipt_save = os.path.join(ipt_save_dir, f"{args.ipt_method}.pt")
                print("Size of test dataset:", len(test_dataset))

            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if not os.path.exists(ipt_save) or args.ipt_update:
                graph_exp_list = []
                if args.ipt_method == "pgexplainer":
                    eval_model = Detector(args)
                    eval_model.load_state_dict(torch.load(model_checkpoint_dir, map_location=args.device))
                    eval_model.to(args.device)
                    graph_exp_list = pgexplainer_run(args, model, eval_model, train_dataset, test_dataset,
                                                     correct_lines)
                elif args.ipt_method == "subgraphx":
                    graph_exp_list = subgraphx_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnn_lrp":
                    graph_exp_list = gnn_lrp_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "deeplift":
                    graph_exp_list = deeplift_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gradcam":
                    graph_exp_list = gradcam_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "gnnexplainer":
                    graph_exp_list = gnnexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "cfexplainer":
                    graph_exp_list = cfexplainer_run(args, model, test_dataset, correct_lines)
                elif args.ipt_method == "vulsnexp":
                    graph_exp_list = vulsnexp_run(args, model, test_dataset, correct_lines)

                torch.save(graph_exp_list, ipt_save)

            eval_cff_exp(ipt_save, model, correct_lines, args)



if __name__ == "__main__":
    main()