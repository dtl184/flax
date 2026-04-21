from .gnn_dataset import create_super_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from torch.utils.tensorboard import SummaryWriter
import time
import copy
import matplotlib.pyplot as plt
import networkx as nx
import os



def train_model(model, dataloaders, criterion, optimizer, use_gpu, print_iter=10, 
                save_iter=50, save_folder='/tmp', num_epochs=1000, global_criterion=None,
                return_last_model_weights=True):
    """Optimize the model and save checkpoints with TensorBoard logging"""

    since = time.time()
    writer = SummaryWriter(log_dir="runs/exp1")

    best_seen_model_weights = None
    best_seen_running_validation_loss = np.inf

    if use_gpu:
        model = model.cuda()
        if criterion is not None:
            criterion = criterion.cuda()
        if global_criterion is not None:
            global_criterion = global_criterion.cuda()

    global_step = 0  # ✅ FIX: unique step counter

    for epoch in range(num_epochs):
        if isinstance(criterion, GRAPEMUSTPlanningLoss):
            if epoch < 100:
                criterion.set_entropy_weight(1e-3)
            elif epoch < 200:
                criterion.set_entropy_weight(1e-4)
            else:
                criterion.set_entropy_weight(0.0)
        if epoch % print_iter == 0:
            print(f'Epoch {epoch}/{num_epochs - 1}', flush=True)
            print('-' * 10, flush=True)

        phases = ['train', 'val']
        running_loss = {'train': 0.0, 'val': 0.0}

        for phase in phases:
            model.train(phase == 'train')

            for data in dataloaders[phase]:
                inputs = data['graph_input']
                targets = data['graph_target']

                if use_gpu:
                    for key in inputs:
                        inputs[key] = inputs[key].cuda()

                for key in targets:
                    if use_gpu:
                        targets[key] = targets[key].cuda()
                    if targets[key] is not None:
                        targets[key] = targets[key].detach()

                optimizer.zero_grad()
                outputs = model(inputs.copy())
                output = outputs[-1]

                loss = 0.0

                if criterion is not None:
                    if isinstance(criterion, GRAPEMUSTPlanningLoss):
                        node_loss = criterion(output['nodes'], targets['nodes'], targets.get('n_node'))
                    else:
                        node_loss = criterion(output['nodes'], targets['nodes'])
                    loss += node_loss
                else:
                    node_loss = torch.tensor(0.0)

                if global_criterion is not None:
                    if isinstance(global_criterion, nn.CrossEntropyLoss):
                        global_loss = global_criterion(
                            output['globals'],
                            targets['globals'].long().view(-1))
                    else:
                        global_loss = global_criterion(
                            output['globals'],
                            targets['globals'])
                    loss += global_loss
                else:
                    global_loss = torch.tensor(0.0)

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                loss_value = loss.item()

                # ✅ normalize loss (better plots)
                running_loss[phase] += loss_value / len(dataloaders[phase])

                # ✅ FIX: log with global_step (not epoch!)
                writer.add_scalar(f"{phase}/batch_loss", loss_value, global_step)

                if criterion is not None:
                    writer.add_scalar(f"{phase}/node_loss", node_loss.item(), global_step)

                if global_criterion is not None:
                    writer.add_scalar(f"{phase}/global_loss", global_loss.item(), global_step)

                global_step += 1

        if epoch % print_iter == 0:
            print("running_loss:", running_loss, flush=True)

        # ✅ log per-epoch loss (clean curves)
        writer.add_scalar("epoch/train_loss", running_loss['train'], epoch)
        if 'val' in phases:
            writer.add_scalar("epoch/val_loss", running_loss['val'], epoch)

        if epoch % save_iter == 0:
            save_path = os.path.join(save_folder, f"model{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint {save_path}")

            if 'val' in phases and running_loss['val'] < best_seen_running_validation_loss:
                best_seen_running_validation_loss = running_loss['val']
                best_seen_model_weights = model.state_dict()
                print(f"New best model (val loss={running_loss['val']}) at epoch {epoch}", flush=True)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s', flush=True)

    writer.flush()   # ✅ ensure data is written
    writer.close()

    if return_last_model_weights:
        return model.state_dict()

    return best_seen_model_weights



def get_model_predictions(model, dataloader, use_gpu=False):
    """
    """
    predictions = []
    model.train(False)
    model.eval()
    for data in dataloader:
        inputs = data['graph_input']
        
        if use_gpu:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()

        outputs = model(inputs.copy())
        graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=use_gpu)
        for graph in graphs:
            graph['nodes'] = graph['nodes'].numpy()
            graph['senders'] = graph['senders'].numpy()
            graph['receivers'] = graph['receivers'].numpy()
            graph['edges'] = graph['edges'].numpy()
            if graph['globals'] is not None:
                graph['globals'] = graph['globals'].numpy()
            graph['n_node'] = graph['n_node'].item()
            graph['n_edge'] = graph['n_edge'].item()
            predictions.append(graph)
    return predictions

def get_single_model_prediction(model, single_input, use_gpu=False):
    """
    """
    model.train(False)
    model.eval()
    inputs = create_super_graph([single_input])
    if use_gpu:
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
    outputs = model(inputs.copy())
    graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=use_gpu)
    assert len(graphs) == 1
    graph = graphs[0]
    graph['nodes'] = graph['nodes'].numpy()
    graph['senders'] = graph['senders'].numpy()
    graph['receivers'] = graph['receivers'].numpy()
    graph['edges'] = graph['edges'].numpy()
    if graph['globals'] is not None:
        graph['globals'] = graph['globals'].numpy()
    graph['n_node'] = graph['n_node'].item()
    graph['n_edge'] = graph['n_edge'].item()
    return graph

def get_multi_model_predictions(model, inputs, use_gpu=False):
    """
    """
    model.train(False)
    model.eval()
    inputs = create_super_graph(inputs)
    if use_gpu:
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
    outputs = model(inputs.copy())
    graphs = split_graphs(convert_to_data(outputs[-1]), use_gpu=use_gpu)
    out = []
    for graph in graphs:
        graph['nodes'] = graph['nodes'].numpy()
        graph['senders'] = graph['senders'].numpy()
        graph['receivers'] = graph['receivers'].numpy()
        graph['edges'] = graph['edges'].numpy()
        if graph['globals'] is not None:
            graph['globals'] = graph['globals'].numpy()
        graph['n_node'] = graph['n_node'].item()
        graph['n_edge'] = graph['n_edge'].item()
        out.append(graph)
    return out


class GRAPEMUSTPlanningLoss(nn.Module):
    """
    Stabilized MUS-only loss for planning guidance.

    Adds:
    - deterministic validation
    - separate train/eval sample counts
    - entropy schedule support
    - moving baseline
    """

    def __init__(
        self,
        n_samples=8,
        eval_n_samples=16,
        miss_penalty_weight=2.0,
        size_penalty_weight=1.0,
        entropy_weight=1e-3,
        baseline_momentum=0.9,
        prob_eps=1e-4,
        deterministic_eval=True,
        threshold=0.5,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.eval_n_samples = eval_n_samples
        self.miss_penalty_weight = miss_penalty_weight
        self.size_penalty_weight = size_penalty_weight
        self.entropy_weight = entropy_weight
        self.baseline_momentum = baseline_momentum
        self.prob_eps = prob_eps
        self.deterministic_eval = deterministic_eval
        self.threshold = threshold

        self.register_buffer("running_baseline", torch.tensor(0.0))
        self.register_buffer("baseline_initialized", torch.tensor(False))

    def set_entropy_weight(self, value):
        self.entropy_weight = float(value)

    def _per_graph_loss(self, y_g, t_g):
        y_g = y_g.float()
        t_g = t_g.float()

        n_total = float(y_g.shape[0])
        n_kept = y_g.sum()

        pos_mask = (t_g == 1)
        n_pos = pos_mask.sum()

        if n_pos.item() > 0:
            missed_pos = ((pos_mask) & (y_g == 0)).float().sum()
            miss_fraction = missed_pos / n_pos
        else:
            miss_fraction = torch.zeros((), device=y_g.device)

        size_fraction = n_kept / n_total

        loss = (
            self.miss_penalty_weight * miss_fraction
            + self.size_penalty_weight * (size_fraction ** 2)
        )
        return loss

    def _split_by_graph(self, tensor, n_node):
        sizes = [int(s) for s in n_node.view(-1).tolist()]
        return torch.split(tensor, sizes)

    def forward(self, logits, targets, n_node=None):
        probs = torch.sigmoid(logits).squeeze(-1)
        probs = probs.clamp(self.prob_eps, 1.0 - self.prob_eps)
        targets_flat = targets.squeeze(-1).float()

        # Deterministic validation
        if (not self.training) and self.deterministic_eval:
            y = (probs > self.threshold).float()

            if n_node is not None:
                y_list = self._split_by_graph(y, n_node)
                t_list = self._split_by_graph(targets_flat, n_node)
                g_losses = torch.stack([
                    self._per_graph_loss(y_g, t_g)
                    for y_g, t_g in zip(y_list, t_list)
                ])
                return g_losses.mean()
            else:
                return self._per_graph_loss(y, targets_flat)

        # Stochastic MUS training / optional stochastic eval
        dist = torch.distributions.Bernoulli(probs=probs)

        num_samples = self.n_samples if self.training else self.eval_n_samples

        total_pg_loss = torch.zeros((), device=logits.device)
        total_entropy = torch.zeros((), device=logits.device)
        total_raw_loss = torch.zeros((), device=logits.device)

        for _ in range(num_samples):
            y = dist.sample()
            log_probs = dist.log_prob(y)
            entropy = dist.entropy()

            if n_node is not None:
                y_list = self._split_by_graph(y, n_node)
                t_list = self._split_by_graph(targets_flat, n_node)
                lp_list = self._split_by_graph(log_probs, n_node)
                ent_list = self._split_by_graph(entropy, n_node)

                g_losses = torch.stack([
                    self._per_graph_loss(y_g, t_g)
                    for y_g, t_g in zip(y_list, t_list)
                ])
                g_log_probs = torch.stack([lp.sum() for lp in lp_list])
                g_entropies = torch.stack([ent.mean() for ent in ent_list])

                sample_raw_loss = g_losses.mean()
                sample_pg_loss = (g_losses.detach() * g_log_probs).mean()
                sample_entropy = g_entropies.mean()
            else:
                raw_loss = self._per_graph_loss(y, targets_flat)
                sample_raw_loss = raw_loss
                sample_pg_loss = raw_loss.detach() * log_probs.sum()
                sample_entropy = entropy.mean()

            total_raw_loss = total_raw_loss + sample_raw_loss
            total_pg_loss = total_pg_loss + sample_pg_loss
            total_entropy = total_entropy + sample_entropy

        avg_raw_loss = total_raw_loss / num_samples
        avg_pg_loss = total_pg_loss / num_samples
        avg_entropy = total_entropy / num_samples

        if self.training:
            with torch.no_grad():
                if not self.baseline_initialized.item():
                    self.running_baseline.copy_(avg_raw_loss.detach())
                    self.baseline_initialized.fill_(True)
                else:
                    self.running_baseline.mul_(self.baseline_momentum).add_(
                        (1.0 - self.baseline_momentum) * avg_raw_loss.detach()
                    )

        advantage = avg_raw_loss.detach() - self.running_baseline.detach()

        loss = advantage * avg_pg_loss - self.entropy_weight * avg_entropy
        loss = loss + 0.0 * avg_raw_loss
        return loss


# https://github.com/cimeister/pu-learning/blob/master/loss.py
# https://github.com/kiryor/nnPUlearning/issues/5
# http://proceedings.mlr.press/v37/plessis15.pdf
class PULoss(nn.Module):
    """wrapper of loss function for PU learning"""

    def __init__(self, prior, loss=(lambda x: torch.sigmoid(-x))):
        super(PULoss,self).__init__()
        if not 0 < prior < 1:
            raise NotImplementedError("The class prior should be in (0, 1)")
        self.prior = prior
        self.loss_func = loss
        self.positive = 1
        self.unlabeled = 0
        self.min_count = torch.tensor(1.)
    
    def forward(self, inp, target, test=False):
        assert(inp.shape == target.shape)        
        positive, unlabeled = target == self.positive, target == self.unlabeled
        positive, unlabeled = positive.type(torch.float), unlabeled.type(torch.float)
        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
            self.prior = self.prior.cuda()
        n_positive = torch.max(self.min_count, torch.sum(positive))
        n_unlabeled = torch.max(self.min_count, torch.sum(unlabeled))
        
        y_positive = self.loss_func(positive*inp)
        y_positive_inv = self.loss_func(-positive*inp)
        y_unlabeled = self.loss_func(-unlabeled*inp)

        positive_risk = self.prior * torch.sum(y_positive)/n_positive
        negative_risk = - self.prior * torch.sum(y_positive_inv)/n_positive + torch.sum(y_unlabeled)/n_unlabeled

        return positive_risk+negative_risk

def visualize_graphs(input_graph, output_graph, outfile, node_color_fn=None, edge_color_fn=None, **kwargs):
    """Draw input and output graphs side by side with networkx
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Input")
    axes[1].set_title("Output")

    if node_color_fn is None:
        node_color_fn = lambda *args : 'black'
    if edge_color_fn is None:
        edge_color_fn = lambda *args : 'black'

    for graph, ax in zip([input_graph, output_graph], axes.flat):
        G = nx.DiGraph()

        # Add nodes with colors
        for node in range(graph['n_node']):
            color = node_color_fn(graph, node, graph['nodes'][node])
            G.add_node(node, color=color)
        node_color_map = [G.nodes[u]['color'] for u in G.nodes()]

        # Add edges with colors
        for u, v, attrs in zip(graph['senders'], graph['receivers'], graph['edges']):
            color = edge_color_fn(graph, u, v, attrs)
            G.add_edge(u, v, color=color)
        edge_color_map = [G[u][v]['color'] for u,v in G.edges()]

        pos = nx.spring_layout(G, iterations=100, seed=0)
        nx.draw(G, pos, ax, node_color=node_color_map, edge_color=edge_color_map, **kwargs)

    plt.savefig(outfile)
    print("Wrote out to {}".format(outfile))

def _compute_stacked_offsets(sizes, repeats, numpy=False, use_gpu=True):
  """Computes offsets to add to indices of stacked np arrays.
  When a set of np arrays are stacked, the indices of those from the second on
  must be offset in order to be able to index into the stacked np array. This
  computes those offsets.
  Args:
    sizes: A 1D sequence of np arrays of the sizes per graph.
    repeats: A 1D sequence of np arrays of the number of repeats per graph.
  Returns:
    The index offset per graph.
  """
  idxs = np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)
  if numpy:
    return idxs
  else:
    if use_gpu:
        return torch.LongTensor(idxs).cuda()
    return torch.LongTensor(idxs)

def convert_to_data(graph):
    for key in graph.keys():
        if graph[key] is not None:
            graph[key] = graph[key].data
    return graph

def replace(graph, graph_dict):
  out_graph = graph.copy()
  for key in graph_dict.keys():
    out_graph[key] = graph_dict[key]
  return out_graph

def _unstack(array):
  """Similar to `tf.unstack`."""
  num_splits = int(array.shape[0])
  return [torch.squeeze(x, dim=0) for x in np.split(array, num_splits, axis=0)]

def split_graphs(graph, use_gpu=True):
  """Splits the stored data into a list of individual data dicts.
  Each list is a dictionary with fields NODES, EDGES, GLOBALS, RECEIVERS,
  SENDERS.
  Args:
    graph: A `graphs.GraphsTuple` instance containing numpy arrays.
  Returns:
    A list of the graph data dictionaries. The GLOBALS field is a tensor of
      rank at least 1, as the RECEIVERS and SENDERS field (which have integer
      values). The NODES and EDGES fields have rank at least 2.
  """
  offset = _compute_stacked_offsets(graph['n_node'].view(-1), graph['n_edge'].view(-1), numpy=False, use_gpu=use_gpu)
  nodes_splits = np.cumsum(graph['n_node'][:-1])
  edges_splits = np.cumsum(graph['n_edge'][:-1])
  graph_of_lists = collections.defaultdict(lambda: [])
  if graph['nodes'] is not None:
    graph_of_lists['nodes'] = np.split(graph['nodes'], nodes_splits)
  if graph['edges'] is not None:
    graph_of_lists['edges'] = np.split(graph['edges'], edges_splits)
  if graph['receivers'] is not None:
    graph_of_lists['receivers'] = np.split(graph['receivers'] - offset, edges_splits)
    graph_of_lists['senders'] = np.split(graph['senders'] - offset, edges_splits)
  if graph['globals'] is not None:
    graph_of_lists['globals'] = _unstack(graph['globals'])

  n_graphs = graph['n_node'].shape[0]
  # Make all fields the same length.
  for k in ['nodes','edges','globals']:
    graph_of_lists[k] += [None] * (n_graphs - len(graph_of_lists[k]))
  graph_of_lists['n_node'] = graph['n_node']
  graph_of_lists['n_edge'] = graph['n_edge']

  result = []
  for index in range(n_graphs):
    result.append({field: graph_of_lists[field][index] for field in ['nodes','edges','receivers','senders','globals','n_node','n_edge']})
  return result

def concat(input_graphs, dim, use_gpu=True):
  """
  In all cases, the NODES, EDGES and GLOBALS dimension are concatenated
  along `dim` (if a fields is `None`, the concatenation is just a `None`).
  If `dim` == 0, then the graphs are concatenated along the (underlying) batch
  dimension, i.e. the RECEIVERS, SENDERS, N_NODE and N_EDGE fields of the tuples
  are also concatenated together.
  If `dim` != 0, then there is an underlying asumption that the receivers,
  SENDERS, N_NODE and N_EDGE fields of the graphs in `values` should all match,
  but this is not checked by this op.
  The graphs in `input_graphs` should have the same set of keys for which the
  corresponding fields is not `None`.
  Args:
    input_graphs: A list of `graphs.GraphsTuple` objects containing `Tensor`s
      and satisfying the constraints outlined above.
    dim: An dim to concatenate on.
    name: (string, optional) A name for the operation.
  Returns: An op that returns the concatenated graphs.
  Raises:
    ValueError: If `values` is an empty list, or if the fields which are `None`
      in `input_graphs` are not the same for all the graphs.
  """
  if not input_graphs:
    raise ValueError("List argument `input_graphs` is empty")
  if len(input_graphs) == 1:
    return input_graphs[0]
  nodes = [gr['nodes'] for gr in input_graphs if gr['nodes'] is not None]
  edges = [gr['edges'] for gr in input_graphs if gr['edges'] is not None]
  globals_ = [gr['globals'] for gr in input_graphs if gr['globals'] is not None]

  nodes = torch.cat(nodes, dim) if nodes else None
  edges = torch.cat(edges, dim) if edges else None
  if globals_:
    globals_ = torch.cat(globals_, dim)
  else:
    globals_ = None

  output = replace(input_graphs[0],{'nodes':nodes, 'edges':edges, 'globals':globals_})
  if dim != 0:
    return output

  test = [torch.sum(gr['n_node']) for gr in input_graphs]
  n_node_per_tuple = torch.stack(
      [torch.sum(gr['n_node']) for gr in input_graphs])
  n_edge_per_tuple = torch.stack(
      [torch.sum(gr['n_edge']) for gr in input_graphs])
  offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple, use_gpu=use_gpu)
  n_node = torch.cat(
      [gr['n_node'] for gr in input_graphs], dim=0)
  n_edge = torch.cat(
      [gr['n_edge'] for gr in input_graphs], dim=0)
  receivers = [
      gr['receivers'] for gr in input_graphs if gr['receivers'] is not None
  ]
  receivers = receivers or None
  if receivers:
    receivers = torch.cat(receivers, dim) + offsets
  senders = [gr['senders'] for gr in input_graphs if gr['senders'] is not None]
  senders = senders or None
  if senders:
    senders = torch.cat(senders, dim) + offsets
  return replace(output, {'receivers':receivers, 'senders':senders, 'n_node':n_node, 'n_edge':n_edge})