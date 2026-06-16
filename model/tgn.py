import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
               std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_destination_embedding_in_message=False,
               use_source_embedding_in_message=False,
               dyrep=False, n_types=2, use_dynamic_topology_features=True,
               use_pair_prior=True):
    super(TGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_destination_embedding_in_message = use_destination_embedding_in_message
    self.use_source_embedding_in_message = use_source_embedding_in_message
    self.dyrep = dyrep
    self.use_dynamic_topology_features = use_dynamic_topology_features
    self.dynamic_topology_feature_dim = 7 if self.use_dynamic_topology_features else 0
    self.use_pair_prior = use_pair_prior and self.use_dynamic_topology_features

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None
    self.reset_dynamic_topology_state()

    self.mean_time_shift_src = mean_time_shift_src
    self.std_time_shift_src = std_time_shift_src
    self.mean_time_shift_dst = mean_time_shift_dst
    self.std_time_shift_dst = std_time_shift_dst

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension + self.dynamic_topology_feature_dim
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device,
                                                       message_dim=message_dimension,
                                                       n_types=n_types)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MLP to compute probability on an edge given two node embeddings
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features,
                                     self.n_node_features,
                                     1)
    if self.use_pair_prior:
      # Lightweight calibration head on historical pair features.
      self.pair_prior_layer = torch.nn.Linear(3, 1)
      torch.nn.init.zeros_(self.pair_prior_layer.weight)
      torch.nn.init.zeros_(self.pair_prior_layer.bias)

  def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                  edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for sources, destinations, and negatively sampled destinations.

    source_nodes [batch_size]: source ids.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Temporal embeddings for sources, destinations and negatives
    """

    n_samples = len(source_nodes)
    nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
    positives = np.concatenate([source_nodes, destination_nodes])
    timestamps = np.concatenate([edge_times, edge_times, edge_times])

    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        nodes_with_messages = self.memory.get_nodes_with_messages()
        memory, last_update = self.get_updated_memory(nodes_with_messages,
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      ### Compute differences between the time the memory of a node was last updated,
      ### and the time for which we want to compute the embedding of a node
      source_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        source_nodes].long()
      source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
      destination_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        destination_nodes].long()
      destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
      negative_time_diffs = torch.LongTensor(edge_times).to(self.device) - last_update[
        negative_nodes].long()
      negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

      time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs],
                             dim=0)

    # Compute the embeddings using the embedding module
    node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                             source_nodes=nodes,
                                                             timestamps=timestamps,
                                                             n_layers=self.n_layers,
                                                             n_neighbors=n_neighbors,
                                                             time_diffs=time_diffs)

    source_node_embedding = node_embedding[:n_samples]
    destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
    negative_node_embedding = node_embedding[2 * n_samples:]

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory only for sources and destinations (since now we have
        # new messages for them)
        self.update_memory(positives, self.memory.messages)

        # assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
        #   "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        self.memory.clear_messages(positives)

      unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                    source_node_embedding,
                                                                    destination_nodes,
                                                                    destination_node_embedding,
                                                                    edge_times, edge_idxs)
      unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                              destination_node_embedding,
                                                                              source_nodes,
                                                                              source_node_embedding,
                                                                              edge_times, edge_idxs)
      self.update_dynamic_topology_state(source_nodes, destination_nodes, edge_times)
      if self.memory_update_at_start:
        self.memory.store_raw_messages(unique_sources, source_id_to_messages)
        self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
      else:
        self.update_memory(unique_sources, source_id_to_messages)
        self.update_memory(unique_destinations, destination_id_to_messages)

      if self.dyrep:
        source_node_embedding = memory[source_nodes]
        destination_node_embedding = memory[destination_nodes]
        negative_node_embedding = memory[negative_nodes]

    return source_node_embedding, destination_node_embedding, negative_node_embedding

  def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                 edge_idxs, n_neighbors=20):
    """
    Compute probabilities for edges between sources and destination and between sources and
    negatives by first computing temporal embeddings using the TGN encoder and then feeding them
    into the MLP decoder.
    :param destination_nodes [batch_size]: destination ids
    :param negative_nodes [batch_size]: ids of negative sampled destination
    :param edge_times [batch_size]: timestamp of interaction
    :param edge_idxs [batch_size]: index of interaction
    :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
    layer
    :return: Probabilities for both the positive and negative edges
    """
    n_samples = len(source_nodes)
    source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
      source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

    score = self.affinity_score(torch.cat([source_node_embedding, source_node_embedding], dim=0),
                                torch.cat([destination_node_embedding,
                                           negative_node_embedding])).squeeze(dim=1)
    if self.use_pair_prior:
      pos_pair_prior = self.get_pair_prior_features(source_nodes, destination_nodes, edge_times)
      neg_pair_prior = self.get_pair_prior_features(source_nodes, negative_nodes, edge_times)
      pair_prior = self.pair_prior_layer(torch.cat([pos_pair_prior, neg_pair_prior], dim=0)).squeeze(dim=1)
      score = score + pair_prior

    pos_score = score[:n_samples]
    neg_score = score[n_samples:]

    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                       destination_node_embedding, edge_times, edge_idxs):
    edge_times = torch.from_numpy(edge_times).float().to(self.device)
    edge_features = self.edge_raw_features[edge_idxs]

    source_memory = self.memory.get_memory(source_nodes) if not \
      self.use_source_embedding_in_message else source_node_embedding
    destination_memory = self.memory.get_memory(destination_nodes) if \
      not self.use_destination_embedding_in_message else destination_node_embedding

    source_time_delta = edge_times - self.memory.last_update[source_nodes]
    source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
      source_nodes), -1)

    source_message = torch.cat([source_memory, destination_memory, edge_features,
                                source_time_delta_encoding],
                               dim=1)
    if self.use_dynamic_topology_features:
      dynamic_topology_features = self.get_dynamic_topology_features(source_nodes, destination_nodes, edge_times)
      source_message = torch.cat([source_message, dynamic_topology_features], dim=1)
    messages = defaultdict(list)
    unique_sources = np.unique(source_nodes)

    for i in range(len(source_nodes)):
      messages[source_nodes[i]].append((source_message[i], edge_times[i]))

    return unique_sources, messages

  def set_neighbor_finder(self, neighbor_finder):
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder

  def reset_dynamic_topology_state(self):
    self.node_temporal_degree = torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device)
    self.node_source_interactions = torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device)
    self.node_destination_interactions = torch.zeros(self.n_nodes, dtype=torch.float32, device=self.device)
    self.pair_interaction_count = {}
    self.pair_last_timestamp = {}

  def get_dynamic_topology_features(self, source_nodes, destination_nodes, edge_times):
    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    destination_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)

    source_temporal_degree = torch.log1p(self.node_temporal_degree[source_nodes_torch]).unsqueeze(1)
    destination_temporal_degree = torch.log1p(self.node_temporal_degree[destination_nodes_torch]).unsqueeze(1)
    source_interaction_count = torch.log1p(self.node_source_interactions[source_nodes_torch]).unsqueeze(1)
    destination_interaction_count = torch.log1p(self.node_destination_interactions[destination_nodes_torch]).unsqueeze(1)
    pair_prior = self.get_pair_prior_features(source_nodes, destination_nodes, edge_times)

    return torch.cat([source_temporal_degree,
                      destination_temporal_degree,
                      source_interaction_count,
                      destination_interaction_count,
                      pair_prior], dim=1)

  def get_pair_prior_features(self, source_nodes, destination_nodes, edge_times):
    pair_count = []
    pair_recency = []
    pair_seen = []
    for source_node, destination_node, edge_time in zip(source_nodes, destination_nodes, edge_times):
      pair_key = self._build_pair_key(int(source_node), int(destination_node))
      historical_count = self.pair_interaction_count.get(pair_key, 0)
      if historical_count > 0:
        delta_t = max(float(edge_time) - self.pair_last_timestamp[pair_key], 0.0)
        recency_score = 1.0 / (1.0 + np.log1p(delta_t))
        seen = 1.0
      else:
        recency_score = 0.0
        seen = 0.0
      pair_count.append(np.log1p(float(historical_count)))
      pair_recency.append(recency_score)
      pair_seen.append(seen)

    pair_count = torch.tensor(pair_count, dtype=torch.float32, device=self.device).unsqueeze(1)
    pair_recency = torch.tensor(pair_recency, dtype=torch.float32, device=self.device).unsqueeze(1)
    pair_seen = torch.tensor(pair_seen, dtype=torch.float32, device=self.device).unsqueeze(1)
    return torch.cat([pair_count, pair_recency, pair_seen], dim=1)

  def update_dynamic_topology_state(self, source_nodes, destination_nodes, edge_times):
    if not self.use_dynamic_topology_features:
      return

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    destination_nodes_torch = torch.from_numpy(destination_nodes).long().to(self.device)

    source_ones = torch.ones(len(source_nodes_torch), dtype=torch.float32, device=self.device)
    destination_ones = torch.ones(len(destination_nodes_torch), dtype=torch.float32, device=self.device)

    self.node_temporal_degree.index_add_(0, source_nodes_torch, source_ones)
    self.node_temporal_degree.index_add_(0, destination_nodes_torch, destination_ones)
    self.node_source_interactions.index_add_(0, source_nodes_torch, source_ones)
    self.node_destination_interactions.index_add_(0, destination_nodes_torch, destination_ones)
    for source_node, destination_node, edge_time in zip(source_nodes, destination_nodes, edge_times):
      pair_key = self._build_pair_key(int(source_node), int(destination_node))
      self.pair_interaction_count[pair_key] = self.pair_interaction_count.get(pair_key, 0) + 1
      self.pair_last_timestamp[pair_key] = float(edge_time)

  def backup_dynamic_topology_state(self):
    if not self.use_dynamic_topology_features:
      return None
    return (self.node_temporal_degree.clone(),
            self.node_source_interactions.clone(),
            self.node_destination_interactions.clone(),
            dict(self.pair_interaction_count),
            dict(self.pair_last_timestamp))

  def restore_dynamic_topology_state(self, dynamic_backup):
    if (not self.use_dynamic_topology_features) or dynamic_backup is None:
      return
    self.node_temporal_degree = dynamic_backup[0].clone()
    self.node_source_interactions = dynamic_backup[1].clone()
    self.node_destination_interactions = dynamic_backup[2].clone()
    self.pair_interaction_count = dict(dynamic_backup[3])
    self.pair_last_timestamp = dict(dynamic_backup[4])

  @staticmethod
  def _build_pair_key(source_node, destination_node):
    return (source_node, destination_node) if source_node <= destination_node else (destination_node, source_node)

  def backup_memory_state(self):
    return self.memory.backup_memory(), self.backup_dynamic_topology_state()

  def restore_memory_state(self, state_backup):
    self.memory.restore_memory(state_backup[0])
    self.restore_dynamic_topology_state(state_backup[1])
