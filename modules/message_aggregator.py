from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        node_messages = messages.get(node_id)
        if node_messages:
            to_update_node_ids.append(node_id)
            unique_messages.append(node_messages[-1][0])
            unique_timestamps.append(node_messages[-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      node_messages = messages.get(node_id)
      if node_messages:
        n_messages += len(node_messages)
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in node_messages]), dim=0))
        unique_timestamps.append(node_messages[-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class TypeBasedAggregator(MessageAggregator):
  def __init__(self, device, message_dim, n_types=2):
    super(TypeBasedAggregator, self).__init__(device)
    self.k = n_types
    self.message_dim = message_dim
    self.scale = 1.0 / math.sqrt(message_dim)

    # Orthogonal initialization using Gram-Schmidt
    V_init = torch.randn(message_dim, n_types)
    V_init[:, 0] = F.normalize(V_init[:, 0], dim=0)
    for i in range(1, n_types):
      for j in range(i):
        V_init[:, i] -= (V_init[:, i] @ V_init[:, j]) * V_init[:, j]
      V_init[:, i] = F.normalize(V_init[:, i], dim=0)

    self.V = nn.Parameter(V_init)

  def aggregate(self, node_ids, messages):
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    to_update_node_ids = []

    for node_id in unique_node_ids:
      node_messages = messages.get(node_id)
      if node_messages:
        to_update_node_ids.append(node_id)

        if len(node_messages) == 1:
          aggregated_msg = node_messages[0][0]
        else:
          # Stack messages: [N, D]
          msg_stack = torch.stack([m[0] for m in node_messages])

          # Point-product attention: [N, D] @ [D, k] -> [N, k]
          attn_scores = torch.matmul(msg_stack, self.V) * self.scale
          attn_weights = F.softmax(attn_scores, dim=0)  # [N, k]

          # Type-space aggregation: [k, N] @ [N, D] -> [k, D]
          type_repr = torch.matmul(attn_weights.t(), msg_stack)

          # Max pooling over types: [k, D] -> [D]
          aggregated_msg, _ = torch.max(type_repr, dim=0)

        unique_messages.append(aggregated_msg)
        unique_timestamps.append(node_messages[-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device, message_dim=None, n_types=2):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  elif aggregator_type == "type_based":
    if message_dim is None:
      raise ValueError("message_dim must be provided for type_based aggregator")
    return TypeBasedAggregator(device=device, message_dim=message_dim, n_types=n_types)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
