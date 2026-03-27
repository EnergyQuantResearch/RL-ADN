from __future__ import annotations

import math
import os
from typing import Tuple

import torch
from torch import Tensor

from rl_adn.algorithms.training_config import Config


class ReplayBuffer:
    """Replay buffer for off-policy algorithms."""

    def __init__(
        self,
        max_size: int,
        state_dim: int,
        action_dim: int,
        gpu_id: int = 0,
        num_seqs: int = 1,
        if_use_per: bool = False,
        args: Config | None = None,
    ) -> None:
        self.args = args or Config()
        self.p = 0
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id >= 0) else "cpu")

        self.states = torch.empty((max_size, num_seqs, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, num_seqs, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, num_seqs), dtype=torch.float32, device=self.device)

        self.if_use_per = if_use_per
        if if_use_per:
            self.sum_trees = [SumTree(buf_len=max_size) for _ in range(num_seqs)]
            self.per_alpha = getattr(self.args, "per_alpha", 0.6)
            self.per_beta = getattr(self.args, "per_beta", 0.4)
        else:
            self.sum_trees = None
            self.per_alpha = None
            self.per_beta = None

    def update(self, items: Tuple[Tensor, ...]) -> None:
        self.add_item = items
        states, actions, rewards, undones = items
        assert states.shape[1:] == (self.args.num_envs, self.args.state_dim)
        assert actions.shape[1:] == (self.args.num_envs, self.args.action_dim)
        assert rewards.shape[1:] == (self.args.num_envs,)
        assert undones.shape[1:] == (self.args.num_envs,)
        self.add_size = rewards.shape[0]

        new_pointer = self.p + self.add_size
        if new_pointer > self.max_size:
            self.if_full = True
            split_index = self.max_size - self.p
            new_pointer -= self.max_size

            self.states[self.p : self.max_size], self.states[0:new_pointer] = states[:split_index], states[-new_pointer:]
            self.actions[self.p : self.max_size], self.actions[0:new_pointer] = actions[:split_index], actions[-new_pointer:]
            self.rewards[self.p : self.max_size], self.rewards[0:new_pointer] = rewards[:split_index], rewards[-new_pointer:]
            self.undones[self.p : self.max_size], self.undones[0:new_pointer] = undones[:split_index], undones[-new_pointer:]
        else:
            self.states[self.p : new_pointer] = states
            self.actions[self.p : new_pointer] = actions
            self.rewards[self.p : new_pointer] = rewards
            self.undones[self.p : new_pointer] = undones

        if self.if_use_per and self.sum_trees is not None:
            data_ids = torch.arange(self.p, new_pointer, dtype=torch.long, device=self.device)
            if new_pointer > self.max_size:
                data_ids = torch.fmod(data_ids, self.max_size)
            for sum_tree in self.sum_trees:
                sum_tree.update_ids(data_ids=data_ids.cpu(), prob=10.0)

        self.p = new_pointer
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sample_len = self.cur_size - 1
        indices = torch.randint(sample_len * self.num_seqs, size=(batch_size,), requires_grad=False)
        time_indices = torch.fmod(indices, sample_len)
        seq_indices = torch.div(indices, sample_len, rounding_mode="floor")
        return (
            self.states[time_indices, seq_indices],
            self.actions[time_indices, seq_indices],
            self.rewards[time_indices, seq_indices],
            self.undones[time_indices, seq_indices],
            self.states[time_indices + 1, seq_indices],
        )

    def sample_for_per(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        beg = -self.max_size
        end = (self.cur_size - self.max_size) if self.cur_size < self.max_size else -1

        assert batch_size % self.num_seqs == 0
        sub_batch_size = batch_size // self.num_seqs
        is_indices = []
        is_weights = []
        for env_index, sum_tree in enumerate(self.sum_trees or []):
            sampled_indices, sampled_weights = sum_tree.important_sampling(batch_size, beg, end, self.per_beta)
            is_indices.append(sampled_indices + sub_batch_size * env_index)
            is_weights.append(sampled_weights)

        index_tensor = torch.hstack(is_indices).to(self.device)
        weight_tensor = torch.hstack(is_weights).to(self.device)

        time_indices = torch.fmod(index_tensor, self.cur_size)
        seq_indices = torch.div(index_tensor, self.cur_size, rounding_mode="floor")
        return (
            self.states[time_indices, seq_indices],
            self.actions[time_indices, seq_indices],
            self.rewards[time_indices, seq_indices],
            self.undones[time_indices, seq_indices],
            self.states[time_indices + 1, seq_indices],
            weight_tensor,
            index_tensor,
        )

    def td_error_update_for_per(self, is_indices: Tensor, td_error: Tensor) -> None:
        prob = td_error.clamp(1e-8, 10).pow(self.per_alpha).squeeze(-1)
        batch_size = td_error.shape[0]
        sub_batch_size = batch_size // self.num_seqs
        for env_index, sum_tree in enumerate(self.sum_trees or []):
            start = env_index * sub_batch_size
            end = start + sub_batch_size
            sum_tree.update_ids(is_indices[start:end].cpu(), prob[start:end].cpu())

    def save_or_load_history(self, cwd: str, if_save: bool) -> None:
        item_names = (
            (self.states, "states"),
            (self.actions, "actions"),
            (self.rewards, "rewards"),
            (self.undones, "undones"),
        )

        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buffer_item = item[: self.cur_size]
                else:
                    buffer_item = torch.vstack((item[self.p : self.cur_size], item[0 : self.p]))
                torch.save(buffer_item, f"{cwd}/replay_buffer_{name}.pth")
            return

        expected_files = [f"{cwd}/replay_buffer_{name}.pth" for _, name in item_names]
        if not all(os.path.isfile(path) for path in expected_files):
            return

        max_sizes = []
        for item, name in item_names:
            buffer_item = torch.load(f"{cwd}/replay_buffer_{name}.pth")
            max_size = buffer_item.shape[0]
            item[:max_size] = buffer_item
            max_sizes.append(max_size)
        assert all(size == max_sizes[0] for size in max_sizes)
        self.cur_size = self.p = max_sizes[0]
        self.if_full = self.cur_size == self.max_size


class SumTree:
    """Binary tree used for prioritized experience replay."""

    def __init__(self, buf_len: int) -> None:
        self.buf_len = buf_len
        self.max_len = (buf_len - 1) + buf_len
        self.depth = math.ceil(math.log2(self.max_len))
        self.tree = torch.zeros(self.max_len, dtype=torch.float32)

    def update_id(self, data_id: int, prob: float = 10.0) -> None:
        tree_id = data_id + self.buf_len - 1
        delta = prob - self.tree[tree_id]
        self.tree[tree_id] = prob
        for _ in range(self.depth - 2):
            tree_id = (tree_id - 1) // 2
            self.tree[tree_id] += delta

    def update_ids(self, data_ids: Tensor, prob: Tensor = 10.0) -> None:
        leaf_ids = data_ids + self.buf_len - 1
        self.tree[leaf_ids] = prob
        for _ in range(self.depth - 2):
            parent_ids = torch.div(leaf_ids - 1, 2, rounding_mode="floor").unique()
            left_ids = parent_ids * 2 + 1
            right_ids = left_ids + 1
            self.tree[parent_ids] = self.tree[left_ids] + self.tree[right_ids]
            leaf_ids = parent_ids

    def get_leaf_id_and_value(self, value: float) -> Tuple[int, float]:
        parent_id = 0
        for _ in range(self.depth - 2):
            left_id = min(2 * parent_id + 1, self.max_len - 1)
            right_id = left_id + 1
            if value <= self.tree[left_id]:
                parent_id = left_id
            else:
                value -= self.tree[left_id]
                parent_id = right_id
        return parent_id, float(self.tree[parent_id])

    def important_sampling(self, batch_size: int, beg: int, end: int, per_beta: float) -> Tuple[Tensor, Tensor]:
        values = (torch.arange(batch_size) + torch.rand(batch_size)) * (self.tree[0] / batch_size)
        leaf_ids, leaf_values = list(zip(*[self.get_leaf_id_and_value(v) for v in values]))
        leaf_ids = torch.tensor(leaf_ids, dtype=torch.long)
        leaf_values = torch.tensor(leaf_values, dtype=torch.float32)

        indices = leaf_ids - (self.buf_len - 1)
        assert indices.max() < self.buf_len

        prob_ary = leaf_values / self.tree[beg:end].min()
        weights = torch.pow(prob_ary, -per_beta)
        return indices, weights
