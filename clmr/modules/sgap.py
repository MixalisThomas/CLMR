import torch
import torch.nn as nn
import torch.utils.checkpoint

from simclr.modules.gather import GatherLayer


class SGAP(nn.Module):
    """Smooth global average precision.

    A single global ranking over every (query, candidate) pair, not a mean of
    per-query APs: `compute_gap` takes flat score/label vectors and a positive's
    rank is estimated by summing a temperature-sigmoid over its score margin
    against every other pair.
    """

    def __init__(self, temperature=0.001, chunk_size=128, checkpoint_chunks=True):
        super().__init__()
        self.tau = temperature
        self.chunk_size = chunk_size
        self.checkpoint_chunks = checkpoint_chunks

    def tau_sigmoid(self, tensor):
        exponent = tensor / self.tau
        return torch.sigmoid(exponent.clamp(-50, 50))

    def sig_diff(self, tensor1, tensor2, target):
        return self.tau_sigmoid(tensor1 - tensor2)

    def heaviside(self, tensor, val=1.0):
        return torch.heaviside(
            tensor, values=torch.tensor(val, device=tensor.device, dtype=tensor.dtype)
        )

    def _chunk_gap(self, scores, target_f, idx_chunk):
        sig = self.tau_sigmoid(
            scores.unsqueeze(0) - scores[idx_chunk].unsqueeze(1)
        )  # (c, N)

        rank_all = sig.sum(dim=1) + 0.5
        rank_pos = (sig * target_f.unsqueeze(0)).sum(dim=1) + 0.5
        return (rank_pos / rank_all).sum()

    def compute_gap(self, scores, target):
        scores = scores.float()
        pos_idx = torch.nonzero(target == 1).squeeze(-1)
        n_pos = pos_idx.numel()
        if n_pos == 0:
            return scores.new_zeros(())
        gap = scores.new_zeros(())
        target_f = target.float() if target.dtype != scores.dtype else target

        use_ckpt = self.checkpoint_chunks and torch.is_grad_enabled() and scores.requires_grad
        for start in range(0, n_pos, self.chunk_size):
            end = min(start + self.chunk_size, n_pos)
            idx_chunk = pos_idx[start:end]
            if use_ckpt:
                contrib = torch.utils.checkpoint.checkpoint(
                    self._chunk_gap, scores, target_f, idx_chunk, use_reentrant=False
                )
            else:
                contrib = self._chunk_gap(scores, target_f, idx_chunk)
            gap = gap + contrib

        return gap / n_pos

    def _pairs_from_embeddings(self, embeddings, targets):
        similarities = torch.einsum("id,jd->ij", embeddings, embeddings)
        eye = torch.eye(
            similarities.shape[0], dtype=torch.bool, device=similarities.device
        )
        similarities = similarities[~eye].reshape(similarities.shape[0], -1)
        labels = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()
        labels = labels[~eye].reshape(labels.shape[0], -1)
        return similarities, labels

    def forward(self, embeddings, targets):
        similarities, labels = self._pairs_from_embeddings(embeddings, targets)
        return 1 - self.compute_gap(similarities.flatten(), labels.flatten())

    def forward_pairs(self, similarities, labels, exclude=None):
        """Entry point for the cross-GPU path.

        `cross_gpu_similarity` already returns this rank's (N, M) slice of the
        global similarity matrix, so recomputing a square matrix from embeddings
        would both duplicate that work and reintroduce the N^2 memory the slice
        exists to avoid. `exclude` masks the trivial self-pairs, which the caller
        has already set to -1.
        """
        scores, target = self._select(similarities, labels, exclude)
        return 1 - self.compute_gap(scores, target)

    def _select(self, similarities, labels, exclude=None):
        scores = similarities.flatten().float()
        target = labels.flatten().float()
        if exclude is not None:
            keep = ~exclude.flatten()
            scores, target = scores[keep], target[keep]
        return scores, target


class SGAPDrop(SGAP):
    """SGAP over a subsample of the pairs, keeping the hardest.

    Dropping is global over the flattened pair vector rather than per row, so a
    row whose negatives are all easy can end up contributing none -- that is the
    intended behaviour, not a bug: the loss is a single global ranking.
    """

    def __init__(self, temperature=0.001, drop_pos=0.0, drop_neg=0.90, chunk_size=128,
                 checkpoint_chunks=True):
        super().__init__(temperature, chunk_size, checkpoint_chunks)
        self.drop_pos = drop_pos
        self.drop_neg = drop_neg

    def _drop(self, similarities, labels):
        pos_sims = similarities[labels]
        neg_sims = similarities[~labels]

        # drop self-similarity from the positives
        pos_sims = pos_sims[~(pos_sims >= 0.9999)]

        if self.drop_pos > 0.0:
            # keep the LOWEST-scoring positives -- the ones the ranking gets wrong
            k = pos_sims.size(0) - int(pos_sims.size(0) * self.drop_pos)
            pos_sims = -torch.topk(-pos_sims, max(k, 1))[0]
        if self.drop_neg > 0.0:
            # keep the HIGHEST-scoring negatives -- hard-negative mining
            k = neg_sims.size(0) - int(neg_sims.size(0) * self.drop_neg)
            neg_sims = torch.topk(neg_sims, max(k, 1))[0]

        similarities = torch.cat([pos_sims, neg_sims], dim=0)
        labels = torch.cat([torch.ones_like(pos_sims), torch.zeros_like(neg_sims)], dim=0)
        return similarities, labels

    def forward(self, embeddings, targets):
        similarities, labels = self._pairs_from_embeddings(embeddings, targets)
        similarities, labels = self._drop(similarities, labels.bool())
        return 1 - self.compute_gap(similarities.flatten(), labels.flatten())

    def forward_pairs(self, similarities, labels, exclude=None):
        scores, target = self._select(similarities, labels, exclude)
        scores, target = self._drop(scores, target.bool())
        return 1 - self.compute_gap(scores, target)


class SGAPContrastive(SGAP):
    """CLMR-compatible wrapper around SGAP.

    Builds the same positive/negative pair structure NT_Xent uses -- each
    sample's positive is its paired augmented view (offset-diagonal trick),
    everything else in the concatenated 2*batch_size*world_size batch is a
    negative -- and scores the flattened result with the smooth global-AP
    objective instead of InfoNCE. This mirrors the (z_i, z_j) call signature
    of simclr.modules.NT_Xent so it drops in via ContrastiveLearning's
    configure_criterion() with no other changes to the training loop.

    Only `temperature` is exposed as a tunable in the current experiment;
    negative dropping is deliberately not wired in here (see SGAPDrop) to
    keep the ablation isolated to temperature alone.
    """

    def __init__(self, batch_size, temperature, world_size=1, chunk_size=128,
                 checkpoint_chunks=True):
        super().__init__(temperature=temperature, chunk_size=chunk_size,
                          checkpoint_chunks=checkpoint_chunks)
        self.batch_size = batch_size
        self.world_size = world_size
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        if self.world_size > 1:
            z_i = torch.cat(GatherLayer.apply(z_i), dim=0)
            z_j = torch.cat(GatherLayer.apply(z_j), dim=0)
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).flatten()
        negative_samples = sim[self.mask].flatten()

        similarities = torch.cat([positive_samples, negative_samples], dim=0)
        labels = torch.cat(
            [torch.ones_like(positive_samples), torch.zeros_like(negative_samples)],
            dim=0,
        )

        loss = 1 - self.compute_gap(similarities, labels)
        loss /= self.world_size
        return loss
