import torch
import torch.nn as nn
import torch.nn.functional as F

def contrive_loss(sim_pos: torch.Tensor, sim_neg: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor, device):
    criterion = nn.CrossEntropyLoss()
    # sim_pos: N*N
    # sim_neg: N*N
    # pos_mask:K
    pos_mask = torch.nonzero(mask_pos).squeeze()
    sim_pos_mask = sim_pos[:, pos_mask][pos_mask, :]
    l_pos, _ = torch.min(sim_pos_mask, dim=0)  # l_pos: K
    # negative logit: K*K
    neg_mask = torch.nonzero(mask_neg).squeeze()
    if torch.equal(neg_mask, pos_mask):
        l_neg = sim_neg[:, neg_mask][neg_mask, :]  # l_neg: K*K
    else:
        l_neg = sim_neg[:, neg_mask][pos_mask, :]  # l_neg: K*H
    # logit: K*(1+K)/K*(1+H)
    logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)

    # contrastive loss
    labels_con = torch.zeros([logits.shape[0]], dtype=torch.long).to(device)  # Positives are in 0-th
    loss = criterion(logits, labels_con).to(device)
    return loss


def triplet_loss(sim_pos, sim_neg, mask_pos, mask_neg, device):
    # hard mining
    # sim_pos: B*N*N
    # sim_ap: the similarity between the anchor and the positive sample
    mask_dim = mask_pos.dim()
    if mask_dim == 1:
        n = mask_pos.shape[0]
        mask_pos = mask_pos.view(n, -1).expand(n, n).eq(mask_pos.view(n, -1).expand(n, n).transpose(1, 0)).float()
    elif mask_dim == 2:  # This situation occurs when the input is the similarity of patches
        b, n = mask_pos.shape[0], mask_pos.shape[1]
        mask_pos = mask_pos.view(b, n, -1).expand(b, n, n).eq(mask_pos.view(b, n, -1).expand(b, n, n).transpose(2, 1)).float()
    sim_pos_, _ = torch.min(sim_pos * mask_pos + (1 - mask_pos) * 1e9, dim=-1)
    sim_ap, _ = torch.min(sim_pos_, dim=-1)

    # hard mining
    # sim_neg: B*N*N
    # sim_an: the similarity between the anchor and the negative sample
    if mask_dim == 1:
        n = mask_pos.shape[0]
        mask_neg = mask_neg.view(n, -1).expand(n, n).eq(mask_neg.view(n, -1).expand(n, n).transpose(1, 0)).float()
    elif mask_dim == 2:
        b, n = mask_neg.shape[0], mask_neg.shape[1]
        mask_neg = mask_neg.view(b, n, -1).expand(b, n, n).eq(mask_neg.view(b, n, -1).expand(b, n, n).transpose(2, 1)).float()
    sim_neg_, _ = torch.max(sim_neg * mask_neg, dim=-1)
    sim_an, _ = torch.max(sim_neg_, dim=-1)

    # triplet loss
    label_tri = sim_an.new().resize_as_(sim_an).fill_(1).to(device)
    loss = F.margin_ranking_loss(sim_ap, sim_an, label_tri, margin=0.2)
    return loss