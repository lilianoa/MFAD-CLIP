import torch
import argparse
import torch.nn.functional as F
from loss import FocalLoss, BinaryDiceLoss
from dataset import Dataset, Dataset_csc
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from model.model import PromptLearner
import clip
from clip import load
from fusion import IT_fusion
from criterions import triplet_loss

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    features_list = args.features_list
    logger = get_logger(args.save_path)
    logger.info(args)

    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts_cfg = {"n_ctx": args.n_ctx,
                   "n_prompts": args.n_prompts,
                   "ctx_init_pos": None,
                   "ctx_init_neg": None}

    clip_cfg = {"clip_name": args.clip_name,
                "last_n_layers": 12}

    clip_model, _ = load(clip_cfg["clip_name"], device="cpu", cliptype="TPDCLIP", design_details=None)
    patch_size = clip_model.visual.conv1.weight.shape[-1]
    # grid_size = model.visual.positional_embedding.shape[0] - 1
    grid_size = args.image_size // patch_size

    print("Turning off gradients in both the image and the text encoder")
    for name, param in clip_model.named_parameters():
        param.requires_grad_(False)

    prompt_learner = PromptLearner(clip_model, prompts_cfg, classnames=["object"])
    prompt_learner.to(device)
    clip_model.to(device)
    clip_model.visual.VVattn_replace(VVattn_layers=20)
    fusion_module = IT_fusion(clip_model.visual.output_dim, nhead=8, dropout=0.2)
    fusion_module.to(device)

    ##########################################################################################
    if args.class_name:
        train_data = Dataset_csc(root=args.train_data_path, transform=preprocess, target_transform=target_transform,
                                 dataset_name=args.dataset, mode='train', class_names=[args.class_name])
    else:
            train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform,
                             dataset_name=args.dataset, mode='test')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # train_cls_dataloader = get_dataloader(dataset=train_cls_data, phase='train', batch_size=args.batch_size)
    ##########################################################################################

    optimizer_params = [
        {"params": [p for p in prompt_learner.parameters()]},
        {"params": [p for p in fusion_module.parameters()]},
    ]
    optimizer = torch.optim.Adam(optimizer_params, lr=args.learning_rate, betas=(0.5, 0.999))

    ##########################################################################################
    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    clip_model.train()
    prompt_learner.train()
    fusion_module.train()
    for epoch in tqdm(range(args.epoch)):
        loss_list = []
        loss_ce_list = []
        loss_p_list = []
        loss_c1_list = []
        loss_c2_list = []
        loss_c3_list = []
        loss_c4_list = []
        loss_p_c1_list = []
        loss_p_c2_list = []
        loss_p_c3_list = []
        loss_p_c4_list = []
        loss_oc_list = []

        for items in train_dataloader:
            image = items['img'].to(device)
            label = items['anomaly'].to(device)
            bs = image.shape[0]
            gt = items['img_mask'].to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
              image_features, patch_features = clip_model.encode_image(image, features_list, layer=20)

            prompts, tokenized_prompts = prompt_learner()

            text_features = clip_model.encode_text(prompts, tokenized_prompts)
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            # text_features: (Np,2,E)   
            normal_text_features = text_features[:, 0, :]
            abnormal_text_features = text_features[:, 1, :]

            # Orthogonal regularization
            normal_text_n = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
            abnormal_text_n = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            s_n = torch.abs(normal_text_n @ normal_text_n.t())
            s_a = torch.abs(abnormal_text_n @ abnormal_text_n.t())
            s_n_sum = s_n.sum() - torch.trace(s_n)
            s_a_sum = s_a.sum() - torch.trace(s_a)
            loss_oc = s_n_sum + s_a_sum

            # fusion
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (Np,2,E)
            sim_nf_list = []
            sim_af_list = []
            loss_c1 = 0
            loss_c2 = 0
            loss_c3 = 0
            loss_c4 = 0
            for idx, text_feature in enumerate(text_features):
                normal_fusions = fusion_module(normal_text_features[idx].expand(bs, 1, -1).transpose(0, 1),
                                               image_features.unsqueeze(1).transpose(0, 1))  # fusion_emd:(S,N,E)
                abnormal_fusions = fusion_module(abnormal_text_features[idx].expand(bs, 1, -1).transpose(0, 1),
                                                 image_features.unsqueeze(1).transpose(0, 1))
                normal_fusions = normal_fusions.transpose(0, 1)  # N,L,C
                abnormal_fusions = abnormal_fusions.transpose(0, 1)  # N,L,C

                
                normal_fusions = normal_fusions / normal_fusions.norm(dim=-1, keepdim=True)   # cosine similarity
                abnormal_fusions = abnormal_fusions / abnormal_fusions.norm(dim=-1, keepdim=True)  # cosine similarity
                
                sim_nf = normal_fusions @ text_feature.unsqueeze(0).permute(0, 2, 1)
                sim_nf_list.append(sim_nf)
                
                sim_af = abnormal_fusions @ text_feature.unsqueeze(0).permute(0, 2, 1)
                sim_af_list.append(sim_af)

                #########################################################################
                # contrastive loss
                normal_fusions = normal_fusions.squeeze(1)  # fusion_emd:(N,E)
                abnormal_fusions = abnormal_fusions.squeeze(1)
                # normal_fusions
                sim_normal_fu = normal_fusions @ normal_fusions.t()
                # normal_fusions and abnormal_fusions
                sim_nta_fu = normal_fusions @ abnormal_fusions.t()
                # abnormal_fusions
                sim_abnormal_fu = abnormal_fusions @ abnormal_fusions.t()

                # N_t+N_i-> <-N_t+N_i， N_t+N_i<-->A_t+N_i
                # loss_c1 = contrive_loss(sim_normal_fu, sim_nta_fu, 1-label, 1-label, device)
                loss_c1 += triplet_loss(sim_normal_fu, sim_nta_fu, 1-label, 1-label, device)

                # A_t+A_i-> <-A_t+A_i， A_t+A_i<-->N_t+A_i
                # loss_c2 = contrive_loss(sim_abnormal_fu, sim_nta_fu.t(), label, label, device)
                loss_c2 += triplet_loss(sim_abnormal_fu, sim_nta_fu.t(), label, label, device)

                # N_t+N_i-> < -N_t+N_i， N_t+N_i< -->N_t+A_i
                # loss_c3 = contrive_loss(sim_normal_fu, sim_normal_fu, 1-label, label, device)
                loss_c3 += triplet_loss(sim_normal_fu, sim_normal_fu, 1-label, label, device)

                # A_t+A_i-> <-A_t+A_i， A_t+A_i<-->A_t+N_i
                # loss_c4 = contrive_loss(sim_abnormal_fu, sim_abnormal_fu, label, 1-label, device)
                loss_c4 += triplet_loss(sim_abnormal_fu, sim_abnormal_fu, label, 1-label, device)

                #########################################################################
            sim_nf = torch.cat(sim_nf_list, dim=1)
            sim_af = torch.cat(sim_af_list, dim=1)

            loss_c1 = loss_c1 / text_features.shape[0]
            loss_c2 = loss_c2 / text_features.shape[0]
            loss_c3 = loss_c3 / text_features.shape[0]
            loss_c4 = loss_c4 / text_features.shape[0]
            #########################################################################
            # image loss

            # When there are N text prompts, take the average
            sim_nf = torch.mean(sim_nf, dim=1)  # (N, Np,2)->(N, 2)
            sim_af = torch.mean(sim_af, dim=1)

            
            sim_n = torch.mean(torch.stack([sim_nf[:, 0], sim_af[:, 0]], dim=-1), dim=-1, keepdim=True)  # (N,)
            sim_a = torch.mean(torch.stack([sim_nf[:, 1], sim_af[:, 1]], dim=-1), dim=-1, keepdim=True)

            probs = torch.cat([sim_n, sim_a], dim=-1) / 0.07
            loss_ce = F.cross_entropy(probs, label.long().cuda())

            #########################################################################
            # patch loss
            similarity_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    # fusion
                    similarity_nf_list = []
                    similarity_af_list = []
                    loss_pc1_list = []
                    loss_pc2_list = []
                    loss_pc3_list = []
                    loss_pc4_list = []
                    for idx, text_feature in enumerate(text_features):
                        normal_fusions_p_list = []
                        abnormal_fusions_p_list = []
                        for p_feat in patch_feature[:, 1:, :].transpose(0, 1):
                            normal_fusions_p = fusion_module(normal_text_features[idx].expand(bs, 1, -1).transpose(0, 1),
                                                             p_feat.unsqueeze(1).transpose(0, 1))  # fusion_emd:(S,N,E)
                            abnormal_fusions_p = fusion_module(abnormal_text_features[idx].expand(bs, 1, -1).transpose(0, 1),
                                                               p_feat.unsqueeze(1).transpose(0, 1))
                            normal_fusions_p = normal_fusions_p.transpose(0, 1)  # N,1,C
                            abnormal_fusions_p = abnormal_fusions_p.transpose(0, 1)  # N,1,C
                            normal_fusions_p_list.append(normal_fusions_p)
                            abnormal_fusions_p_list.append(abnormal_fusions_p)
                        normal_fusions_p = torch.cat(normal_fusions_p_list, dim=1)  # (N,L,C)
                        abnormal_fusions_p = torch.cat(abnormal_fusions_p_list, dim=1)

                        
                        normal_fusions_p = normal_fusions_p / normal_fusions_p.norm(dim=-1, keepdim=True)
                        abnormal_fusions_p = abnormal_fusions_p / abnormal_fusions_p.norm(dim=-1, keepdim=True)

                        similarity_nf = clip.compute_similarity(normal_fusions_p, text_feature)
                        similarity_af = clip.compute_similarity(abnormal_fusions_p, text_feature)
                        similarity_nf_list.append(similarity_nf)
                        similarity_af_list.append(similarity_af)

                        #########################################################################
                        # patch contrastive loss
                        for i_b, l in enumerate(label):
                            if l:
                                img_mask = gt[i_b]
                                img_mask = img_mask.reshape(grid_size, patch_size, grid_size, patch_size)
                                img_mask = img_mask.permute(0, 2, 1, 3)
                                img_mask = img_mask.reshape(grid_size * grid_size, -1)
                                num_patch = img_mask.shape[-1] * 0.1  # More than 10% of the pixels in the patch are marked as abnormal
                                patch_label = torch.sum(img_mask, dim=-1)
                                patch_label[patch_label < num_patch] = 0  
                                patch_label[patch_label >= num_patch] = 1  
                                patch_label.to(device)
                                # similarity of normal_fusions_p:(N,L,E)
                                sim_normal_fu_p = normal_fusions_p[i_b] @ normal_fusions_p[i_b].t()
                                # normal_fusions_p and abnormal_fusions_p
                                sim_nta_fu_p = normal_fusions_p[i_b] @ abnormal_fusions_p[i_b].t()
                                # abnormal_fusions_p
                                sim_abnormal_fu_p = abnormal_fusions_p[i_b] @ abnormal_fusions_p[i_b].t()

                                loss_pc1 = triplet_loss(sim_normal_fu_p, sim_nta_fu_p, 1 - patch_label, 1 - patch_label,
                                                        device)
                                loss_pc2 = triplet_loss(sim_abnormal_fu_p, sim_nta_fu_p.t(), patch_label, patch_label,
                                                        device)

                                loss_pc3 = triplet_loss(sim_normal_fu_p, sim_normal_fu_p, 1 - patch_label, patch_label,
                                                        device)
                                loss_pc4 = triplet_loss(sim_abnormal_fu_p, sim_abnormal_fu_p, patch_label,
                                                        1 - patch_label, device)

                                loss_pc1_list.append(loss_pc1)
                                loss_pc2_list.append(loss_pc2)
                                loss_pc3_list.append(loss_pc3)
                                loss_pc4_list.append(loss_pc4)
                        #########################################################################
                    if loss_pc1_list:
                        loss_p_c1 = torch.mean(torch.stack(loss_pc1_list), 0)
                        loss_p_c2 = torch.mean(torch.stack(loss_pc2_list), 0)
                        loss_p_c3 = torch.mean(torch.stack(loss_pc3_list), 0)
                        loss_p_c4 = torch.mean(torch.stack(loss_pc4_list), 0)
                    else:
                        loss_p_c1 = loss_p_c2 = loss_p_c3 = loss_p_c4 = torch.zeros_like(loss_c1)

                    similarity_nf = torch.stack(similarity_nf_list, dim=0)
                    similarity_af = torch.stack(similarity_af_list, dim=0)

                    # When there are N text prompts, take the average
                    similarity_nf = torch.mean(similarity_nf, dim=0)  # (Np, N, w*h, 2)
                    similarity_af = torch.mean(similarity_af, dim=0)

                    similarity_n = torch.mean(torch.stack([similarity_nf[..., 0], similarity_af[..., 0]], dim=-1), dim=-1,
                                              keepdim=True)
                    similarity_a = torch.mean(torch.stack([similarity_nf[..., 1], similarity_af[..., 1]], dim=-1), dim=-1,
                                              keepdim=True)

                    similarity_map_n = clip.get_similarity_map(similarity_n, args.image_size)
                    similarity_map_a = clip.get_similarity_map(similarity_a, args.image_size)

                    similarity_map = torch.cat([similarity_map_n, similarity_map_a], dim=-1).permute(0, 3, 1, 2)

                    similarity_map_list.append(similarity_map)

            loss_p = 0
            for i in range(len(similarity_map_list)):
                loss_p += loss_focal(similarity_map_list[i], gt)
                loss_p += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss_p += loss_dice(similarity_map_list[i][:, 0, :, :], 1-gt)

            loss = loss_ce + loss_p + args.alpha * (
                        loss_c1 + loss_c2 + loss_c3 + loss_c4 + loss_p_c1 + loss_p_c2 + loss_p_c3 + loss_p_c4) + args.beta * loss_oc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            loss_ce_list.append(loss_ce.item())
            loss_p_list.append(loss_p.item())
            loss_c1_list.append(loss_c1.item())
            loss_c2_list.append(loss_c2.item())
            loss_c3_list.append(loss_c3.item())
            loss_c4_list.append(loss_c4.item())
            loss_p_c1_list.append(loss_p_c1.item())
            loss_p_c2_list.append(loss_p_c2.item())
            loss_p_c3_list.append(loss_p_c3.item())
            loss_p_c4_list.append(loss_p_c4.item())
            loss_oc_list.append(loss_oc.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                'epoch [{}/{}], loss:{:.4f}, loss_ce:{:.4f}, loss_p:{:.4f}, loss_oc:{:.4f}, \n \
                loss_c1:{:.4f}, loss_c2:{:.4f}, loss_c3:{:.4f}, loss_c4:{:.4f}, \n \
                loss_p_c1:{:.4f}, loss_p_c2:{:.4f}, loss_p_c3:{:.4f}, loss_p_c4:{:.4f}'.format(
                    epoch + 1, args.epoch, np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_p_list), np.mean(loss_oc_list),
                    np.mean(loss_c1_list), np.mean(loss_c2_list), np.mean(loss_c3_list), np.mean(loss_c4_list),
                    np.mean(loss_p_c1_list), np.mean(loss_p_c2_list), np.mean(loss_p_c3_list), np.mean(loss_p_c4_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict(), "fusion_module": fusion_module.state_dict()}, ckp_path)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("TPDCLIP", add_help=True)
    parser.add_argument("--clip_name", type=str, default='ViT-L/14@336px', help="CLIP name")
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/mvtec', help='path to save results')

    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--class_name", type=str, default=None, help="train class name")

    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--n_prompts", type=int, default=2, help="number of normal text prompt")
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of triplet_loss")
    parser.add_argument("--beta", type=float, default=0.5, help="weight of oc_loss")

    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)


