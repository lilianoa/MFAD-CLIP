import torch
import argparse
import torch.nn.functional as F
from dataset import Dataset, Dataset_csc, generate_class_info
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
import csv

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from tabulate import tabulate

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test(args, clip_model, fusion_module, prompt_learner, test_dataloader, obj, device):
    features_list = args.features_list
    results = {}
    metrics = {}
    results[obj] = {}
    results[obj]['gt_sp'] = []
    results[obj]['pr_sp'] = []
    results[obj]['imgs_masks'] = []
    results[obj]['anomaly_maps'] = []
    clip_model.eval()
    fusion_module.eval()
    prompt_learner.eval()
    prompts, tokenized_prompts = prompt_learner()

    with torch.no_grad():
        text_features = clip_model.encode_text(prompts, tokenized_prompts)
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        # text_features: (N,2,E) 
        normal_text_features = text_features[:, 0, :]
        abnormal_text_features = text_features[:, 1, :]
        # text_embedding: (L,E) -> (N,L,E) visual_features: (N,S,E)  L=1,S=1
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        bs = image.shape[0]
        cls_name = items['cls_name']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = clip_model.encode_image(image, features_list)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
            #########################################################################
            # fusion
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # (Np,2,E)x
            sim_nf_list = []
            sim_af_list = []
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

            sim_nf = torch.cat(sim_nf_list, dim=1)
            sim_af = torch.cat(sim_af_list, dim=1)

            sim_nf = (sim_nf / 0.07).softmax(-1)
            sim_af = (sim_af / 0.07).softmax(-1)
            #########################################################################

            # When there are N text prompts, take the average
            sim_nf = torch.mean(sim_nf, dim=1)  # (N, Np,2)->(N, 2)
            sim_af = torch.mean(sim_af, dim=1)
 
            sim_n = torch.mean(torch.stack([sim_nf[:, 0], sim_af[:, 0]], dim=-1), dim=-1, keepdim=True)  # (N,)
            sim_a = torch.mean(torch.stack([sim_nf[:, 1], sim_af[:, 1]], dim=-1), dim=-1, keepdim=True)

            text_probs = torch.cat([sim_n, sim_a], dim=-1)
            text_probs = text_probs[..., 1]

            #########################################################################
            # anomaly map
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    # fusion
                    similarity_nf_list = []
                    similarity_af_list = []
                    for idx, text_feature in enumerate(text_features):
                        normal_fusions_p_list = []
                        abnormal_fusions_p_list = []
                        for p_feat in patch_feature[:, 1:, :].transpose(0, 1):
                            normal_fusions_p = fusion_module(
                                normal_text_features[idx].expand(bs, 1, -1).transpose(0, 1),
                                p_feat.unsqueeze(1).transpose(0, 1))  # fusion_emd:(S,N,E)
                            abnormal_fusions_p = fusion_module(
                                abnormal_text_features[idx].expand(bs, 1, -1).transpose(0, 1),
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
                    similarity_nf = torch.stack(similarity_nf_list, dim=0)
                    similarity_af = torch.stack(similarity_af_list, dim=0)

                    # When there are N text prompts, take the average
                    similarity_nf = torch.mean(similarity_nf, dim=0)  # (Np, N, w*h, 2)
                    similarity_af = torch.mean(similarity_af, dim=0)


                    similarity_n = torch.mean(torch.stack([similarity_nf[..., 0], similarity_af[..., 0]], dim=-1),
                                              dim=-1,
                                              keepdim=True)
                    similarity_a = torch.mean(torch.stack([similarity_nf[..., 1], similarity_af[..., 1]], dim=-1),
                                              dim=-1,
                                              keepdim=True)

                    similarity_map_n = clip.get_similarity_map(similarity_n, args.image_size)
                    similarity_map_a = clip.get_similarity_map(similarity_a, args.image_size)

                    similarity_map = torch.cat([similarity_map_n, similarity_map_a], dim=-1)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim=0)

            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)

    table = []
    table.append(obj)
    results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
    results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
    if args.metrics == 'image-level':
        image_auroc = image_level_metrics(results, obj, "image-auroc")
        image_ap = image_level_metrics(results, obj, "image-ap")
        image_f1max = image_level_metrics(results, obj, "image-f1max")
        print("image-auroc", image_auroc)
        print("image-ap", image_ap)
        print("image-f1max", image_f1max)
        table.append(str(np.round(image_auroc * 100, decimals=1)))
        table.append(str(np.round(image_ap * 100, decimals=1)))
        table.append(str(np.round(image_f1max * 100, decimals=1)))
        return table, image_auroc, image_ap, image_f1max

    elif args.metrics == 'pixel-level':
        pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
        pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
        pixel_f1max = pixel_level_metrics(results, obj, "pixel-f1max")
        table.append(str(np.round(pixel_auroc * 100, decimals=1)))
        table.append(str(np.round(pixel_aupro * 100, decimals=1)))
        table.append(str(np.round(pixel_f1max * 100, decimals=1)))
        print("pixel-auroc", pixel_auroc)
        print("pixel-aupro", pixel_aupro)
        print("pixel-f1max", pixel_f1max)
        return table, pixel_auroc, pixel_aupro, pixel_f1max

    elif args.metrics == 'image-pixel-level':
        image_auroc = image_level_metrics(results, obj, "image-auroc")
        image_ap = image_level_metrics(results, obj, "image-ap")
        image_f1max = image_level_metrics(results, obj, "image-f1max")
        print("image-auroc", image_auroc)
        print("image-ap", image_ap)
        print("image-f1max", image_f1max)
        pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
        pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
        pixel_f1max = pixel_level_metrics(results, obj, "pixel-f1max")
        print("pixel-auroc", pixel_auroc)
        print("pixel-aupro", pixel_aupro)
        print("pixel-f1max", pixel_f1max)

        table.append(str(np.round(pixel_auroc * 100, decimals=1)))
        table.append(str(np.round(pixel_aupro * 100, decimals=1)))
        table.append(str(np.round(pixel_f1max * 100, decimals=1)))
        table.append(str(np.round(image_auroc * 100, decimals=1)))
        table.append(str(np.round(image_ap * 100, decimals=1)))
        table.append(str(np.round(image_f1max * 100, decimals=1)))

        return table, image_auroc, image_ap, image_f1max, pixel_auroc, pixel_aupro, pixel_f1max


def main(args):
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
    clip_model.visual.VVattn_replace(VVattn_layers=20)
    prompt_learner = PromptLearner(clip_model, prompts_cfg, classnames=["object"])
    fusion_module = IT_fusion(clip_model.visual.output_dim, nhead=8, dropout=0.2)

    if args.class_name:
        obj_list = [args.class_name]
    else:
        obj_list, _ = generate_class_info(args.dataset)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    image_f1max_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    pixel_f1max_list = []

    checkpoint_path = os.path.join(args.checkpoint_path, "epoch_15.pth")
    checkpoint = torch.load(checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    clip_model.to(device)
    fusion_module.load_state_dict(checkpoint["fusion_module"])
    fusion_module.to(device)

    for obj in obj_list:
        # checkpoint_path = os.path.join(args.checkpoint_path, str(obj) + "/epoch_15.pth")

        ##########################################################################################
        test_data = Dataset_csc(root=args.data_path, transform=preprocess, target_transform=target_transform,
                                dataset_name=args.dataset, mode='test', class_names=[obj])
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
        print(f"Test on {obj}.")
        ##########################################################################################
        if args.metrics == 'image-level':
            table, image_auroc, image_ap, image_f1max = test(args, clip_model,
                                                                                                fusion_module,
                                                                                                prompt_learner,
                                                                                                test_dataloader, obj,
                                                                                                device)
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            image_f1max_list.append(image_f1max)
            table_ls.append(table)

        elif args.metrics == 'pixel-level':
            table, pixel_auroc, pixel_aupro, pixel_f1max = test(args, clip_model,
                                                            fusion_module,
                                                            prompt_learner,
                                                            test_dataloader, obj,
                                                            device)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1max_list.append(pixel_f1max)
            table_ls.append(table)

        elif args.metrics == 'image-pixel-level':
            table, image_auroc, image_ap, image_f1max, pixel_auroc, pixel_aupro, pixel_f1max = test(args, clip_model,
                                                                                                fusion_module,
                                                                                                prompt_learner,
                                                                                                test_dataloader, obj,
                                                                                                device)
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            image_f1max_list.append(image_f1max)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
            pixel_f1max_list.append(pixel_f1max)
            table_ls.append(table)

    table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                     str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                     str(np.round(np.mean(pixel_f1max_list) * 100, decimals=1)),
                     str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                     str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                     str(np.round(np.mean(image_f1max_list) * 100, decimals=1))])
    results = tabulate(table_ls,
                       headers=['objects', 'pixel_auroc', 'pixel_aupro', 'pixel_f1max', 'image_auroc', 'image_ap',
                                'image_f1max'],
                       tablefmt="pipe")
    logger.info("\n%s", results)

    csv_path = os.path.join(args.save_path, f'{args.dataset}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in results.split('\n'):
            writer.writerow(row.split('|'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TPDCLIP", add_help=True)
    parser.add_argument("--clip_name", type=str, default='ViT-L/14@336px', help="CLIP name")
    parser.add_argument("--data_path", type=str, default="./data/visa", help="dataset path")
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/mvtec', help='path to chechpoints')
    parser.add_argument("--save_path", type=str, default='./results/visa', help='path to save results')

    parser.add_argument("--dataset", type=str, default='visa', help="dataset name")
    parser.add_argument("--class_name", type=str, default=None, help="class name")

    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--n_prompts", type=int, default=2, help="number of text prompt")

    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="features used")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')

    parser.add_argument("--batch_size", type=int, default=24, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    args = parser.parse_args()
    setup_seed(args.seed)
    main(args)


