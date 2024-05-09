import os
import copy
import time
from tqdm import trange, tqdm

import torch
import pandas as pd
import clip.clip as clip
from clip.loss import ClipLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from typing import List
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests
import pickle
import random
import pdb
import math
import wandb
import numpy as np

def set_seed(seed: int = 42, if_torch: bool=True) -> None:
    """
    Set random
    """
    np.random.seed(seed)
    random.seed(seed)
    if if_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(0)

def seq_curri_guid(list_guidance: List, cur_guidance_id=None, cur_str_times=None, ctype='out_curri', loop_times=1):
    # sequentially use guidance 
    if ctype == 'no_curri':
        # iteratively loop over all guidance
        cur_guidance_id += 1
        if cur_guidance_id >= len(list_guidance):
            cur_guidance_id = 0  # guidance = 0
        cur_guidance = list_guidance[cur_guidance_id]
        return cur_guidance_id, cur_guidance

    elif ctype == 'in_curri':
        # have fixed curriculum length
        if cur_str_times < loop_times:
            # cur_guidance_id unchanged 
            cur_str_times += 1
        else:
            cur_str_times = 1
            cur_guidance_id += 1

            if cur_guidance_id >= len(list_guidance):
                cur_guidance_id = len(list_guidance) - 1

        cur_guidance = list_guidance[cur_guidance_id]
        return cur_guidance_id, cur_guidance, cur_str_times

    elif ctype == 'out_curri':
        cur_guidance = 100
        cur_str_times = 1
        cur_guidance_id = list_guidance.index(cur_guidance)
        return cur_guidance_id, cur_guidance, cur_str_times
    else:
        raise ValueError(f"invalid ctype {ctype}")


def explore_guid(args, epoch, logger, largest_guid, list_progress):
    # randomly select a guid with 15%, use the largest with 85%
    rand_prob = random.uniform(0, 1)
    if args.tau_curriculum:
        tau_thres = 0.75 - 0.5 * epoch / args.curriculum_epoch
    else:
        if not args.explore_fixguid:
            tau_thres = 0.15
        else:
            tau_thres = 0.5

    if rand_prob <= tau_thres:
        if args.explore_fixguid:
            #############################
            # choose guid from a fixed sequence of guid
            # fix_guid = sorted(list_guidance, reverse=False)
            # fix_guid_id = int(epoch / args.curriculum_epoch * len(list_guidance))
            # next_guid = [list_guidance[fix_guid_id], 0]
            # logger.info(f"Select from sequence guid = {next_guid[0]}")

            #############################
            # select 100 guid
            next_guid = [100, 0]
            logger.info(f"Select guid = {next_guid[0]}")
        else:
            next_guid = random.choice(list_progress)
            logger.info(f"Randomly select guid = {next_guid[0]}")
    else:
        next_guid = largest_guid
        logger.info(f"Select largest guid = {next_guid[0]}")
    return tau_thres, next_guid


def load_data(logger, args, clip_encoder, cur_guidance=None, cur_str_times=1, epoch=0, ori_proportion=None,
              uniform_guid=False, include_neg=False, list_imgs=None):
    if cur_guidance is not None:
        logger.info(f"loading image guidance = {cur_guidance}, loop times {cur_str_times}")
        if not args.debug:
            wandb.log({"Epoch": epoch, "Image Guidance": cur_guidance})
            if ori_proportion is not None:
                wandb.log({"Epoch": epoch, "Porportion of 100": ori_proportion})

    # load dataloader
    img_text_data = get_data(args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0,
                             merge_ori=args.merge_ori, subsample=args.subsample, return_img_id=True,
                             datalimit=args.datalimit, guidance=cur_guidance, list_imgs=list_imgs,
                             ori_proportion=ori_proportion, uniform_guid=uniform_guid, include_neg=include_neg,
                             logger=logger)
    assert len(img_text_data), 'At least one train or eval dataset must be specified.'

    ft_dataloader = img_text_data['train_ft'].dataloader
    if not args.debug:
        wandb.log({"Epoch": epoch, "Cur Dataloader Batch": len(ft_dataloader)})
    return ft_dataloader


def generate_class_head(model, args, epoch):
    # get classification head embedding
    args.current_epoch = epoch
    classification_head_new = get_zeroshot_classifier(args, model.module.model)
    classification_head_new = classification_head_new.cuda()
    return classification_head_new


def general_eval(model, args, stats, epoch: int, logger, print_log=False, print_class=False, log_dir=None,
                 wandb_comment=''):
    """

    :param model:
    :param args:
    :param stats:
    :param epoch:
    :param logger:
    :param print_log:
    :param print_class:
    :param log_dir:
    :param wandb_comment:
    :return:
    """

    epoch_stats = {}
    epoch_stats['Epoch'] = epoch
    epoch_stats['epoch'] = epoch
    classification_head_new = generate_class_head(model, args, epoch)
    _ = evaluate(model, args, classification_head_new, epoch_stats, logger=logger)

    ood_acc = 0
    num_datasets = 0
    for k, v in epoch_stats.items():
        if 'Accuracy' in k and 'Class' not in k:
            if k == 'ImageNet Accuracy':
                # ignore the ID acc term
                continue
            ood_acc += v
            num_datasets += 1
    if num_datasets != 0:
        ood_acc = ood_acc / num_datasets
    else:
        ood_acc = 0

    if print_class and log_dir is not None:
        # class acc:
        class_stats = dict()
        ind_dataset = {k: i for i, k in enumerate(args.eval_datasets)}
        for k, v in epoch_stats.items():
            if 'Class' not in k:
                continue
            if k == 'ImageNet Accuracy':
                # ignore the ID acc term
                continue

            list_k = k.split(' Class ')
            dataset_n = list_k[0]
            ds_id = ind_dataset[dataset_n]
            if 'Accuracy' in k:
                cls_label = list_k[1].replace(' Accuracy', '')
                cur_label_name = f"Class {cls_label}"
                if cur_label_name not in class_stats:
                    cur_res = [0] * 2 * len(args.eval_datasets)
                    class_stats[cur_label_name] = cur_res
                class_stats[cur_label_name][2 * ds_id] = v
            elif 'Number' in k:
                cls_label = list_k[1].replace(' Number', '')
                cur_label_name = f"Class {cls_label}"
                if cur_label_name not in class_stats:
                    cur_res = [0] * 2 * len(args.eval_datasets)
                    class_stats[cur_label_name] = cur_res
                class_stats[cur_label_name][2 * ds_id + 1] = v

        list_colum = [''] * 2 * len(args.eval_datasets)
        for i in range(len(args.eval_datasets)):
            list_colum[2 * i] = args.eval_datasets[i]
            list_colum[2 * i + 1] = args.eval_datasets[i] + ' Count'

        class_stats_df = pd.DataFrame.from_dict(class_stats, orient='index', columns=list_colum)
        class_stats_df.to_csv(log_dir + f'/class_stats{epoch}.tsv', sep='\t')

    epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
    if print_log:
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")

    # logger.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
    # epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
    epoch_stats = {key: values for key, values in epoch_stats.items() if ' Class' not in key}
    if args.train_dataset == 'ImageNet':
        epoch_stats = {f"{wandb_comment}{key}" if 'Accuracy' in key else key: values for key, values in epoch_stats.items()}
    else:
        epoch_stats = {f"{wandb_comment}{key}" if 'IWildCam' in key else key: values for key, values in epoch_stats.items()}

    if log_dir is not None:
        del epoch_stats['dict_img_guid']
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    ################################################
    # Evaluation logging
    if not args.debug:
        wandb.log(epoch_stats)
    return stats


def kw_test(Dict_guid_progs, list_guid):
    # Perform Kruskal-Wallis Test
    k_statistic, p_value_kw = stats.kruskal(Dict_guid_progs[list_guid[0]], Dict_guid_progs[list_guid[1]],
                                            Dict_guid_progs[list_guid[2]], Dict_guid_progs[list_guid[3]])
    print(f"Kruskal-Wallis Test: H-statistic = {k_statistic}, P-value = {p_value_kw}")
    print(f"=" * 50)

    comparisons = []
    for i in range(len(list_guid) - 1):
        for j in range(i + 1, len(list_guid)):
            comparisons.append((list_guid[i], list_guid[j]))
    # comparisons = [(100, 90), (100, 70), (100, 50), (90, 70), (90, 50), (70, 50)]
    p_values = []
    p_values = []

    for combo in comparisons:
        stat, p = mannwhitneyu(Dict_guid_progs[combo[0]], Dict_guid_progs[combo[1]], alternative='two-sided')
        p_values.append(p)

    # Apply Bonferroni Correction
    corrected_p = multipletests(p_values, method='bonferroni', alpha=0.01)

    # Determine the best category based on how many times it comes out as better
    best_category_count = {cat: 0 for cat in list_guid}
    # Interpret the corrected p-values and compare median or mean ranks
    for (x, y), p_val, reject in zip(comparisons, corrected_p[1], corrected_p[0]):
        if np.median(Dict_guid_progs[x]) > np.median(Dict_guid_progs[y]):
            higher = x
        else:
            higher = y

        best_category_count[higher] += 1

        if reject:
            print(f"{x} vs {y}: P-value = {p_val:.4f}, {higher} is significantly higher.")

        else:
            print(
                f"{x} vs {y}: P-value = {p_val:.4f}, No significant difference, {higher} with a slightly higher median.")

    return best_category_count


def progress_eval(model, args, last_perform, epoch: int, logger, progress_guid=False, progress_sample=False,
                  progress_ma=None, print_log=True, sel_imgs=None, prev_probs=None, unif_begin=False):
    """
    Find best guidance based on guid group
    :param print_log:
    :param model:
    :param args:
    :param last_perform:
    :param epoch:
    :param logger:
    :param progress_guid:
    :param progress_sample:
    :param progress_ma:
    :return:
    """

    def find_best_progress(dict_guid_prog):
        dict_guid_res = kw_test(dict_guid_prog, list_guid=list(dict_guid_prog.keys()))
        return res_progress

    def remove_outliers(data):
        # Calculate Q1, Q3, and IQR
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # Determine outliers using IQR
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
        return filtered_data

    def rnd_prog(input_progress, ):
        return np.round(input_progress, 6)

    classification_head_new = generate_class_head(model, args, epoch)
    Dict_cur_guidance = {}
    if not args.partial_update:
        sel_imgs = None

    if unif_begin and progress_sample and sel_imgs is not None and args.explore and args.uniform_set:
        # explore some samples randomly 
        exclude_probs = [item for item in prev_probs if item[:3] not in sel_imgs]
        explore_cnt = int(0.15 * len(exclude_probs))
        explore_imgs = random.sample(exclude_probs, explore_cnt)
        explore_imgs = [item[:3] for item in explore_imgs]
        sel_imgs.extend(explore_imgs)
        logger.info(f"explore {explore_cnt} of images")

    _ = evaluate(model, args, classification_head_new, Dict_cur_guidance, logger=logger, progress_guid=progress_guid,
                progress_sample=progress_sample, eval_imgs=sel_imgs)
    str_progress = dict()
    res_progress = dict()
    saved_diff = dict()
    saved_probs = None

    if progress_guid:
        if args.progress_metric == 'Acc':
            keywords = 'Accuracy'
        elif args.progress_metric == 'Prob':
            keywords = 'Values'
        else:
            keywords = 'F1'
        logger.info(f"Computing progress based on metric {keywords}")

        dict_guid_prog = {}
        for key, value in Dict_cur_guidance.items():
            if 'Number' in key:
                continue
            if keywords not in key:
                continue

            if key not in last_perform:
                if isinstance(value, float):
                    last_perform[key] = 0
                else:
                    last_perform[key] = copy.deepcopy(value)
                    last_perform[key][0] = np.zeros_like(last_perform[key][0])

            list_img_id = None
            list_img_emb = None
            if args.progress_metric == 'Prob':
                list_img_id = copy.deepcopy(value[1])
                list_img_emb = copy.deepcopy(value[2])
                value = value[0]

            # guidance value
            list_replace = ['Strength', 'Guidance', ' Accuracy', ' F1', ' Values']
            guidance_i = copy.deepcopy(key)
            for replace_word in list_replace:
                guidance_i = guidance_i.replace(replace_word, '')
            guidance_i = int(guidance_i)

            # compute moving average of progress based on history
            # moving average the progress here
            # adding current
            weighted_hist_prog = None
            if args.ma_progress and progress_ma is not None:
                if guidance_i in progress_ma:
                    weighted_hist_prog = progress_ma[guidance_i][-1]
                else:
                    progress_ma[guidance_i] = []

            if args.progress_metric != 'Prob':
                cur_progress = value - last_perform[key]
                if weighted_hist_prog is not None:
                    cur_progress = 0.8 * cur_progress + 0.2 * weighted_hist_prog

                str_progress[f"Guidance {guidance_i}"] = rnd_prog(cur_progress)  # for logging
                res_progress[guidance_i] = cur_progress  # for guidance ranking
                if print_log:
                    logger.info(f"Guidance {guidance_i} diff: {cur_progress}")

            else:
                # progress as relative increase of prob in each image
                value_arr = np.array(value)
                last_arr = np.array(last_perform[key][:2])[0, :]
                cur_progress = value_arr - last_arr
                saved_diff[guidance_i] = [value_arr.copy(), last_arr.copy(), list_img_id, list_img_emb]  # saved for
                # analysis

                if weighted_hist_prog is not None:
                    cur_progress = 0.9 * cur_progress + 0.1 * weighted_hist_prog

                # remove outliers
                cur_progress = remove_outliers(cur_progress)
                dict_guid_prog[guidance_i] = cur_progress

                # use 75% quantile as criteria
                thres_diff = np.percentile(cur_progress, 75)

                # relative_diff = cur_progress / value_arr
                mean_diff = np.mean(cur_progress)
                std_diff = np.std(cur_progress)

                str_progress[f"Guidance {guidance_i}"] = rnd_prog(mean_diff)  # for logging
                # res_progress[guidance_i] = np.max(cur_progress) - np.min(cur_progress)  # for guidance ranking
                # res_progress[guidance_i] = std_diff  # for guidance ranking
                res_progress[guidance_i] = mean_diff  # for guidance ranking
                if print_log:
                    logger.info(
                        f"Guidance {guidance_i}, 75%: {rnd_prog(thres_diff)}, mean: {rnd_prog(mean_diff)}, std: {rnd_prog(std_diff)}")

            if args.ma_progress and progress_ma is not None:
                # adding current eval to MA list
                progress_ma[guidance_i].append(cur_progress)

        # # select the guid with highest confidence
        # res_progress = find_best_progress(dict_guid_prog)

        # pdb.set_trace()
        last_perform = copy.deepcopy(Dict_cur_guidance)
        list_sample_prob = []

    elif progress_sample:
        logger.info(f"Finding best samples")
        # find samples with largest progress?
        # first evaluate each samples' progress
        # {img_id: [[label, guid, seed, prob], [label, guid, seed, prob]]}
        Dict_sample_prob = Dict_cur_guidance['progress_res']

        if args.partial_update and sel_imgs is not None:
            # update the probability with the new prob
            list_sample_prob = []
            for img_pair in prev_probs:
                img_id, guid, seed, last_prob, prev_prob = img_pair
                if [img_id, guid, seed] in sel_imgs:
                    img_probs = Dict_sample_prob[img_id]
                    new_prob = [item[-1] for item in img_probs if item[1] == guid and item[2] == seed][0]
                    # change to new progress
                    list_sample_prob.append([img_id, guid, seed, new_prob, last_prob])
                else:
                    # progress unchanged
                    list_sample_prob.append([img_id, guid, seed, last_prob, prev_prob])

            logger.info(f"Updating probs, num before: {len(prev_probs)}, num after: {len(list_sample_prob)}")
            list_sample_prob = sorted(list_sample_prob, key=lambda x: (x[2], x[1], x[0]), reverse=False)

            list_last = [item[:-1] for item in list_sample_prob]
            list_prev_prob = [item[-1] for item in list_sample_prob]
            saved_diff['progress_res'] = [list_last,
                                          list_prev_prob]  # saved for analysis

        else:
            # list_sample_prob: [img_id, guid, prob]
            list_sample_prob = [[key, val[1], val[2], val[3]] for key, value in Dict_sample_prob.items() for val in
                                value]
            list_sample_prob = sorted(list_sample_prob, key=lambda x: (x[2], x[1], x[0]), reverse=False)

            if 'progress_res' not in last_perform:
                last_perform['progress_res'] = None

            else:
                assert len(list_sample_prob) == len(last_perform['progress_res']), "length of sample prob are different"

            saved_diff['progress_res'] = [copy.deepcopy(list_sample_prob),
                                          copy.deepcopy(last_perform['progress_res'])]  # saved for analysis

            # merge with last perform
            for idx, pair in enumerate(list_sample_prob):
                if last_perform['progress_res'] is not None:
                    last_prob = last_perform['progress_res'][idx][-1]
                else:
                    last_prob = 0
                pair.append(last_prob)

        # find top samples
        # top_n = 100000
        top_n = len(list_sample_prob) // 4
        list_sample_prob = sorted(list_sample_prob, key=lambda x: x[-2] - x[-1], reverse=True)
        top_samples = list_sample_prob[:top_n]

        if args.random_guid and progress_sample and sel_imgs is not None:
            # explore some samples randomly 
            explore_cnt = int(0.4 * len(list_sample_prob))
            explore_imgs = random.sample(list_sample_prob, explore_cnt)
            top_samples = explore_imgs
            logger.info(f"random select {explore_cnt} of images")
        # elif args.explore:
        #     # 2
        #     explore_cnt = int(0.15 * (len(list_sample_prob) - top_n))
        #     explore_samples = random.sample(list_sample_prob[top_n:], explore_cnt)
        #     top_samples.extend(explore_samples)
        #     logger.info(f"explore {explore_cnt} of images")

        str_progress = top_samples
        res_progress = top_samples
        last_perform['progress_res'] = saved_diff['progress_res'][0]

    return res_progress, str_progress, last_perform, saved_diff, list_sample_prob


def init_guidance_setting(args, logger, ):
    cur_guidance = None
    cur_guidance_id = 0
    len_data = None
    loop_times = 1
    list_guidance = None
    num_batch_ori = None
    ori_proportion = None

    if args.curriculum:
        df_ori = pd.read_csv(args.ft_data, delimiter='\t')

        len_data = len(df_ori)
        list_guidance = list(set(df_ori['guidance'].values.tolist()))
        # list_guidance = sorted(list_guidance, reverse=False)  # 0 --> 100
        list_guidance = sorted(list_guidance, reverse=True)  # 0 --> 100
        if args.curriculum_epoch is None:
            # start from guidance = 0
            cur_guidance_id = 0
            cur_guidance = list_guidance[cur_guidance_id]
        else:
            # using curriculum_epoch to decide the current guidance
            # finish viewing all guidance data during curriculum_epoch
            len_ori = len(df_ori[df_ori['guidance'] == 100])
            num_batch_ori = int(len_ori / args.batch_size)  # num of batch in non curriculum epoch (update iterations)
            # keep number of iteration during the entire training process the same
            if args.train_dataset == 'ImageNet':
                num_batch_ori = num_batch_ori // 5
            total_iteration = num_batch_ori * args.curriculum_epoch * args.batch_size

            # estimate the number of times loading for each guid
            len_all_guid = len(df_ori[df_ori['guidance'] != 100])
            if args.merge_ori:
                num_guid = len(list_guidance)
                loop_times = int(args.curriculum_epoch / num_guid)
            else:
                loop_times = math.ceil(total_iteration / len_all_guid)

            # start from guidance = 100
            # cur_guidance_id = len(list_guidance) - 1
            cur_guidance_id = 0
            cur_guidance = list_guidance[cur_guidance_id]

    elif args.baseline:
        # train baseline with img guidance = 100
        cur_guidance = 100
        list_guidance = [cur_guidance]
        cur_guidance_id = 0

    if args.guidance != -1:
        # for single guidance training
        df_ori = pd.read_csv(args.ft_data, delimiter='\t')
        df_ori = df_ori[df_ori['guidance'] == args.guidance]
        len_data = len(df_ori)

        if args.datalimit != -1:
            logger.info(f"Sample {args.datalimit} from original dataset")
            df_ori = df_ori.sample(n=min(len_data, args.datalimit), random_state=1)
            len_data = len(df_ori)

        list_guidance = [args.guidance, ]
        cur_guidance_id = 0
        cur_guidance = args.guidance

    if args.proportion:
        # mix guid=100 with other guidance data
        ori_proportion = 0.1

    return cur_guidance_id, cur_guidance, list_guidance, loop_times, len_data, num_batch_ori, ori_proportion


def flyp_loss(args, clip_encoder, classification_head, logger):
    model_path = ''

    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    clip_encoder.process_images = True
    print_every = 100
    clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1,
                            use_horovod=False)

    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(
        args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)

    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))

    model = model.cuda()
    classification_head = classification_head.cuda()

    ############################
    # load finetuned model here
    if args.cont_finetune:
        model_path = os.path.join("checkpoints_base/iwildcam/flyp_loss_ori_eval/_BS256_WD0.2_LR1e-05_run1",
                                  f'checkpoint_15.pt')

        # model_path = os.path.join("checkpoints/flyp_loss_v7152/_BS300_WD0.2_LR1e-05_run1",
        #                           f'checkpoint_1.pt')
        logger.info('Loading model ' + str(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)  
        # model.load_state_dict(checkpoint['model_state_dict'])

    ############################
    # Data initialization
    cur_str_times = 1
    start_epoch = -1
    load_ckpt = False

    dataset_class = getattr(datasets, args.train_dataset)
    logger.info(f"Training dataset {args.train_dataset}")

    model = torch.nn.DataParallel(model, device_ids=devices)
    # classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)

    # init wandb if not debug mode
    if not args.debug:
        wandb.init(project="sd_exprs", config=args, name=args.exp_name, group=args.wandb_group_name, tags=[args.wandb_tag, ])
        wandb.watch(model, log="gradients", log_freq=100)

    # classification_head.train()
    model.train()

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    
    init_data = init_guidance_setting(args, logger, )
    cur_guidance_id, cur_guidance, list_guidance, loop_times, len_data, num_batch_ori, ori_proportion = init_data

    cnt = 0
    step = 0
    stats = []
    last_perform = {}
    prev_probs = None
    list_img_guid = None
    ############################
    # Based on breakpoint to keep training
    if os.path.exists(args.save):
        ckpt_file = f"prevcheckpoint.pt"
        loading_file = os.path.join(args.save, ckpt_file)
        if os.path.exists(loading_file):
            load_ckpt = True
            logger.info(f"Loading existing checkpoint {ckpt_file} and keep training...")

            checkpoint = torch.load(loading_file)  
            start_epoch = checkpoint['epoch']
            step = checkpoint['step']
            num_batches = checkpoint['num_batches']
            list_img_guid = checkpoint['list_img_guid']
            prev_probs = checkpoint['prev_probs']
            stats = checkpoint['stats']
            last_perform = checkpoint['last_perform']
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Num batches is {num_batches}")
            # next step: start a new stage training with selected samples list_img_guid, the previous progress: prev_probs

    if not load_ckpt:
        ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance, cur_str_times=cur_str_times,
                                epoch=0, ori_proportion=ori_proportion, )
        ft_iterator = iter(ft_dataloader)
        num_batches = len(ft_dataloader)
        if args.guidance == -1 and args.curriculum:
            if args.curriculum_epoch is None:
                num_batches = int(len_data / args.batch_size) if len_data is not None else num_batches * len(list_guidance)
            else:
                num_batches = num_batch_ori
        logger.info(f"Num batches is {num_batches}")

    if args.scheduler in ('default', 'drestart'):
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, (args.epochs + 1) * num_batches, args.min_lr)
    elif args.scheduler in ('default_slower',):
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, (args.epochs + 1) * num_batches * 2, args.min_lr)
    elif args.scheduler in ('crestart',):
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_batches, T_mult=1, eta_min=0.01, last_epoch=-1)
    else:
        raise ValueError(f'invalid scheduler type {args.scheduler}!')

    total_iter = 0
    next_change_guid = False
    pre_guidance = None
    start_uniform = 0

    if args.train_dataset == 'ImageNet':
        stats = general_eval(model, args, [], 0, logger=logger, print_log=True, print_class=True,
                             log_dir=log_dir)

    if args.progress_sample:
        if not load_ckpt:
            # start of progress sample, not matter whether use uniform dataset or not, need to run on a small set to obtain progress
            # ALA progress on sample, need to record the prob for all samples before training
            eval_res = progress_eval(model, args, last_perform, 0, logger, progress_sample=True, print_log=False)
            last_perform = eval_res[2]
            prev_probs = eval_res[4]
            ft_dataloader = load_data(logger, args, clip_encoder, epoch=0, uniform_guid=True)
            next_change_guid = True
        else:
            # start next training stage based on saved candidate images
            ft_dataloader = load_data(logger, args, clip_encoder, epoch=start_epoch, list_imgs=list_img_guid,)
        ft_iterator = iter(ft_dataloader)

    elif args.progress_guid and args.uniform_set:
        start_uniform = total_iter
        # start with guid found on uniformly distributed dataset
        eval_res = progress_eval(model, args, last_perform, 0, logger, progress_guid=True, print_log=False)
        last_perform = eval_res[2]

        ft_dataloader = load_data(logger, args, clip_encoder, epoch=0, uniform_guid=True)
        next_change_guid = True
        ft_iterator = iter(ft_dataloader)

    # record the progress history (prob diff)
    # when compute the current progress, progress = 0.8 * current + 0.2 * previous progress
    progress_ma = dict()
    adjust_epoch = False
    for epoch in trange(start_epoch + 1, args.epochs):
        # If set curriculum epochs
        if args.curriculum_epoch is not None and epoch >= args.curriculum_epoch:
            if args.scheduler == 'drestart':
                logger.info('Restart scheduler')
                scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                                      (args.epochs - start_epoch - args.curriculum_epoch) * num_batches, args.min_lr)

            if cur_guidance != 100:
                logger.info('Restart dataloader')
                cur_guidance = 100

                ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance, epoch=epoch, )
                ft_iterator = iter(ft_dataloader)
                num_batches = len(ft_dataloader)

        logger.info(f"Epoch : {epoch}")
        epoch_stats = {}
        epoch_stats['Epoch'] = epoch
        epoch_stats['epoch'] = epoch

        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()
        # classification_head.train()

        # list_loss_pairs = []
        for i in trange(num_batches):
            if args.test:
                logger.info(f"Skipping training process")
                break

            step += 1
            optimizer.zero_grad()
            if load_ckpt and step >= num_batches * args.curriculum_epoch and not adjust_epoch:
                # adjust the gap of steps of two experiments
                adjust_epoch = True
                break

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                ori_proportion = None
                uniform_set = False  # run on uniform set right not
                skip_loading = False
                if not args.curriculum or (args.curriculum_epoch is not None and epoch > args.curriculum_epoch):
                    # train on baseline without curriculum strategy / curriculum period ends
                    skip_loading = True
                elif (not args.progress_guid) and (not args.progress_sample):
                    # do not select next guid based on progress
                    if args.uniform_set and not next_change_guid:
                        cur_guidance = None
                        uniform_set = True
                        next_change_guid = True
                        logger.info(f"Running on uniform set")
                    else:
                        next_change_guid = False
                        # sequentially use guidance
                        if pre_guidance is not None:
                            cur_guidance_id = list_guidance.index(pre_guidance)
                            logger.info(f"changing curriculum .. cur_guidance_id={cur_guidance_id}")

                    if args.random_guid:
                        # not running progress eval
                        cur_guidance = random.choice(list_guidance)
                        logger.info(f"randomly select guid {cur_guidance}")
                        cur_guidance_id = list_guidance.index(cur_guidance)
                        cur_str_times = 0
                    elif args.curriculum_epoch is None:
                        # sequentially use guidance
                        guid_res = seq_curri_guid(list_guidance, cur_guidance_id=cur_guidance_id, ctype='no_curri')
                        cur_guidance_id, cur_guidance = guid_res
                    else:
                        guid_res = seq_curri_guid(list_guidance, cur_guidance_id=cur_guidance_id,
                                                  cur_str_times=cur_str_times, ctype='in_curri', loop_times=loop_times)
                        cur_guidance_id, cur_guidance, cur_str_times = guid_res
                        logger.info(f"new guid={cur_guidance}, cur_guidance_id={cur_guidance_id}")

                        # find the largest guidance based on progress
                        # eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_guid=True,
                        #                          progress_ma=progress_ma)
                        # res_progress, _, last_perform, saved_diff, _ = eval_res
                        # with open(f"{log_dir}/progress{cnt}.pkl", 'wb') as f:
                        #     pickle.dump(saved_diff, f)
                        # cnt += 1

                    # cur_guidance = 100  # cur_guidance_id = list_guidance.index(cur_guidance)
                elif args.progress_guid:
                    # select next guid based on progress
                    if args.uniform_set and not next_change_guid:
                        # not training progress eval to find the best guid
                        # run training on uniformly distributed dataset first
                        # evaluate the improvement on this uniformly distributed dataset
                        # use the largest improvement as the next guid
                        logger.info(f"Running on uniform set")
                        cur_guidance = None
                        uniform_set = True
                        next_change_guid = True
                        start_uniform = total_iter

                        # # record beginning progress prob  # eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_guid=True,  #                          print_log=False, )  # last_perform = eval_res[2]

                        # # eval performance on ood dataset  # _ = general_eval(model, args, stats, epoch, logger=logger, wandb_comment='Change ')

                    else:
                        next_change_guid = False

                        if args.random_guid:
                            # not running progress eval
                            cur_guidance = random.choice(list_guidance)
                            logger.info(f"randomly select guid {cur_guidance}")
                            cur_guidance_id = list_guidance.index(cur_guidance)
                            cur_str_times = 0
                        else:
                            # find the largest guidance based on progress
                            eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_guid=True,
                                                     progress_ma=progress_ma)
                            res_progress, _, last_perform, saved_diff, _ = eval_res
                            with open(f"{log_dir}/progress{cnt}.pkl", 'wb') as f:
                                pickle.dump(saved_diff, f)
                            cnt += 1

                            list_progress = [(guid, prog) for guid, prog in res_progress.items()]
                            list_progress = sorted(list_progress, key=lambda x: x[-1], reverse=True)
                            largest_guid = list_progress[0]

                            if args.explore:
                                tau_thres, next_guid = explore_guid(args, epoch, logger, largest_guid, list_progress)
                            else:
                                next_guid = largest_guid
                                tau_thres = 0
                                logger.info(f"Select largest guid = {next_guid[0]}")

                            if not args.debug:
                                wandb.log({"Epoch": epoch, "tau": tau_thres, })

                            cur_guidance = next_guid[0]
                            cur_guidance_id = list_guidance.index(cur_guidance)
                            cur_str_times = 0

                            # # eval performance on ood dataset  # _ = general_eval(model, args, stats, epoch, logger=logger, wandb_comment='Change ')

                    if args.proportion:
                        ori_proportion = 1 - 1 / args.curriculum_epoch * epoch

                elif args.progress_sample:
                    # select samples to train based on progress
                    if args.uniform_set and not next_change_guid:
                        # not training progress eval to find the best guid
                        # run training on uniformly distributed dataset first
                        # evaluate the improvement on this uniformly distributed dataset
                        # use the largest improvement as the next guid
                        cur_guidance = None
                        uniform_set = True
                        next_change_guid = True
                        # record beginning progress prob
                        eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_sample=True,
                                                 print_log=False, prev_probs=prev_probs, sel_imgs=list_img_guid, unif_begin=True)
                        last_perform = eval_res[2]
                        prev_probs = eval_res[4]
                        logger.info(f"Running on uniform set")  
                        # _ = general_eval(model, args, stats, epoch, logger=logger, wandb_comment='Change ')

                    else:
                        # finish a stage of training or finish the uniform training
                        next_change_guid = False
                        # find the largest guidance based on progress
                        eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_sample=True,
                                                 progress_ma=progress_ma, prev_probs=prev_probs, sel_imgs=list_img_guid)
                        res_progress, _, last_perform, saved_diff, prev_probs = eval_res
                        with open(f"{log_dir}/progress{cnt}.pkl", 'wb') as f:
                            pickle.dump((saved_diff, list_img_guid), f)
                        cnt += 1

                        # find samples with largest progress
                        list_img_guid = [item[:3] for item in res_progress]

                        # eval performance on ood dataset  # _ = general_eval(model, args, stats, epoch, logger=logger, wandb_comment='Change ')
                        os.makedirs(args.save, exist_ok=True)
                        model_path = os.path.join(args.save, f'prevcheckpoint.pt')
                        torch.save({'epoch': epoch, 'step': step, 'num_batches': num_batches, 'list_img_guid': list_img_guid, 'prev_probs': prev_probs, 'stats': stats, 'last_perform': last_perform, 
                                    'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), }, model_path)
                        logger.info('Saving model to' + str(model_path))

                if not skip_loading:
                    if not args.progress_sample or uniform_set:
                        ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance,
                                                  cur_str_times=cur_str_times, epoch=epoch, uniform_guid=uniform_set,
                                                  ori_proportion=ori_proportion, include_neg=args.include_neg)
                    else:
                        # select training samples
                        ft_dataloader = load_data(logger, args, clip_encoder, epoch=epoch, list_imgs=list_img_guid,
                                                  include_neg=args.include_neg)

                ft_iterator = iter(ft_dataloader)
                ft_batch = next(ft_iterator)

            ft_image, ft_text, ft_imgid = ft_batch

            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()
            ft_image_features, ft_text_features, logit_scale2 = model(ft_image, ft_text)
            if len(logit_scale2.shape) >= 1:
                logit_scale2 = logit_scale2[0]
            ft_clip_loss_peritem = clip_loss_fn(ft_image_features, ft_text_features, logit_scale2)

            img_id_loss = list(zip(ft_imgid.detach().cpu().numpy(), ft_clip_loss_peritem.detach().cpu().numpy()))
            # list_loss_pairs.extend(img_id_loss)

            ft_clip_loss = torch.mean(ft_clip_loss_peritem)
            ft_clip_loss.backward()
            optimizer.step()

            if args.scheduler == 'crestart':
                scheduler.step(epoch)
            else:
                scheduler(step)

            id_flyp_loss_sum += ft_clip_loss.item()

            # Training logging
            if not args.debug:
                if args.scheduler in ('default', 'drestart', 'default_slower'):
                    lr = optimizer.param_groups[0]['lr']
                elif args.scheduler in ('crestart',):
                    lr = scheduler.get_lr()[0]
                else:
                    lr = args.lr

                wandb.log({"Epoch": epoch, "ID FLYP Loss": ft_clip_loss.item(), "Learning Rate": lr, })

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                            f"ID FLYP Loss: {ft_clip_loss.item():.4f}")

            if args.uniform_set and total_iter - start_uniform == 1:
                if args.progress_guid:
                    # start with guid found on uniformly distributed dataset
                    eval_res = progress_eval(model, args, last_perform, epoch, logger, progress_guid=True,
                                             print_log=False, )
                    last_perform = eval_res[2]

            total_iter += 1

        id_flyp_loss_avg = id_flyp_loss_sum / num_batches

        #############################################
        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            torch.save({'model_state_dict': model.module.state_dict(), }, model_path)
            logger.info('Saving model to' + str(model_path))

        #############################################
        # Save the prediction score for each image and prompt for confusion matrix
        if args.debug:
            for i in range(19, 20):
                model_path = f'checkpoints/flyp_loss_imgnet_base/_BS300_WD0.1_LR1e-05_run1/checkpoint_{i}.pt'
                logger.info(f"evaluation on {model_path} ...")

                # load model
                checkpoint = torch.load(model_path)
                model.module.load_state_dict(checkpoint['model_state_dict'])
                # model = model.cuda()
                # model = torch.nn.DataParallel(model, device_ids=devices)

                classification_head_new = generate_class_head(model, args, epoch)
                # evaluate on training set
                eval_results = evaluate(model, args, classification_head_new, epoch_stats, progress_guid=True, logger=logger)
                
                dict_best_guid = epoch_stats['dict_img_guid']
                print(f"{len(dict_best_guid)}")

                # save guidance_score:
                with open(log_dir + f'/model_{i}_train.pkl', 'wb') as f:
                    pickle.dump(dict_best_guid, f)

            # continue
            exit(0)

        #############################################
        # Evaluate
        logger.info(f"Formal evaluation ...")
        stats = general_eval(model, args, stats, epoch, logger=logger, print_log=True, print_class=True,
                             log_dir=log_dir)

    if args.save is not None:
        return model_path

    os.system('wandb sync')
