from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
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
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets
import pickle
import random
import pdb
import math
import wandb
import numpy as np


def seq_curri_guid(list_strength: List, cur_strength_id=None, cur_str_times=None, ctype='out_curri', loop_times=1):
    # sequentially use strength 
    if ctype == 'no_curri':
        # iteratively loop over all guidance
        cur_strength_id += 1
        if cur_strength_id >= len(list_strength):
            cur_strength_id = 0
        cur_strength = list_strength[cur_strength_id]
        return cur_strength_id, cur_strength

    elif ctype == 'in_curri':
        # have fixed curriculum length
        if cur_str_times < loop_times:
            # cur_strength_id unchanged 
            cur_str_times += 1
        else:
            cur_str_times = 1
            cur_strength_id += 1

            if cur_strength_id >= len(list_strength):
                cur_strength_id = len(list_strength) - 1

        cur_strength = list_strength[cur_strength_id]
        return cur_strength_id, cur_strength, cur_str_times

    elif ctype == 'out_curri':
        cur_strength = 0
        cur_str_times = 1
        return cur_strength_id, cur_strength, cur_str_times
    else:
        raise ValueError(f"invalid ctype {ctype}")


def load_data(logger, args, clip_encoder, cur_strength=None, cur_str_times=1, list_classes=None, ):

    if cur_strength is not None:
        logger.info(f"loading image guidance = {100-cur_strength}, loop times {cur_str_times}")
        if not args.debug:
            wandb.log({"Image Guidance": 100-cur_strength})

    # load dataloader
    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0, strength=cur_strength, list_selection=list_classes)
    assert len(img_text_data), 'At least one train or eval dataset must be specified.'

    ft_dataloader = img_text_data['train_ft'].dataloader
    return ft_dataloader


def generate_class_head(model, args, epoch):
    # get classification head embedding
    args.current_epoch = epoch
    classification_head_new = get_zeroshot_classifier(
        args, model.module.model)
    classification_head_new = classification_head_new.cuda()
    return classification_head_new


def progress_eval(model, args, last_strength, epoch, logger, progress_ma=None):
    
    classification_head_new = generate_class_head(model, args, epoch)
    Dict_cur_strength = {}
    last_results = evaluate(model, args, classification_head_new,
                                    Dict_cur_strength, logger, progress_eval=True)
    str_progress = dict()
    res_progress = dict()
    cur_stats = dict()

    for key, value in Dict_cur_strength.items():
        if 'Number' in key: 
            continue
        if key not in last_strength:
            last_strength[key] = 0

        guidance_i = int(key.replace('Strength ', '').replace(' Accuracy', ''))

        # compute moving average of progress
        if args.ma_progress and progress_ma is not None:
            # adding current eval to ma list
            progress_ma[guidance_i].append(value)
            # compute for average here
            value = np.mean(np.array(progress_ma[guidance_i]))

        str_progress[f"Guidance {100-guidance_i}"] = np.round(value - last_strength[key], 6)
        res_progress[guidance_i] = value - last_strength[key]
        cur_stats[guidance_i] = value

    last_strength = copy.deepcopy(Dict_cur_strength)
    return res_progress, str_progress, last_strength, cur_stats


def flyp_loss_progress(args, clip_encoder, classification_head, logger):
    def train_model_basedon_guid(guid, cur_step):
        cur_str_times = 1
        id_flyp_loss_sum = 0
        ft_dataloader = load_data(logger, args, clip_encoder, cur_strength=guid)
        ft_iterator = iter(ft_dataloader)
        num_batches = len(ft_dataloader)

        for i in trange(num_batches):
            # step = i + epoch * num_batches
            optimizer.zero_grad()
            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                break

            ft_image, ft_text = ft_batch
            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

            ft_image_features, ft_text_features, logit_scale2 = model(
                ft_image, ft_text)
            ft_clip_loss = clip_loss_fn(ft_image_features,
                                        ft_text_features,
                                        logit_scale2)

            ft_clip_loss.backward()
            optimizer.step()

            if args.scheduler == 'crestart':
                scheduler.step(epoch)
            else:
                scheduler(cur_step)

            cur_step += 1
            id_flyp_loss_sum += ft_clip_loss.item()
            wandb.log({"Epoch": epoch, "ID FLYP Loss": ft_clip_loss.item(), })
            if i % 100 == 0:
                percent_complete = 100 * i / num_batches
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID FLYP Loss: {ft_clip_loss.item():.4f}")

                res_progress, _, _, _ = progress_eval(model, args, last_strength, epoch=epoch, logger=logger)
                res_progress["Epoch"] = epoch
                res_progress["Trained_guid"] = guid
                # wandb.log(res_progress)
                
        id_flyp_loss_avg = id_flyp_loss_sum / num_batches
        return cur_step, id_flyp_loss_avg

    def loading_model(model_path):
        logger.info('Loading model ' + str(model_path))
        checkpoint = torch.load(last_model_path)
        new_state_dict = dict()
        for key, value in checkpoint['model_state_dict'].items():
            new_key = 'module.' + key
            new_state_dict[new_key] = value

        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_strength = checkpoint['last_progress']
        step = checkpoint['step']
        return last_strength, step
         

    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100

    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
        args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)

    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))

    ############################
    # load finetuned model here
    if args.cont_finetune:
        model_path = os.path.join("checkpoints_base/iwildcam/flyp_loss_ori_eval/_BS256_WD0.2_LR1e-05_run1", f'checkpoint_15.pt')
        logger.info('Loading model ' + str(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)

    ############################
    # Data initialization
    list_classes = None
    if args.cont_finetune:
        df_acc = pd.read_csv('expt_logs/iwildcam/flyp_loss_ori_eval/_BS256_WD0.2_LR1e-05_run1/class_stats15.tsv', delimiter='\t')
        df_filter = df_acc[(df_acc['IWildCamOOD'] <= 0.5) & (df_acc['IWildCamOOD Count'] >= 50)]
        # df_filter = df_acc[(df_acc['IWildCamOOD'] <= 0.5) & (df_acc['IWildCamOOD Count'] <= 50)]
        list_classes = df_filter['Unnamed: 0'].values.tolist()
        list_classes = [int(item.replace('Class ', '')) for item in list_classes]
        if 0 not in list_classes:
            list_classes.append(0)
        logger.info(f"Only continuing finetune ckpt based on {len(list_classes)} classes: {list_classes}")
    
    cur_strength = None
    cur_strength_id = 0
    len_data = None
    cur_str_times = 1
    start_epoch = 0

    dataset_class = getattr(datasets, args.train_dataset)
    logger.info(f"Training dataset {args.train_dataset}")

    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)

    # init wandb if not debug mode
    if not args.debug:
        wandb.init(project="sd_exprs", config=args, name=args.exp_name, group=args.wandb_group_name)
        wandb.watch(model, log="gradients", log_freq=100)

    if args.curriculum:
        df_ori = pd.read_csv(args.ft_data, delimiter='\t')
        if args.cont_finetune:
            df_ori = df_ori[df_ori['label'].isin(list_classes)]

        len_data = len(df_ori)
        list_strength = list(set(df_ori['strength'].values.tolist()))
        list_strength = sorted(list_strength, reverse=True)
        if args.curriculum_epoch is None:
            if cur_strength is None:
                cur_strength_id = 0
                cur_strength = list_strength[cur_strength_id]
        else:
            # using curriculum_epoch to decide the current strength
            # finish viewing all strength data during curriculum_epoch
            len_ori = len(df_ori[df_ori['strength']==0])
            num_batch_ori = int(len_ori/args.batch_size)  # num of batch in non curriculum epoch (update iterations)
            len_all_stre = len(df_ori[df_ori['strength']!=0])
            total_viewing = num_batch_ori * args.curriculum_epoch * args.batch_size
            loop_times = math.ceil(total_viewing / len_all_stre)

            # start from strength = 0 / img guidance = 100
            if cur_strength is None:
                cur_strength_id = 0
                cur_strength = list_strength[cur_strength_id]
    
    elif args.baseline:
        # train baseline
        cur_strength_id = 0
        cur_strength = 0

    # run progress eval after single strength training
    if args.strength != -1:
        df_ori = pd.read_csv(args.ft_data, delimiter='\t')
        df_ori = df_ori[df_ori['strength'] == args.strength]
        len_data = len(df_ori)

        list_strength = [args.strength, ]
        cur_strength_id = 0
        cur_strength = args.strength

    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False,
                            gather_with_grad=False,
                            cache_labels=True,
                            rank=0,
                            world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler in ('default', 'drestart'):
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                            (args.epochs - start_epoch) * 1000, args.min_lr)
    else:
        raise ValueError(f'invalid scheduler type {args.scheduler}!')

    stats = []
    last_strength = {}
    epoch = -1

    ## progress validation expr
    # 0. save the current start model
    os.makedirs(args.save, exist_ok=True)
    model_path = os.path.join(args.save, f'cur_point{epoch}_guidpath00.pt')
    torch.save({'model_state_dict': model.module.state_dict(), 'last_progress': last_strength, 'optimizer_state_dict': optimizer.state_dict(), 'step': 0}, model_path)
    logger.info('Saving model to' + str(model_path))

    stats = []
    list_last = [model_path, ]  # last trained top-5 best model
    step = 0
    while epoch <= 1:
        epoch += 1
        list_model_performance = []
        for last_model_path in list_last:
            # 1. load the last saved ckpt
            # load the the last saved model and train the model with guidance data
            last_model_name = last_model_path.split('/')[-1].replace('.pt', '')
            last_guid_path_str = last_model_name.split('guidpath')[1]
            last_guid_path = last_guid_path_str.split('=')
            last_guid_path = list(map(int, last_guid_path))  # transform to int
            
            last_strength, step = loading_model(last_model_path)

            # # 2. eval progress of different guidance based on this last model
            res_progress, str_progress, last_strength, _ = progress_eval(model, args, last_strength, epoch=-1, logger=logger)
            list_progress = [(guid, value) for guid, value in res_progress.items()]
            list_progress = sorted(list_progress, key=lambda x: x[-1], reverse=True)

            str_progress['epoch'] = epoch
            df_str_progress = pd.DataFrame.from_dict(str_progress, orient='index', )
            df_str_progress.to_csv(log_dir + f'/progress{epoch}_before_guidpath{last_guid_path_str}.tsv', sep='\t')

            # 3. train the model with different guidance data (all start from the same ckpt)
            for guid_pair in list_progress:
                epoch_stats = {}
                epoch_stats['last_model_name'] = last_model_name
                epoch_stats['epoch'] = epoch

                # pdb.set_trace()
                # load the the last saved model and train the model with guidance data
                last_strength, step = loading_model(last_model_path)
                # pdb.set_trace()

                logger.info(f'start step: {str(step)}')

                # load guidance data
                guid_int = guid_pair[0]
                progress = guid_pair[1]
                
                cur_guid_path = copy.deepcopy(last_guid_path)
                cur_guid_path.append(guid_int)
                cur_guid_path_str = "=".join(list(map(str, cur_guid_path)))

                # train model
                # pdb.set_trace()
                step, id_flyp_loss_avg = train_model_basedon_guid(guid_int, step)
                # pdb.set_trace()
                logger.info(f'end step: {step}')
                
                # 4. eval the trained model on the guidance dataset / wildcamp dataset
                # guidance dataset
                res_progress, str_progress, last_strength, _ = progress_eval(model, args, last_strength, epoch=epoch, logger=logger)
                res_progress["Epoch"] = epoch
                res_progress["Trained_guid"] = guid_int
                # wandb.log(res_progress)

                str_progress['epoch'] = epoch
                df_str_progress = pd.DataFrame.from_dict(str_progress, orient='index', )
                df_str_progress.to_csv(log_dir + f'/progress{epoch}_after_guidpath{cur_guid_path_str}.tsv', sep='\t')

                # wildcamp dataset
                classification_head_new = generate_class_head(model, args, epoch)
                eval_results = evaluate(model, args, classification_head_new,
                                        epoch_stats, logger)

                ood_acc = 0
                num_datasets = 0
                for k, v in epoch_stats.items():
                    if 'Accuracy' in k and 'Class' not in k:
                        if k == 'ImageNet Accuracy':
                            #ignore the ID acc term
                            continue
                        ood_acc += v
                        num_datasets += 1
                if num_datasets != 0:
                    ood_acc = ood_acc / num_datasets
                else:
                    ood_acc = 0
                
                epoch_stats['Trained_guid'] = guid_int
                epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
                logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
                logger.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
                epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
                epoch_stats = {key: values for key, values in epoch_stats.items() if ' Class' not in key}
                
                list_model_performance.append([epoch, guid_int, last_strength, cur_guid_path_str, step, epoch_stats['IWildCamOODF1-macro_all'], model.module.state_dict(), ])

                stats.append(epoch_stats)
                stats_df = pd.DataFrame(stats)
                stats_df.to_csv(log_dir + f'/stats{epoch}_after_guidpath{cur_guid_path_str}.tsv', sep='\t')

        # delete previous top-5 model:
        for model_path in list_last:
            os.system(f"rm {model_path}")

        list_model_performance = sorted(list_model_performance, key=lambda x: x[-2], reverse=True)[:5]
        list_last = []
        for model_para in list_model_performance:
            model_path = os.path.join(args.save, f'cur_point{epoch}_guidpath{model_para[3]}.pt')
            torch.save({'model_state_dict': model_para[-1], 'last_progress': model_para[2], 'optimizer_state_dict': optimizer.state_dict(), 'step': model_para[4]}, model_path)
            logger.info('Saving model to' + str(model_path))
            list_last.append(model_path)
        logger.info(f"Saved top-5 model: {list_last}")

    os.system('wandb sync')
    exit(0)

    os.system('wandb sync')
    