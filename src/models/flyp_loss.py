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


def seq_curri_guid(list_guidance: List,
                   cur_guidance_id=None,
                   cur_str_times=None,
                   ctype='out_curri',
                   loop_times=1):
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


def load_data(logger,
              args,
              clip_encoder,
              cur_guidance=None,
              cur_str_times=1,
              list_classes=None,
              epoch=0,
              ori_proportion=None):
    if cur_guidance is not None:
        logger.info(f"loading image guidance = {cur_guidance}, loop times {cur_str_times}")
        if not args.debug:
            wandb.log({"Epoch": epoch, "Image Guidance": cur_guidance})
            if ori_proportion is not None:
                wandb.log({"Epoch": epoch, "Porportion of 100": ori_proportion})

    # load dataloader
    img_text_data = get_data(args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess), epoch=0,
        guidance=cur_guidance, list_selection=list_classes, ori_proportion=ori_proportion)
    assert len(img_text_data), 'At least one train or eval dataset must be specified.'

    ft_dataloader = img_text_data['train_ft'].dataloader
    if not args.debug:
        wandb.log({"Epoch": epoch, "Cur Dataloader Batch": len(ft_dataloader)})
    return ft_dataloader


def generate_class_head(model,
                        args,
                        epoch):
    # get classification head embedding
    args.current_epoch = epoch
    classification_head_new = get_zeroshot_classifier(args, model.module.model)
    classification_head_new = classification_head_new.cuda()
    return classification_head_new


def progress_eval(model,
                  args,
                  last_perform,
                  epoch,
                  logger,
                  progress_ma=None):
    classification_head_new = generate_class_head(model, args, epoch)
    Dict_cur_guidance = {}
    last_results = evaluate(model, args, classification_head_new, Dict_cur_guidance, logger, progress_eval=True)
    str_progress = dict()
    res_progress = dict()
    cur_stats = dict()

    for key, value in Dict_cur_guidance.items():
        if 'Number' in key:
            continue
        if 'F1' not in key:
            continue
        if key not in last_perform:
            last_perform[key] = 0

        guidance_i = int(
            key.replace('Strength ', '').replace('Guidance ', '').replace(' Accuracy', '').replace(' F1', ''))

        # compute moving average of progress
        if args.ma_progress and progress_ma is not None:
            # adding current eval to ma list
            progress_ma[guidance_i].append(value)
            # compute for average here
            value = np.mean(np.array(progress_ma[guidance_i]))

        str_progress[f"Guidance {guidance_i}"] = np.round(value - last_perform[key], 6)
        res_progress[guidance_i] = value - last_perform[key]
        cur_stats[guidance_i] = value

    last_perform = copy.deepcopy(Dict_cur_guidance)
    return res_progress, str_progress, last_perform, cur_stats


def progress_eval_train(model,
                        args,
                        epoch,
                        logger,
                        progress_ma=None):
    """
    Evaluate the best guidance on training dataset for each image

    :param model:
    :param args:
    :param epoch:
    :param logger:
    :param progress_ma:
    :return:
    """
    classification_head_new = generate_class_head(model, args, epoch)

    dict_guid_prob = {}
    _ = evaluate(model, args, classification_head_new, dict_guid_prob, logger, progress_train=True)

    dict_best_guid = dict()
    for img_id, list_guid_prob in dict_guid_prob.items():
        # compute moving average of progress
        if args.ma_progress and progress_ma is not None:
            # adding current eval to ma list
            progress_ma[img_id].extend(list_guid_prob)
            # compute for average here
            new_list_guid = progress_ma[img_id]
            all_guid = set([item[0] for item in new_list_guid])
            list_guid_prob_new = []
            for guid_int in all_guid:
                guid_probs = [item[1] for item in new_list_guid if item[0] == guid_int]
                value = np.mean(np.array(guid_probs))
                list_guid_prob_new.append([guid_int, value])

            list_guid_prob = list_guid_prob_new

        # find best guid for each image
        list_guid_prob = sorted(list_guid_prob, key=lambda x: x[-1], reverse=True)
        best_guid = list_guid_prob[0][0]
        dict_best_guid[img_id] = best_guid

    return dict_best_guid


def init_guidance_setting(args,
                          list_classes=None):
    cur_guidance = None
    cur_guidance_id = 0
    len_data = None
    cur_str_times = 1
    list_guidance = None

    if args.curriculum:
        df_ori = pd.read_csv(args.ft_data, delimiter='\t')
        if args.cont_finetune:
            df_ori = df_ori[df_ori['label'].isin(list_classes)]

        len_data = len(df_ori)
        list_guidance = list(set(df_ori['guidance'].values.tolist()))
        list_guidance = sorted(list_guidance, reverse=False)  # 0 --> 100
        if args.curriculum_epoch is None:
            if cur_guidance is None:
                # start from guidance = 0
                cur_guidance_id = 0
                cur_guidance = list_guidance[cur_guidance_id]
        else:
            # using curriculum_epoch to decide the current guidance
            # finish viewing all guidance data during curriculum_epoch
            len_ori = len(df_ori[df_ori['guidance'] == 100])
            num_batch_ori = int(len_ori / args.batch_size)  # num of batch in non curriculum epoch (update iterations)
            len_all_guid = len(df_ori[df_ori['guidance'] != 100])
            total_viewing = num_batch_ori * args.curriculum_epoch * args.batch_size
            loop_times = math.ceil(total_viewing / len_all_guid)

            # start from guidance = 0
            if cur_guidance is None:
                cur_guidance_id = 0
                cur_guidance = list_guidance[cur_guidance_id]

    elif args.baseline:
        # train baseline with img guidance = 100
        cur_guidance = 100
        list_guidance = [cur_guidance]
        cur_guidance_id = 0

    if args.guidance != -1:
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

    return cur_guidance_id, cur_guidance, list_guidance, loop_times, len_data, num_batch_ori


def flyp_loss(args,
              clip_encoder,
              classification_head,
              logger):
    model_path = ''

    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100

    log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(
        args.lr) + "_run" + str(args.run)
    os.makedirs(log_dir, exist_ok=True)

    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))

    ############################
    # load finetuned model here
    if args.cont_finetune:
        # model_path = os.path.join("checkpoints_base/iwildcam/flyp_loss_ori_eval/_BS256_WD0.2_LR1e-05_run1", f'checkpoint_15.pt')
        model_path = os.path.join("checkpoints/flyp_loss_curriculum_v1001/_BS256_WD0.2_LR1e-05_run1",
                                  f'checkpoint_19.pt')
        logger.info('Loading model ' + str(model_path))
        checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])

    ############################
    # Data initialization
    list_classes = None
    if args.cont_finetune:
        df_acc = pd.read_csv('expt_logs/iwildcam/flyp_loss_ori_eval/_BS256_WD0.2_LR1e-05_run1/class_stats15.tsv',
                             delimiter='\t')
        df_filter = df_acc[(df_acc['IWildCamOOD'] <= 0.5) & (df_acc['IWildCamOOD Count'] >= 50)]
        # df_filter = df_acc[(df_acc['IWildCamOOD'] <= 0.5) & (df_acc['IWildCamOOD Count'] <= 50)]
        list_classes = df_filter['Unnamed: 0'].values.tolist()
        list_classes = [int(item.replace('Class ', '')) for item in list_classes]
        if 0 not in list_classes:
            list_classes.append(0)
        logger.info(f"Only continuing finetune ckpt based on {len(list_classes)} classes: {list_classes}")

    cur_str_times = 1
    start_epoch = 0

    dataset_class = getattr(datasets, args.train_dataset)
    logger.info(f"Training dataset {args.train_dataset}")

    # dataset = dataset_class(preprocess_fn,
    #                         location=args.data_location,
    #                         batch_size=args.batch_size)

    # # ############################
    # # # Based on breakpoint to keep training
    # if os.path.exists(args.save):
    #     list_files = os.listdir(args.save)
    #     if len(list_files) > 0:
    #         list_ckpt = [int(item.replace('checkpoint_', '')) for item in list_files if 'checkpoint_' in item]
    #         list_ckpt = sorted(list_ckpt, reverse=True)
    #         ckpt_file = f"checkpoint_{list_ckpt[0]}"
    #         loading_file = os.path.join(args.save, ckpt_file)
    #         logger.info(f"Loading existing checkpoint {ckpt_file} and keep training...")

    #         checkpoint = torch.load(loading_file)  
    #         start_epoch = checkpoint['epoch']
    #         cur_guidance = checkpoint['cur_guidance']
    #         cur_str_times = checkpoint['cur_str_times']
    #         cur_guidance_id = checkpoint['cur_guidance_id']
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head, device_ids=devices)

    # init wandb if not debug mode
    if not args.debug:
        wandb.init(project="sd_exprs", config=args, name=args.exp_name, group=args.wandb_group_name)
        wandb.watch(model, log="gradients", log_freq=100)

    cur_guidance_id, cur_guidance, list_guidance, loop_times, len_data, num_batch_ori = init_guidance_setting(args,
                                                                                                              list_classes=list_classes)

    ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance, cur_str_times=cur_str_times,
                              list_classes=list_classes, epoch=0, ori_proportion=0.1)
    ft_iterator = iter(ft_dataloader)
    num_batches = len(ft_dataloader)

    if args.curriculum:
        if args.curriculum_epoch is None:
            num_batches = int(len_data / args.batch_size) if len_data is not None else num_batches * len(list_guidance)
        else:
            num_batches = num_batch_ori
    logger.info(f"Num batches is {num_batches}")

    classification_head.train()
    model.train()

    clip_loss_fn = ClipLoss(local_loss=False, gather_with_grad=False, cache_labels=True, rank=0, world_size=1,
                            use_horovod=False)

    clip_params = list(model.parameters())
    total_params = clip_params
    params = [p for p in total_params if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler in ('default', 'drestart'):
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, (args.epochs - start_epoch) * num_batches,
                              args.min_lr)
    elif args.scheduler in ('default_slower',):
        scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, (args.epochs - start_epoch) * num_batches * 2,
                              args.min_lr)
    elif args.scheduler in ('crestart',):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=num_batches, T_mult=1,
                                                                         eta_min=0.01, last_epoch=-1)
    else:
        raise ValueError(f'invalid scheduler type {args.scheduler}!')

    stats = []
    last_perform = {}
    for epoch in trange(start_epoch + 1, args.epochs):
        # If set curriculum epochs
        if args.curriculum_epoch is not None and epoch >= args.curriculum_epoch:
            if args.scheduler == 'drestart':
                logger.info('Restart scheduler')
                scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
                                      (args.epochs - start_epoch - args.curriculum_epoch) * num_batches, args.min_lr)

            if cur_guidance != 0:
                logger.info('Restart dataloader')
                cur_guidance = len(list_guidance) - 1
                cur_str_times = 1

                ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance,
                                          cur_str_times=cur_str_times, list_classes=list_classes, epoch=epoch,
                                          ori_proportion=None)
                ft_iterator = iter(ft_dataloader)
                num_batches = len(ft_dataloader)

        logger.info(f"Epoch : {epoch}")
        epoch_stats = {}
        epoch_stats['Epoch'] = epoch

        progress_ma = dict()

        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()
        classification_head.train()

        if not args.test:
            for i in trange(num_batches):
                step = i + epoch * num_batches
                optimizer.zero_grad()

                try:
                    ft_batch = next(ft_iterator)
                except StopIteration:
                    ori_proportion = None
                    if args.curriculum:
                        if epoch > args.curriculum_epoch:
                            cur_guidance = 100
                            cur_guidance_id = list_guidance.index(cur_guidance)
                            cur_str_times = 1
                        else:
                            if not args.progress:
                                # sequentially use guidance 
                                if args.curriculum_epoch is None:
                                    cur_guidance_id, cur_guidance = seq_curri_guid(list_guidance,
                                                                                   cur_guidance_id=cur_guidance_id,
                                                                                   ctype='no_curri')
                                else:
                                    cur_guidance_id, cur_guidance, cur_str_times = seq_curri_guid(list_guidance,
                                                                                                  cur_guidance_id=cur_guidance_id,
                                                                                                  cur_str_times=cur_str_times,
                                                                                                  ctype='in_curri',
                                                                                                  loop_times=loop_times)

                            else:
                                # select guidance based on progress
                                res_progress, _, last_perform, _ = progress_eval(model, args, last_perform, epoch,
                                                                                 logger)
                                list_progress = [(guid, prog) for guid, prog in res_progress.items()]
                                list_progress = sorted(list_progress, key=lambda x: x[-1], reverse=True)
                                largest_guid = list_progress[0]
                                if args.explore:
                                    # randomly select a guid with 15%, use the largest with 85%
                                    rand_prob = random.uniform(0, 1)
                                    if rand_prob <= 0.15:
                                        next_guid = random.choice(list_progress)
                                    else:
                                        next_guid = largest_guid
                                else:
                                    next_guid = largest_guid

                                cur_guidance = next_guid[0]
                                cur_guidance_id = list_guidance.index(cur_guidance)
                                cur_str_times = 0

                            if args.proportion:
                                ori_proportion = 1 / args.curriculum_epoch * epoch  # cur_str_times = loop_times

                        # ori_proportion
                        ft_dataloader = load_data(logger, args, clip_encoder, cur_guidance=cur_guidance,
                                                  cur_str_times=cur_str_times, list_classes=list_classes, epoch=epoch,
                                                  ori_proportion=ori_proportion)

                    ft_iterator = iter(ft_dataloader)
                    ft_batch = next(ft_iterator)

                ft_image, ft_text = ft_batch
                ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

                ft_image_features, ft_text_features, logit_scale2 = model(ft_image, ft_text)
                ft_clip_loss = clip_loss_fn(ft_image_features, ft_text_features, logit_scale2)

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

                if args.ma_progress and (num_batches - i) % 100 == 0:
                    logger.info(f"Running progress evaluation for moving average with i={i}")
                    # calculate progress multiple times
                    _, _, _, cur_stats = progress_eval(model, args, last_perform, epoch, logger)

                    for guid, value in cur_stats.items():
                        if guid not in progress_ma:
                            progress_ma[guid] = []
                        progress_ma[guid].append(value)

        id_flyp_loss_avg = id_flyp_loss_sum / num_batches

        #############################################
        # Saving model
        if args.save is not None and not args.ma_progress:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            torch.save({'epoch': epoch, 'cur_guidance': cur_guidance, 'cur_str_times': cur_str_times,
                'cur_guidance_id': cur_guidance_id, 'model_state_dict': model.module.state_dict(), }, model_path)
            # optimizer_path = os.path.join(args.save, f'optimizer_{epoch}.pt')
            # torch.save({'optimizer_state_dict': optimizer.state_dict(),}, optimizer_path)
            logger.info('Saving model to' + str(model_path))

        #############################################
        # Find the best guidance for each img for current model
        if args.progress_train:
            logger.info(f"Progress evaluation on training data ...")
            dict_best_guid = progress_eval_train(model=model, args=args, epoch=epoch, logger=logger,
                                                             progress_ma=progress_ma)
            dict_best_guid['Epoch'] = epoch

            # save progress_ma:
            with open(log_dir + f'/best_guid{epoch}.pkl', 'wb') as f:
                pickle.dump(dict_best_guid, f)

        #############################################
        # Evaluate progress on different group of cur_guidance for this epoch
        if args.progress_eval:
            logger.info(f"Progress evaluation ...")
            _, str_progress, last_perform, _ = progress_eval(model, args, last_perform, epoch, logger,
                                                             progress_ma=progress_ma)

            str_progress['Epoch'] = epoch
            df_str_progress = pd.DataFrame.from_dict(str_progress, orient='index', )
            df_str_progress.to_csv(log_dir + f'/progress{epoch}.tsv', sep='\t')

            # save progress_ma:
            with open(log_dir + f'/progress{epoch}.pkl', 'wb') as f:
                pickle.dump([last_perform, progress_ma], f)

            progress_ma = dict()

        #############################################
        # Evaluate
        logger.info(f"Formal evaluation ...")
        classification_head_new = generate_class_head(model, args, epoch)
        eval_results = evaluate(model, args, classification_head_new, epoch_stats, logger)

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

        ################################################
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
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logger.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
        epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
        epoch_stats = {key: values for key, values in epoch_stats.items() if ' Class' not in key}

        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

        ################################################
        # Evaluation logging
        if not args.debug:
            wandb.log(epoch_stats)

    if args.save is not None:
        return model_path

    os.system('wandb sync')
