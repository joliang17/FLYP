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

from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.models.eval import evaluate
from src.models.modeling import ClassificationHead, CLIPEncoder, ImageClassifier
from src.models.utils import cosine_lr, torch_load, LabelSmoothing, get_logits
from src.models.zeroshot import get_zeroshot_classifier
from src.datasets.laion import get_data
import src.datasets as datasets
import pdb
import math


def flyp_loss(args, clip_encoder, classification_head, logger):
    assert args.train_dataset is not None, "Please provide a training dataset."
    logger.info('Fine-tuning Using FLYP Loss')
    model = clip_encoder
    input_key = 'images'
    preprocess_fn = clip_encoder.train_preprocess
    image_enc = None
    clip_encoder.process_images = True
    print_every = 100

    dataset_class = getattr(datasets, args.train_dataset)
    print(f"Training dataset {args.train_dataset}")

    # dataset = dataset_class(preprocess_fn,
    #                         location=args.data_location,
    #                         batch_size=args.batch_size)

    cur_strength = None
    len_data = None
    cur_str_times = 1
    if args.curriculum:
        df_ori = pd.read_csv(args.ft_data, delimiter='\t')
        len_data = len(df_ori)
        list_strength = list(set(df_ori['strength'].values.tolist()))
        list_strength = sorted(list_strength, reverse=True)
        if args.curriculum_epoch is None:
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

            cur_strength_id = 0
            cur_strength = list_strength[cur_strength_id]

        print(f"loading strength = {cur_strength}")
        
    img_text_data = get_data(
        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
        epoch=0, strength=cur_strength)
    assert len(
        img_text_data), 'At least one train or eval dataset must be specified.'
    ft_dataloader = img_text_data['train_ft'].dataloader
    ft_iterator = iter(ft_dataloader)
    num_batches = len(ft_dataloader)
    if args.curriculum:
        if args.curriculum_epoch is None:
            num_batches = int(len_data/args.batch_size) if len_data is not None else num_batches * len(list_strength)
        else:
            num_batches = num_batch_ori

    print(f"Num batches is {num_batches}")

    model = model.cuda()
    classification_head = classification_head.cuda()
    devices = list(range(torch.cuda.device_count()))
    logger.info('Using devices' + str(devices))

    ############################
    # TODO: load finetuned model here
    model_path = os.path.join("checkpoints_base/iwildcam/flyp_loss_ori_eval/_BS256_WD0.2_LR1e-05_run1", f'checkpoint_15.pt')
    logger.info('Loading model ' + str(model_path))
    model.load_state_dict(torch.load(model_path))
    pdb.set_trace()

    model = torch.nn.DataParallel(model, device_ids=devices)
    classification_head = torch.nn.DataParallel(classification_head,
                                                device_ids=devices)
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1, eta_min=0.01, last_epoch=-1)

    # scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
    #                       args.epochs * num_batches, args.min_lr)

    stats = []
    for epoch in trange(0, args.epochs):
        # If set curriculum epochs
        if args.curriculum_epoch is not None and epoch >= args.curriculum_epoch:
            # print('Restart scheduler')
            # scheduler = cosine_lr(optimizer, args.lr, args.warmup_length,
            #             (args.epochs - args.curriculum_epoch) * num_batches, args.min_lr)

            if cur_strength != 0:
                print('Restart dataloader')
                cur_strength = 0
                cur_str_times = 1
                
                print(f"loading strength = {cur_strength}, loop times {cur_str_times}")
                img_text_data = get_data(
                    args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
                    epoch=0, strength=cur_strength)
                ft_dataloader = img_text_data['train_ft'].dataloader
                ft_iterator = iter(ft_dataloader)
                num_batches = len(ft_dataloader)

        print("Epoch : ", epoch)
        epoch_stats = {}
        epoch_stats['epoch'] = epoch
        id_flyp_loss_sum = 0
        model.train()
        model = model.cuda()
        classification_head.train()

        for i in trange(num_batches):
            start_time = time.time()
            step = i + epoch * num_batches
            if epoch != -1:
                scheduler(step)
            optimizer.zero_grad()

            # pdb.set_trace()

            try:
                ft_batch = next(ft_iterator)
            except StopIteration:
                if args.curriculum:
                    if args.curriculum_epoch is None:
                        cur_strength_id += 1
                        if cur_strength_id >= len(list_strength):
                            cur_strength_id = 0
                        cur_strength = list_strength[cur_strength_id]
                    else:
                        if epoch <= args.curriculum_epoch:
                            if cur_str_times < loop_times:
                                cur_str_times += 1
                            else:
                                cur_str_times = 1
                                cur_strength_id += 1
                                if cur_strength_id >= len(list_strength):
                                    cur_strength_id = len(list_strength) - 1
                                cur_strength = list_strength[cur_strength_id]
                        else:
                            cur_strength = 0
                            cur_str_times = 1
                    
                    print(f"loading strength = {cur_strength}, loop times {cur_str_times}")
                    img_text_data = get_data(
                        args, (clip_encoder.train_preprocess, clip_encoder.val_preprocess),
                        epoch=0, strength=cur_strength)
                    ft_dataloader = img_text_data['train_ft'].dataloader

                ft_iterator = iter(ft_dataloader)
                ft_batch = next(ft_iterator)

            ft_image, ft_text = ft_batch
            ft_image, ft_text = ft_image.cuda(), ft_text.cuda()

            ft_image_features, ft_text_features, logit_scale2 = model(
                ft_image, ft_text)
            ft_clip_loss = clip_loss_fn(ft_image_features,
                                        ft_text_features,
                                        logit_scale2)

            ft_clip_loss.backward()
            optimizer.step()

            id_flyp_loss_sum += ft_clip_loss.item()

            if i % print_every == 0:
                percent_complete = 100 * i / num_batches
                logger.info(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{num_batches}]\t"
                    f"ID FLYP Loss: {ft_clip_loss.item():.4f}")

        id_flyp_loss_avg = id_flyp_loss_sum / num_batches

        # Evaluate
        args.current_epoch = epoch
        classification_head_new = get_zeroshot_classifier(
            args, model.module.model)
        classification_head_new = classification_head_new.cuda()


        eval_results = evaluate(model, args, classification_head_new,
                                epoch_stats, logger)
        # pdb.set_trace()

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            model_path = os.path.join(args.save, f'checkpoint_{epoch}.pt')
            logger.info('Saving model to' + str(model_path))
            # model.module.save(model_path)
            torch.save(model.module.state_dict(), model_path)
            optim_path = os.path.join(args.save, f'optim_{epoch}.pt')
            torch.save(optimizer.state_dict(), optim_path)

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

        ################################################
        # class acc:
        class_stats = dict()
        ind_dataset = {k: i for i, k in enumerate(args.eval_datasets)}
        for k, v in epoch_stats.items():
            if 'Class' in k:
                    if k == 'ImageNet Accuracy':
                        #ignore the ID acc term
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
                        class_stats[cur_label_name][2*ds_id] = v
                    if 'Number' in k:
                        cls_label = list_k[1].replace(' Number', '')
                        cur_label_name = f"Class {cls_label}"
                        if cur_label_name not in class_stats:
                            cur_res = [0] * 2 * len(args.eval_datasets)
                            class_stats[cur_label_name] = cur_res
                        class_stats[cur_label_name][2*ds_id + 1] = v

        list_colum = [''] * 2 * len(args.eval_datasets)
        for i in range(len(args.eval_datasets)):
            list_colum[2*i] = args.eval_datasets[i]
            list_colum[2*i+1] = args.eval_datasets[i] + ' Count'

        class_stats_df = pd.DataFrame.from_dict(class_stats, orient='index', columns=list_colum)
        log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(log_dir, exist_ok=True)
        class_stats_df.to_csv(log_dir + f'/class_stats{epoch}.tsv', sep='\t')

        epoch_stats['Avg OOD Acc'] = round(ood_acc, 4)
        logger.info(f"Avg OOD Acc : {ood_acc:.4f}")
        logger.info(f"Avg ID FLYP Loss : {id_flyp_loss_avg:.4f}")
        epoch_stats['Avg ID FLYP Loss'] = round(id_flyp_loss_avg, 4)
        epoch_stats = {key: values for key, values in epoch_stats.items() if ' Class' not in key}
        stats.append(epoch_stats)
        stats_df = pd.DataFrame(stats)
        log_dir = "expt_logs/" + args.exp_name + "/" + "_BS" + str(
            args.batch_size) + "_WD" + str(args.wd) + "_LR" + str(args.lr) + "_run" + str(args.run)
        os.makedirs(log_dir, exist_ok=True)
        stats_df.to_csv(log_dir + '/stats.tsv', sep='\t')

    if args.save is not None:
        return model_path