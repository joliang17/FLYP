import os
import json

import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.laion import get_data, get_csv_dataset

import src.datasets as datasets
import torch.nn.functional as F
import pdb


def eval_single_dataset(image_classifier, dataset, args, classification_head):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head.eval()

    if not args.self_data:
        ## equals to dataloader = dataset.test_loader
        dataloader = get_dataloader(dataset,
                                    is_train=False,
                                    args=args,
                                    image_encoder=image_enc)

    else:
        dataloader = get_csv_dataset(args, image_classifier.module.val_preprocess, is_train=False, ).dataloader
    
    batched_data = enumerate(dataloader)
    device = args.device

    if args.self_data or hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        dict_class = dict()
        for i, data in batched_data:

            data = maybe_dictionarize(data)
            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits = utils.get_logits(x, model, classification_head)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths,
                                                   args)
                correct += acc1
                n += num_total
            else:
                correct += pred.eq(y.view_as(pred)).sum().item()
                n += y.size(0)

                classes = torch.unique(y)
                for cls_i in classes:
                    cls_i = cls_i.item()
                    sap_ids = (y == cls_i).nonzero(as_tuple=True)
                    cur_pred = pred[sap_ids]
                    cur_correct = (cur_pred == cls_i).sum().item()
                    cur_num = len(sap_ids[0])
                    if cls_i not in dict_class:
                        dict_class[cls_i] = [0, 0]
                    
                    dict_class[cls_i][0] += cur_correct
                    dict_class[cls_i][1] += cur_num

            if args.self_data or hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data[
                    'metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n

        if args.self_data or hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            if not args.self_data:
                metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                    all_metadata, args)
            else:
                preds_temp = all_preds.argmax(dim=1, keepdim=True).view_as(all_labels)
                correct = preds_temp.eq(all_labels).sum().item()
                all_cnt = preds_temp.size(0)
                acc = correct / all_cnt
                metrics = {'acc': acc}

            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1
    
    if len(dict_class) > 0:
        metrics['class_top1'] = dict_class

    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args,
                              classification_head, data):

    model = image_classifier
    input_key = 'images'

    model.eval()
    classification_head.eval()

    device = args.device

    if hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    with torch.no_grad():
        top1, correct, n, cnt_loss = 0., 0., 0., 0.

        data = maybe_dictionarize(data)
        x = data[input_key].to(device)
        y = data['labels'].to(device)

        assert x.shape[0] == 2 * args.k, 'val mismatch size'

        if 'image_paths' in data:
            image_paths = data['image_paths']

        logits = utils.get_logits(x, model, classification_head)

        projection_fn = getattr(dataset, 'project_logits', None)
        if projection_fn is not None:
            logits = projection_fn(logits, device)

        if hasattr(dataset, 'project_labels'):
            y = dataset.project_labels(y, device)

        cnt_loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
        if hasattr(dataset, 'accuracy'):
            acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
            correct += acc1
            n += num_total
        else:
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels.append(y.cpu().clone().detach())
            all_preds.append(logits.cpu().clone().detach())
            metadata = data['metadata'] if 'metadata' in data else image_paths
            all_metadata.extend(metadata)

        top1 = correct / n

        if hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            metrics = dataset.post_loop_metrics(all_labels, all_preds,
                                                all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics['top1'], cnt_loss.item()


def evaluate(image_classifier,
             args,
             classification_head,
             train_stats={},
             logger=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)
        dataset_class = getattr(datasets, dataset_name)
        if not args.self_data:
            dataset = dataset_class(image_classifier.module.val_preprocess,
                                    location=args.data_location,
                                    batch_size=args.batch_size)
        else:
            dataset = None

        results = eval_single_dataset(image_classifier, dataset, args,
                                      classification_head)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            if logger != None:
                logger.info(
                    f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
            train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)
        
        if 'class_top1' in results:
            list_acc = [[key, value[0]/value[1]] for key, value in results['class_top1'].items()]
            list_acc = sorted(list_acc, key=lambda x: x[1], reverse=False)
            for pair in list_acc:
                print(f"{dataset_name} Class Top-1 accuracy: {pair[0]} {pair[1]:.4f}")
                if logger != None:
                    logger.info(
                        f"{dataset_name} Class Top-1 accuracy: {pair[0]} {pair[1]:.4f}")
                train_stats[dataset_name + f" Class {pair[0]} Accuracy"] = round(pair[1], 4)

        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
                if logger != None:
                    logger.info(f"{dataset_name} {key}: {val:.4f}")
                train_stats[dataset_name + key] = round(val, 4)

    return info