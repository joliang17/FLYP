import os
import json

import torch
import numpy as np
from src.models import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.laion import get_data, get_csv_dataset

from tqdm import tqdm
import src.datasets as datasets
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_f1_score
from src.datasets.iwildcam import IWildCamOOD
import pdb


def logging_input(curinput='', logger=None):
    if logger is not None:
        logger.info(curinput)
    else:
        print(curinput)
    return


def process_train_stat(results, train_stats, logger, dataset_name=''):
    for key, val in results.items():
        if ('worst' in key or 'f1' in key.lower() or 'pm0' in key) and 'guidance' not in key.lower():
            logging_input(f"{dataset_name} {key}: {val:.4f}", logger)
            train_stats[dataset_name + key] = round(val, 4)
    return


def eval_single_dataset_onTrain(image_classifier, args, classification_head, ):
    model = image_classifier
    input_key = 'images'

    model.eval()
    classification_head.eval()

    dataloader = get_csv_dataset(args, image_classifier.module.val_preprocess, is_train=False, return_guidance=True,
                                 return_img_id=True, only_img_id=True).dataloader

    batched_data = enumerate(dataloader)
    device = args.device

    dict_preds = dict()  # save predict value of currect class for each image with different strength

    with torch.no_grad():
        for i, data in tqdm(batched_data, total=len(dataloader)):
            data = maybe_dictionarize(data, progress_train=True)
            x = data[input_key].to(device)
            y = data['labels'].to(device)
            guidances = data['guidance'].to(device)
            img_ids = data['img_id'].to(device)

            logits = utils.get_logits(x, model, classification_head)

            # find the largest prob of y
            all_prob = F.softmax(logits, dim=-1)
            for i, img_id_t in enumerate(img_ids):
                img_id = img_id_t.item()
                cur_label = y[i].item()
                cur_prob = all_prob[i, cur_label].item()
                cur_guid = guidances[i].item()
                if img_id not in dict_preds:
                    dict_preds[img_id] = []
                dict_preds[img_id].append([cur_guid, cur_prob])

    metrics = {}
    # dict_best_guid = dict()
    # for img_id,list_guid_prob in dict_preds.items():
    #     list_guid_prob = sorted(list_guid_prob, key=lambda x: x[-1], reverse=True)
    #     best_guid = list_guid_prob[0][0]
    #     dict_best_guid[img_id] = best_guid

    metrics['best_guid'] = dict_preds
    return metrics


def eval_single_dataset(image_classifier, dataset, args, classification_head, progress_eval=False, ):

    model = image_classifier
    input_key = 'images'
    image_enc = None

    model.eval()
    classification_head.eval()

    if progress_eval:
        if args.progress_train:
            dataloader = get_csv_dataset(args, image_classifier.module.val_preprocess, is_train=False,
                                         return_guidance=True, return_img_id=True, only_img_id=True).dataloader
        else:
            dataloader = get_csv_dataset(args, image_classifier.module.val_preprocess, is_train=False,
                                         return_guidance=True).dataloader


    elif not args.self_data:
        ## equals to dataloader = dataset.test_loader
        dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=image_enc)

        ## for oxfordpet dataset
        if getattr(dataset, 'index_cat', None) is not None:
            list_index_cat = dataset.index_cat
            list_index_dog = dataset.index_dog

            index_dog = 79
            index_cat = 66
    else:
        dataloader = get_csv_dataset(args, image_classifier.module.val_preprocess, is_train=False, ).dataloader

    batched_data = enumerate(dataloader)
    device = args.device

    if args.self_data or hasattr(dataset, 'post_loop_metrics'):
        # keep track of labels, predictions and metadata
        all_labels, all_preds, all_metadata = [], [], []

    if progress_eval:
        dict_labels = dict()
        dict_preds = dict()

    list_index = None
    if isinstance(dataset, IWildCamOOD) and not args.progress_train:
        import pickle
        with open(f"../data/analysis/test_used_id/all_index.pkl", 'rb') as f:
            list_index = pickle.load(f)

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        dict_class = dict()
        dict_guidance = dict()
        for i, data in batched_data:
            if args.progress_train:
                data = maybe_dictionarize(data, progress_train=True)
            else:
                data = maybe_dictionarize(data, progress_eval=progress_eval)

            x = data[input_key].to(device)
            y = data['labels'].to(device)

            if 'guidance' in data:
                guidance = data['guidance']

            if 'image_paths' in data:
                image_paths = data['image_paths']

            logits = utils.get_logits(x, model, classification_head)

            projection_fn = getattr(dataset, 'project_logits', None)
            if projection_fn is not None:
                logits = projection_fn(logits, device)

            if hasattr(dataset, 'project_labels'):
                y = dataset.project_labels(y, device)
            pred = logits.argmax(dim=1, keepdim=True).to(device)

            ## for oxfordpet dataset
            if getattr(dataset, 'index_cat', None) is not None:
                y_new = torch.ones_like(y) * index_cat
                for i in range(len(y)):
                    if y[i] not in list_index_cat:
                        y_new[i] = index_dog
                y = y_new  # pdb.set_trace()

            if hasattr(dataset, 'accuracy'):
                acc1, num_total = dataset.accuracy(logits, y, image_paths, args)
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

                if progress_eval and args.progress_train:
                    guidances = torch.unique(guidance)
                    for guid_i in guidances:
                        guid_i = guid_i.item()
                        sap_ids = (guidance == guid_i).nonzero(as_tuple=True)
                        cur_pred = pred[sap_ids]
                        cur_y = y[sap_ids]

                        cur_correct = cur_pred.eq(cur_y.view_as(cur_pred)).sum().item()
                        cur_num = len(sap_ids[0])
                        if guid_i not in dict_guidance:
                            dict_guidance[guid_i] = [0, 0]

                        dict_guidance[guid_i][0] += cur_correct
                        dict_guidance[guid_i][1] += cur_num

                        if guid_i not in dict_labels:
                            dict_labels[guid_i] = []
                            dict_preds[guid_i] = []
                        dict_labels[guid_i].append(cur_y.cpu().clone().detach())
                        dict_preds[guid_i].append(cur_pred.cpu().clone().detach())

            if args.self_data or hasattr(dataset, 'post_loop_metrics'):
                all_labels.append(y.cpu().clone().detach())
                all_preds.append(logits.cpu().clone().detach())
                metadata = data['metadata'] if 'metadata' in data else image_paths
                all_metadata.extend(metadata)

        top1 = correct / n
        # pdb.set_trace()
        if args.self_data or hasattr(dataset, 'post_loop_metrics'):
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)

            if list_index is not None:
                # exclude test cases in validate set
                mask = torch.ones(all_labels.shape, dtype=torch.bool)
                mask[list_index] = False
                all_labels = all_labels[mask]
                all_preds = all_preds[mask]

            if not args.self_data:
                metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
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

    if progress_eval:
        dict_guidance_f1 = dict()
        for guid_i in dict_labels.keys():
            cur_str_labels = dict_labels[guid_i]
            cur_str_preds = dict_preds[guid_i]
            # pdb.set_trace()
            cur_str_labels = torch.cat(cur_str_labels)
            cur_str_preds = torch.cat(cur_str_preds)
            cur_str_preds = torch.squeeze(cur_str_preds)
            f1_cur_str = multiclass_f1_score(cur_str_preds, cur_str_labels, num_classes=181, average="macro")
            dict_guidance_f1[guid_i] = f1_cur_str.item()
        metrics['guidance_f1'] = dict_guidance_f1

    if 'top1' not in metrics:
        metrics['top1'] = top1

    if len(dict_class) > 0:
        metrics['class_top1'] = dict_class

    if len(dict_guidance) > 0:
        metrics['guidance_top1'] = dict_guidance

    return metrics


def eval_single_batch_dataset(image_classifier, dataset, args, classification_head, data):

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
            metrics = dataset.post_loop_metrics(all_labels, all_preds, all_metadata, args)
            if 'acc' in metrics:
                metrics['top1'] = metrics['acc']
        else:
            metrics = {}
    if 'top1' not in metrics:
        metrics['top1'] = top1

    return metrics['top1'], cnt_loss.item()


def evaluate(image_classifier, args, classification_head, train_stats={}, logger=None, progress_eval=False,
             progress_train=False):
    if args.eval_datasets is None:
        return
    info = vars(args)

    if progress_train:
        # Evaluate the best guidance on training dataset for each image
        logging_input(f"Evaluating on training dataset", logger)
        results = eval_single_dataset_onTrain(image_classifier, args, classification_head, )

        train_stats[f"Best Guid per Image"] = results['best_guid']
        return info

    if progress_eval:
        # load specific curriculum data and evaluate performance on group of guidance
        logging_input(f"Evaluating on curriculum evaluation dataset", logger)
        dataset = None
        results = eval_single_dataset(image_classifier, dataset, args, classification_head, progress_eval=True)
        if 'guidance_f1' in results:
            dict_guidance_f1 = results['guidance_f1']
            list_acc = [[key, value] for key, value in dict_guidance_f1.items()]

            for pair in list_acc:
                logging_input(f"Guidance F1: {pair[0]} {pair[1]:.4f}", logger)
                train_stats[f"Guidance {pair[0]} F1"] = round(pair[1], 4)

        if 'guidance_top1' in results:
            # pdb.set_trace()

            list_acc = [[key, value[0] / value[1], value[1]] for key, value in results['guidance_top1'].items()]
            list_acc = sorted(list_acc, key=lambda x: x[1], reverse=False)
            for pair in list_acc:
                logging_input(f"Guidance Top-1 accuracy: {pair[0]} {pair[1]:.4f}", logger)
                train_stats[f"Guidance {pair[0]} Accuracy"] = round(pair[1], 4)
                train_stats[f"Guidance {pair[0]} Number"] = pair[2]

        process_train_stat(results, train_stats, logger)

        return info

    for i, dataset_name in enumerate(args.eval_datasets):
        logging_input(f"Evaluating on {dataset_name}", logger)

        dataset_class = getattr(datasets, dataset_name)
        if not args.self_data:
            dataset = dataset_class(image_classifier.module.val_preprocess, location=args.data_location,
                                    batch_size=args.batch_size)
        else:
            dataset = None

        results = eval_single_dataset(image_classifier, dataset, args, classification_head)

        if 'top1' in results:
            logging_input(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}", logger)
            train_stats[dataset_name + " Accuracy"] = round(results['top1'], 4)

        if 'class_top1' in results:
            list_acc = [[key, value[0] / value[1], value[1]] for key, value in results['class_top1'].items()]
            list_acc = sorted(list_acc, key=lambda x: x[1], reverse=False)
            for pair in list_acc:
                # logging_input(f"{dataset_name} Class Top-1 accuracy: {pair[0]} {pair[1]:.4f}", logger)
                train_stats[dataset_name + f" Class {pair[0]} Accuracy"] = round(pair[1], 4)
                train_stats[dataset_name + f" Class {pair[0]} Number"] = pair[2]

        process_train_stat(results, train_stats, logger, dataset_name)

    return info
