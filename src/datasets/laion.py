import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import pdb
import pickle

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize


def logging_input(curinput='', logger=None):
    if logger is not None:
        logger.info(curinput)
    else:
        print(curinput)
    return


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", label_key=None, guidance=None,
                 datalimit=-1, ori_proportion=None, uniform_guid=False, return_guidance=False,
                 return_img_id=False, only_img_id=False, reshift_distribution=False, include_neg=False,logger=None):
        # logging_input(f'Loading csv data from {input_filename}.', logger)
        df = pd.read_csv(input_filename, sep=sep)
        df_pos = df[df['label'] != 0]
        df_neg = df[df['label'] == 0]
        len_neg = len(df_neg)

        ##########################
        # mixture from original data * image guidance
        df_ori = None
        if ori_proportion is not None:
            df_ori = df[df['guidance'] == 100]

        if reshift_distribution:
            df = df[df['guidance'] == 100]
            df = df.sample(n=10000, replace=False, ignore_index=True)
        
        # for sample experiment, only sample few samples from training data
        self.only_img_id = only_img_id
        if self.only_img_id:
            # sort the df by img_id
            # generated img only here
            df = df[df['img_id'] >= 0]
            # df = df.sample(n=10000, replace=False, ignore_index=True) 
            df = df.sort_values(by='img_id', )

        if uniform_guid:
            # only train on a uniformly distributed dataset
            # df = df.sample(n=10000, replace=False, ignore_index=True)
            # method1 :
            # including guid1 : guid2 : guid3 : .. : guidn : neg = 1:1:...:1
            if include_neg:
                df_pos_temp = df_pos.groupby('guidance').apply(lambda x: x.sample(n=1000, replace=False, )).reset_index(drop=True)
                df_neg_temp = df_neg.sample(n=min(len_neg, 1000), replace=False, ignore_index=True).reset_index(drop=True)
                df = pd.concat([df_pos_temp, df_neg_temp])
                logging_input(f'sampling pos data {len(df_pos_temp)}, neg data{len(df_neg_temp)}.', logger)
            else:
                df = df_pos.groupby('guidance').apply(lambda x: x.sample(n=1000, replace=False, )).reset_index(drop=True)

            logging_input(f'sampling total data {len(df)}.', logger)

        ##########################
        # only loading guidance
        if guidance is not None:
            # only positive is included if guid != 100
            df = df[df['guidance'] == guidance]
            if datalimit != -1 and len(df) > datalimit:
                df = df.sample(n=datalimit, replace=False, ignore_index=True)
                logging_input(f'sampling guid={guidance} with {len(df)} samples.', logger)
            if guidance != 100 and include_neg:
                # mix with negative samples
                neg_cnt = min(len_neg, int(len(df) / 2))
                df_neg_temp = df_neg.sample(n=neg_cnt, replace=False, ignore_index=True)
                df = pd.concat([df, df_neg_temp])
                logging_input(f'sampling neg with {len(df_neg_temp)} samples.', logger)

        ##########################
        # mixture from original data * image guidance
        if ori_proportion is not None:
            num_df = len(df)
            num_ori = min(len(df_ori), int(num_df / (1 - ori_proportion) * ori_proportion))
            df_ori = df_ori.sample(n=num_ori, replace=False, ignore_index=True)
            df = pd.concat([df, df_ori])
            logging_input(f'Concatted data {num_df} + {num_ori} = {len(df)}.', logger)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        title_col = [item for item in df.columns if caption_key in item]
        num_columns = len(title_col)

        self.return_guidance = return_guidance
        if self.return_guidance:
            self.guidance = df['guidance'].tolist()
            self.img_trans = T.ToPILImage()

        self.return_img_id = return_img_id
        if self.return_img_id:
            self.img_id = df['img_id'].tolist()

        self.captions_list = []
        for k in range(1, num_columns):
            self.captions_list.append(df[f"{caption_key}_{k}"])

        self.return_label = False
        if label_key is not None:
            self.return_label = True
            self.labels = list(map(int, df[label_key].tolist()))
            self.img_path = df["filepath"].tolist()
            self.prompt = df["title"].tolist()
        self.transforms = transforms

        # self.classes = max(self.labels) + 1
        logging_input(f'Loading data with length {len(self.images)}.', logger)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        if img_path.endswith('.pkl'):
            with open(img_path, 'rb') as f:
                images = pickle.load(f)
            if torch.is_tensor(images):
                images = self.img_trans(images)
        else:
            images = Image.open(img_path)

        images = self.transforms(images)

        texts = tokenize([str(self.captions[idx])])[0]

        return_label = [images, texts, ]
        if len(self.captions_list) > 0:
            texts_list = [tokenize([str(self.captions_list[i][idx])])[0] for i in range(len(self.captions_list))]
            texts_list.append(texts)
            texts_list = torch.stack(texts_list, dim=0)
            perm = torch.randperm(texts_list.shape[0])
            texts_list = texts_list[perm, :]

            return_label.append(texts_list)

        if self.return_label:
            label = self.labels[idx]
            f_path = self.img_path[idx]
            f_title = self.prompt[idx]

            return_label.append(label)
            return_label.append(f_path)
            return_label.append(f_title)

        if self.return_guidance:
            guidance = self.guidance[idx]
            return_label.append(guidance)

        if self.return_img_id:
            img_id = self.img_id[idx]
            return_label.append(img_id)

        return return_label


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text):
    return tokenize([str(text)])[0]


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined  # some common dataset sizes (at time of authors last download)  # CC3M (train): 2905954  # CC12M: 10968539  # LAION-400M: 407332084  # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             sampler=sampler, )

    return DataInfo(dataloader=dataloader, sampler=sampler)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption(sample):
    return 'txt' in sample


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1, ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, urls, nshards=sys.maxsize, worker_seed=None, deterministic=False, epoch=-1, ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False):
    input_shards = args.replay_data  # if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError('Currently, number of dataset samples must be specified for training dataset. '
                                   'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([detshuffle2(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL, seed=args.seed,
                                         epoch=shared_epoch, ), wds.split_by_node, wds.split_by_worker, ])
        pipeline.extend([  # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL, ), ])
    else:
        pipeline.extend(
            [wds.split_by_worker,  # at this point, we have an iterator over the shards assigned to each worker
             wds.tarfile_to_samples(handler=log_and_continue), ])
    pipeline.extend([wds.select(filter_no_caption), wds.decode("pilrgb", handler=log_and_continue),
                     wds.rename(image="jpg;png", text="txt"), wds.map_dict(image=preprocess_img, text=preprocess_txt),
                     wds.to_tuple("image", "text"), wds.batched(args.batch_size, partial=not is_train), ])

    dataset = wds.DataPipeline(*pipeline)
    # import pdb;pdb.set_trace()
    if is_train:
        print("entered")
        # import pdb;pdb.set_trace()
        if not resampled:
            print(args.workers)
            print(args.workers * 1)
            print(num_shards)

            # assert num_shards >= args.workers * 1, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * 1
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=args.workers,
                               persistent_workers=True, )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * 1
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, guidance=None, ori_proportion=None, 
                    uniform_guid=False, return_guidance=False, return_img_id=False, only_img_id=False,
                    reshift_distribution=False, include_neg=False, datalimit=None, logger=None):
    # normal training / curriculum eval on test dataset
    input_filename = args.ft_data if is_train else args.ft_data_test
    assert input_filename

    if args.get_labeled_csv:
        label_key = args.supervised_label_key

    else:
        label_key = None

    if not is_train:
        label_key = 'label'

    dataset = CsvDataset(input_filename, preprocess_fn, logger=logger, img_key=args.csv_img_key,
                         caption_key=args.csv_caption_key, sep=args.csv_separator, label_key=label_key,
                         guidance=guidance, datalimit=datalimit,
                         uniform_guid=uniform_guid, reshift_distribution=reshift_distribution,
                         return_guidance=return_guidance, return_img_id=return_img_id, only_img_id=only_img_id,
                         ori_proportion=ori_proportion, include_neg=include_neg, )
    num_samples = len(dataset)
    # sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    sampler = None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                            pin_memory=True, sampler=sampler, drop_last=False, )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    # dataloader.num_classes = dataset.classes

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, logger=None, epoch=0, guidance=None, ori_proportion=None, uniform_guid=False, datalimit=None, 
             return_img_id=False, reshift_distribution=False, include_neg=False):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    data["train_ft"] = get_dataset_fn(args.ft_data, args.dataset_type)(args, preprocess_train, is_train=True,
                                                                       epoch=epoch, guidance=guidance,
                                                                       ori_proportion=ori_proportion,
                                                                       uniform_guid=uniform_guid,
                                                                       logger=logger, datalimit=datalimit,
                                                                       reshift_distribution=reshift_distribution,
                                                                       return_img_id=return_img_id, include_neg=include_neg, )

    return data
