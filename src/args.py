import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help=
        "Which datasets to use for evaluation. Split by comma, e.g. CIFAR101,CIFAR102."
        " Note that same model used for all datasets, so much have same classnames"
        "for zero shot.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        help="For fine tuning or linear probe, which dataset to train on",
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help=
        "Which prompt template is used. Leave as None for linear probe, etc.",
    )
    parser.add_argument(
        "--classnames",
        type=str,
        default="openai",
        help="Which class names to use.",
    )
    parser.add_argument(
        "--alpha",
        default=[0.5],
        nargs='*',
        type=float,
        help=
        ('Interpolation coefficient for ensembling. '
         'Users should specify N-1 values, where N is the number of '
         'models being ensembled. The specified numbers should sum to '
         'less than 1. Note that the order of these values matter, and '
         'should be the same as the order of the classifiers being ensembled.'
         ))
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only.")

    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B/32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")

    parser.add_argument("--ls",
                        type=float,
                        default=0.0,
                        help="Label smoothing.")
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help=
        "Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=
        "Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--freeze-encoder",
        default=False,
        action="store_true",
        help=
        "Whether or not to freeze the image encoder. Only relevant for fine-tuning."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--fisher",
        type=lambda x: x.split(","),
        default=None,
        help="TODO",
    )
    parser.add_argument(
        "--fisher_floor",
        type=float,
        default=1e-8,
        help="TODO",
    )

    parser.add_argument(
        "--ft_data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )

    parser.add_argument(
        "--ft_data_test",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )

    parser.add_argument('--ce_ablation', action=argparse.BooleanOptionalAction)

    parser.add_argument('--curriculum', action=argparse.BooleanOptionalAction)

    parser.add_argument('--baseline', action=argparse.BooleanOptionalAction)

    parser.add_argument('--cont_finetune', action=argparse.BooleanOptionalAction)

    parser.add_argument('--progress_eval', action=argparse.BooleanOptionalAction)

    parser.add_argument('--progress_train', action=argparse.BooleanOptionalAction)

    parser.add_argument('--progress', action=argparse.BooleanOptionalAction)

    parser.add_argument('--progress_validation', action=argparse.BooleanOptionalAction)

    parser.add_argument(
        "--progress_metric",
        type=str,
        default='Acc',
        help="Acc or F1.",
    )

    parser.add_argument(
        "--cluster",
        type=str,
        default='',
        help="cluster method, loss / others",
    )
    parser.add_argument('--ma_progress', action=argparse.BooleanOptionalAction)

    parser.add_argument('--explore', action=argparse.BooleanOptionalAction)

    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)

    parser.add_argument('--proportion', action=argparse.BooleanOptionalAction)

    parser.add_argument('--test', action=argparse.BooleanOptionalAction)

    parser.add_argument('--scheduler', type=str, default='default',)

    parser.add_argument('--datalimit', type=int, default=-1,)

    parser.add_argument(
        "--curriculum_epoch",
        type=int,
        default=None,
        help=
        "Number of samples in dataset. Required for webdataset if not available in info file.",
    )

    parser.add_argument('--self_data', action=argparse.BooleanOptionalAction)

    parser.add_argument("--dataset-type",
                        choices=["webdataset", "csv", "auto"],
                        default="auto",
                        help="Which type of dataset to process.")

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help=
        "Number of samples in dataset. Required for webdataset if not available in info file.",
    )

    parser.add_argument("--k",
                        type=int,
                        default=None,
                        help="k for few shot ImageNet")
                        
    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="Default random seed.")

    parser.add_argument("--workers",
                        type=int,
                        default=6,
                        help="Number of dataloader workers per GPU.")

    parser.add_argument("--csv-separator",
                        type=str,
                        default="\t",
                        help="For csv-like datasets, which separator to use.")
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths.")
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions.")


    parser.add_argument(
        "--clip_load",
        type=str,
        default=None,
        help="Load finetuned clip",
    )

    parser.add_argument(
        "--wise_save",
        type=str,
        default=None,
        help="Save path for wiseft results",
    )

    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Repeated run number",
    )

    parser.add_argument("--get_labeled_csv",
                        default=False,
                        action="store_true",
                        help="get labels from csv.")


    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="minimum LR for cosine scheduler",
    )

    parser.add_argument(
        "--wandb_group_name",
        type=str,
        default='default',
        help="wandb group for expr results",
    )

    parser.add_argument("--guidance",
                        type=int,
                        default=-1,
                        help="Number of dataloader workers per GPU.")

    parser.add_argument("--slurm_job_id",
                        type=int,
                        default=-1,
                        help="SLURM job id.")
                        
    parsed_args = parser.parse_args()

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
