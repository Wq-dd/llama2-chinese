from pathlib import Path
import argparse

import pandas as pd
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import debugpy
from tqdm import tqdm

from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from dataset_sft import SFTDataset
from dataset import PretrainDataset
from model import Transformer, ModelArgs



def parse_args():
    parser = argparse.ArgumentParser(description="sft")

    # For train.
    parser.add_argument("--train_type", default="pretrain", type=str, choices=["sft", "pretrain"])
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    # For MoE (Mixture of Experts).
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )
    parser.add_argument(
        "--ep-world-size", default=1, type=int, help="(moe) expert parallel world size"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="number of experts list, MoE related.",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="Only applicable when num-experts > 1, accepts [standard, residual]",
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def create_moe_param_groups(model):
    """Create separate parameter groups for each expert."""
    parameters = {"params": [p for p in model.parameters()], "name": "parameters"}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": 48,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15,
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "offload_optimizer": {
                "device": "none",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            # "cpu_offload": False,
        },
    }
    return ds_config


def init_model():
    # model init
    # model init
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    max_seq_len = globals()["max_len"]
    dropout = 0
    init_from = "scratch"
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,#64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )  # start with model_args from command line
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    return model


def sft_train_epoch(args, model_engine, trainloader, local_device, local_rank):
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(enumerate(trainloader), dynamic_ncols=True) as pbar:
            for i, (X, Y, loss_mask) in pbar:
                X = X.to(local_device)
                Y = Y.to(local_device)
                loss_mask = loss_mask.to(local_device)

                with torch.cuda.amp.autocast():
                    logits = model_engine(X, Y)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0,reduce=False)
                    loss_mask = loss_mask.view(-1)
                    loss = torch.sum(loss*loss_mask)/loss_mask.sum()

                model_engine.backward(loss)
                model_engine.step()

                # Print statistics
                running_loss += loss.item()
                if local_rank == 0 and i % args.log_interval == (
                    args.log_interval - 1
                ):  # Print every log_interval mini-batches.
                    pbar.write(
                        f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
                    )
                    running_loss = 0.0

    return


def pretrain_epoch(args, model_engine, trainloader, local_device, local_rank):
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        with tqdm(enumerate(trainloader), dynamic_ncols=True, total=len(trainloader)) as pbar:
            for i, (X, Y) in pbar:
                X = X.to(local_device)
                Y = Y.to(local_device)
                with torch.cuda.amp.autocast():
                    logits = model_engine(X, Y)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
                    loss = torch.sum(loss)/(loss.shape[0])
                    running_loss += loss.item()
                model_engine.backward(loss)
                model_engine.step()
                if local_rank == 0 and i % args.log_interval == ( args.log_interval - 1):
                    pbar.write(
                        f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
                    )
                    running_loss = 0.0
    return


def train(args):
    globals()["max_len"] = 1024
    if args.local_rank == -1:
        local_device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        local_device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        local_device  = torch.device(f"{local_device}")
        deepspeed.init_distributed()
    tokenizer=ChatGLMTokenizer(vocab_file='/data/wq/code/github/llama2-chinese/chatglm_tokenizer/tokenizer.model')
    if args.train_type == "sft":
        df = pd.read_csv('./sft_data/sft_data.csv')
        trainset = SFTDataset(df,tokenizer, max_length=globals()["max_len"])
    elif args.train_type == "pretrain":
        trainset = PretrainDataset(data_path_lst=[
            "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_1.bin",
            "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_2.bin",
            "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_3.bin",
            "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_4.bin",
            "/data/wq/dataset/pretrain_data_bin/baidubaike/baidubaike_563w_5.bin",
        ],
        max_length=globals()["max_len"]
        )

    net = init_model()
    estimate_zero3_model_states_mem_needs_all_live(net, num_gpus_per_node=1, num_nodes=1)

    # import debugpy
    # debugpy.listen(("localhost", 5678))
    # print("Listen port: 5678")
    # debugpy.wait_for_client()

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    ds_config = get_ds_config(args)
    torch.distributed.barrier()
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=parameters,
        training_data=trainset,
        config=ds_config,
    )
    local_rank = model_engine.local_rank
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    if args.train_type == "sft":
        sft_train_epoch(args, model_engine, trainloader, local_device, local_rank)
    elif args.train_type == "pretrain":
        pretrain_epoch(args, model_engine, trainloader, local_device, local_rank)

    return


if __name__ == "__main__":
    args = parse_args()
    train(args)
