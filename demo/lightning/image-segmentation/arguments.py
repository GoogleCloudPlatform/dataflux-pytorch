"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import argparse
import os

PARSER = argparse.ArgumentParser(description="UNet-3D")

PARSER.add_argument("--log_dir", dest="log_dir", type=str, default="/tmp")
PARSER.add_argument("--save_ckpt_path",
                    dest="save_ckpt_path",
                    type=str,
                    default="")
PARSER.add_argument("--load_ckpt_path",
                    dest="load_ckpt_path",
                    type=str,
                    default="")
PARSER.add_argument("--loader", dest="loader", default="pytorch", type=str)
PARSER.add_argument("--local_rank",
                    default=os.environ.get("LOCAL_RANK", 0),
                    type=int)

PARSER.add_argument("--local",
                    dest="local",
                    default=False,
                    action="store_true")
PARSER.add_argument("--epochs", dest="epochs", type=int, default=5)
PARSER.add_argument("--accelerator",
                    dest="accelerator",
                    type=str,
                    default="auto")
PARSER.add_argument("--quality_threshold",
                    dest="quality_threshold",
                    type=float,
                    default=0.908)
PARSER.add_argument("--ga_steps", dest="ga_steps", type=int, default=1)
PARSER.add_argument("--warmup_steps", dest="warmup_steps", type=int, default=4)
PARSER.add_argument("--batch_size", dest="batch_size", type=int, default=20)
PARSER.add_argument("--layout",
                    dest="layout",
                    type=str,
                    choices=["NCDHW"],
                    default="NCDHW")
PARSER.add_argument("--input_shape",
                    nargs="+",
                    type=int,
                    default=[128, 128, 128])
PARSER.add_argument("--val_input_shape",
                    nargs="+",
                    type=int,
                    default=[128, 128, 128])
PARSER.add_argument("--seed", dest="seed", default=-1, type=int)
PARSER.add_argument("--num_devices", dest="num_devices", type=int, default=1)
PARSER.add_argument("--num_nodes", dest="num_nodes", type=int, default=1)
PARSER.add_argument("--num_workers", dest="num_workers", type=int, default=10)
PARSER.add_argument("--prefetch_factor",
                    dest="prefetch_factor",
                    type=int,
                    default=2)
PARSER.add_argument("--exec_mode",
                    dest="exec_mode",
                    choices=["train", "evaluate"],
                    default="train")

PARSER.add_argument("--benchmark",
                    dest="benchmark",
                    default=False,
                    action="store_true")
PARSER.add_argument("--listing_only",
                    dest="listing_only",
                    default=False,
                    action="store_true")
PARSER.add_argument("--step_time", dest="step_time", type=float, default=0)
PARSER.add_argument("--amp", dest="amp", action="store_true", default=False)
PARSER.add_argument(
    "--optimizer",
    dest="optimizer",
    default="sgd",
    choices=["sgd", "adam", "lamb"],
    type=str,
)
PARSER.add_argument("--learning_rate",
                    dest="learning_rate",
                    type=float,
                    default=1.0)
PARSER.add_argument("--init_learning_rate",
                    dest="init_learning_rate",
                    type=float,
                    default=1e-4)
PARSER.add_argument("--lr_warmup_epochs",
                    dest="lr_warmup_epochs",
                    type=int,
                    default=0)
PARSER.add_argument("--lr_decay_epochs", nargs="+", type=int, default=[])
PARSER.add_argument("--lr_decay_factor",
                    dest="lr_decay_factor",
                    type=float,
                    default=1.0)
PARSER.add_argument("--lamb_betas", nargs="+", type=int, default=[0.9, 0.999])
PARSER.add_argument("--momentum", dest="momentum", type=float, default=0.9)
PARSER.add_argument("--weight_decay",
                    dest="weight_decay",
                    type=float,
                    default=0.0)
PARSER.add_argument("--evaluate_every",
                    "--eval_every",
                    dest="evaluate_every",
                    type=int,
                    default=None)
PARSER.add_argument("--start_eval_at",
                    dest="start_eval_at",
                    type=int,
                    default=None)
PARSER.add_argument("--verbose",
                    "-v",
                    dest="verbose",
                    action="store_true",
                    default=False)
PARSER.add_argument(
    "--normalization",
    dest="normalization",
    type=str,
    choices=["instancenorm", "batchnorm"],
    default="instancenorm",
)
PARSER.add_argument(
    "--activation",
    dest="activation",
    type=str,
    choices=["relu", "leaky_relu"],
    default="relu",
)

PARSER.add_argument("--oversampling",
                    dest="oversampling",
                    type=float,
                    default=0.4)
PARSER.add_argument("--overlap", dest="overlap", type=float, default=0.5)
PARSER.add_argument(
    "--include_background",
    dest="include_background",
    action="store_true",
    default=False,
)
PARSER.add_argument("--cudnn_benchmark",
                    dest="cudnn_benchmark",
                    action="store_true",
                    default=False)
PARSER.add_argument(
    "--cudnn_deterministic",
    dest="cudnn_deterministic",
    action="store_true",
    default=False,
)

PARSER.add_argument(
    "--images_prefix",
    dest="images_prefix",
    type=str,
    default=False,
)

PARSER.add_argument(
    "--labels_prefix",
    dest="labels_prefix",
    type=str,
    default=False,
)

PARSER.add_argument(
    "--gcp_project",
    dest="gcp_project",
    type=str,
    default=False,
)

PARSER.add_argument(
    "--gcs_bucket",
    dest="gcs_bucket",
    type=str,
    default=False,
)
