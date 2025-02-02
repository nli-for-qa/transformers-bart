import logging
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import (
    ALL_PRETRAINED_MODEL_ARCHIVE_MAP,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_auto import MODEL_MAPPING

logger = logging.getLogger(__name__)

if os.environ.get("LIGHTNING_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=LEVEL)

ALL_MODELS = tuple(ALL_PRETRAINED_MODEL_ARCHIVE_MAP)
MODEL_CLASSES = tuple(m.model_type for m in MODEL_MAPPING)

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class BaseTransformer(pl.LightningModule):
    def __init__(self, hparams, num_labels=None, mode="base"):
        "Initialize a model."

        super(BaseTransformer, self).__init__()
        self.hparams = hparams
        self.hparams.model_type = self.hparams.model_type.lower()
        config = AutoConfig.from_pretrained(
            self.hparams.config_name

            if self.hparams.config_name else self.hparams.model_name_or_path,
            **({
                "num_labels": num_labels
            } if num_labels is not None else {}),
            cache_dir=self.hparams.cache_dir

            if self.hparams.cache_dir else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name if self.hparams.tokenizer_name else
            self.hparams.model_name_or_path,
            do_lower_case=self.hparams.do_lower_case,
            cache_dir=self.hparams.cache_dir

            if self.hparams.cache_dir else None,
        )
        model = MODEL_MODES[mode].from_pretrained(
            self.hparams.model_name_or_path,
            from_tf=bool(".ckpt" in self.hparams.model_name_or_path),
            config=config,
            cache_dir=self.hparams.cache_dir

            if self.hparams.cache_dir else None,
        )
        self.config, self.tokenizer, self.model = config, tokenizer, model

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()

                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()

                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon)
        self.opt = optimizer

        return [optimizer]

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       second_order_closure=None):

        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1]
        }

        return tqdm_dict

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def train_dataloader(self):
        train_batch_size = self.hparams.train_batch_size
        dataloader = self.load_dataset("train", train_batch_size)

        t_total = ((len(dataloader.dataset)
                    // (train_batch_size * max(1, self.hparams.n_gpu)))
                   // self.hparams.gradient_accumulation_steps * float(
                       self.hparams.num_train_epochs))
        scheduler = get_linear_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total)
        self.lr_scheduler = scheduler

        return dataloader

    def val_dataloader(self):
        return self.load_dataset("dev", self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.load_dataset("test", self.hparams.eval_batch_size)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams.data_dir,
            "cached_{}_{}_{}".format(
                mode,
                list(filter(None,
                            self.hparams.model_name_or_path.split("/"))).pop(),
                str(self.hparams.max_seq_length),
            ),
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument(
            "--model_type",
            default=None,
            type=str,
            required=True,
            help="Model type selected in the list: "
            + ", ".join(MODEL_CLASSES),
        )
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pre-trained model or shortcut name selected in the list: "
            + ", ".join(ALL_MODELS),
        )
        parser.add_argument(
            "--config_name",
            default="",
            type=str,
            help="Pretrained config name or path if not the same as model_name"
        )
        parser.add_argument(
            "--tokenizer_name",
            default="",
            type=str,
            help="Pretrained tokenizer name or path if not the same as model_name",
        )
        parser.add_argument(
            "--cache_dir",
            default="",
            type=str,
            help="Where do you want to store the pre-trained models downloaded from s3",
        )
        parser.add_argument(
            "--do_lower_case",
            action="store_true",
            help="Set this flag if you are using an uncased model.")
        parser.add_argument(
            "--learning_rate",
            default=5e-5,
            type=float,
            help="The initial learning rate for Adam.")
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight decay if we apply some.")
        parser.add_argument(
            "--adam_epsilon",
            default=1e-8,
            type=float,
            help="Epsilon for Adam optimizer.")
        parser.add_argument(
            "--warmup_steps",
            default=0,
            type=int,
            help="Linear warmup over warmup_steps.")
        parser.add_argument(
            "--num_train_epochs",
            default=3,
            type=int,
            help="Total number of training epochs to perform.")

        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results

            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(
                pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(
                            key, str(metrics[key])))


def add_generic_args(parser, root_dir):
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_tpu_cores", type=int, default=0)
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument('--early_stop_patience', default=0, type=int)

    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")


def setup_trainer(args):
    # init model
    set_seed(args)

    # Setup distant debugging if needed

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.output_dir, '{epoch}'),
        monitor="val_loss",
        mode="min",
        verbose=True,
        save_top_k=1)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.early_stop_patience,
        verbose=True,
        mode='min')
    # wandb logger
    wandb_logger = WandbLogger(project="bart-qa-to-nli")
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        early_stop_callback=early_stop_callback,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
        logger=wandb_logger,
        callbacks=[LoggingCallback()],
        val_check_interval=0.25,
    )

    if args.fp16:
        train_params["use_amp"] = args.fp16
        train_params["amp_level"] = args.fp16_opt_level

    if args.n_tpu_cores > 0:
        global xm
        import torch_xla.core.xla_model as xm

        train_params["num_tpu_cores"] = args.n_tpu_cores
        train_params["gpus"] = 0

    if args.n_gpu > 1:
        train_params["distributed_backend"] = "ddp"

    trainer = pl.Trainer(**train_params)

    return trainer
