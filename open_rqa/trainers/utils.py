import os
import shutil
import logging
import torch
import pathlib
import sys


logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
    logging.getLogger("rouge_score.rouge_scorer").setLevel(logging.ERROR)
    return logger


def create_dir_if_not_exists(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def remove_optimizer_weights(save_dir):
    for checkpoint_dirs in os.listdir(save_dir):
        checkpoint_dir = os.path.join(save_dir, checkpoint_dirs)
        if os.path.isdir(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith('global_step'):
                    optimizer_dir = os.path.join(checkpoint_dir, file)
                    # remove the entire folder. This is used by deepspeed to store optimizer states
                    print('removing', optimizer_dir)
                    shutil.rmtree(optimizer_dir)
                elif file.startswith("optimizer.pt"):
                    optimizer_file = os.path.join(checkpoint_dir, file)
                    print('removing', optimizer_file)
                    os.remove(optimizer_file)
    return