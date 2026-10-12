"""MVUNet entry point; keep the training implementation in train.py.

    python train_mvunet.py --config config/segmentation/mvu_net.yaml
"""
from train import train_pipeline


if __name__ == '__main__':
    train_pipeline()
