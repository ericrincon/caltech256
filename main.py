import argparse

from models import build_model
from train import train_model

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model')

    args = argument_parser.parse_args()

    model = build_model(args.model_name, )






if __name__ == '__main__':
    main()
