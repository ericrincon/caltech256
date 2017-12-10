import argparse

from train import run
from models import get_available_models

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--model')
    argument_parser.add_argument('--train-data', default='data/train')
    argument_parser.add_argument('--test-data', default='data/test')
    argument_parser.add_argument('--height', default=139, type=int)
    argument_parser.add_argument('--width', default=139, type=int)
    argument_parser.add_argument('--batch-size', default=32, type=int)
    argument_parser.add_argument('--image_shape', default='', type=str)
    argument_parser.add_argument('--epochs', type=int, default=50)
    argument_parser.add_argument('--lr', type=float, default=.001)

    args = argument_parser.parse_args()

    if args.image_shape == '':
        image_shape = (args.width, args.height, 3)
    else:
        image_shape = [int(x) for x in args.image_shape.split(',')]

    default_params = {
        'image_shape': image_shape,
        'train_data': args.train_data,
        'test_data': args.test_data,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }

    if args.model is None:
        models = []

        for model_name in get_available_models():
            params = default_params.copy()

            if model_name == 'pvgg16' or model_name == 'irsv2':
                params['lr'] = .0001
                params['epochs'] = 25

            models.append((model_name, params))
    else:
        models = [(args.model, default_params)]

    for model, params in models:
        run(model, **params)






if __name__ == '__main__':
    main()
