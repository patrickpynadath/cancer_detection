from processing import MammographyPreprocessor, get_paths
import argparse





# data preprocessing


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Breast-Cancer based on Mammography")

    # preprocessing data args
    parser.add_argument('--process', help='run image preprocessing', action='store_true')
    parser.add_argument('--finalheight', default=256, type=int, help='final img height for processing')
    parser.add_argument('--finalwidth', default=128, type=int, help='final img width for processing')
    parser.add_argument('--par', help='run processing in parralel', action='store_false')
    parser.add_argument('--base_data_dir', help='base dir for data', type=str, default='data')

    args = parser.parse_args()

    if args.process:
        final_size = (args.finalwidth, args.finalheight)
        base_dir = args.base_data_dir
        mp = MammographyPreprocessor(size=final_size, csv_path = f'{base_dir}/train.csv', train_path=f'{base_dir}/train_images')
        paths = get_paths()
        mp.preprocess_all(paths, parallel=args.par)


