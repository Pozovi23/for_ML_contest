import argparse
from inference import inference_sahi
import os

def main():
    parser = argparse.ArgumentParser(description='Run SAHI inference on an image.')
    parser.add_argument('file_dir_src', type=str, help='Path to the source image file')
    parser.add_argument('--file_dir_out', type=str, default='./output', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0'], help='Device to run inference on')

    args = parser.parse_args()

    os.makedirs(args.file_dir_out, exist_ok=True)

    for file in os.listdir(args.file_dir_src):
        inference_sahi(
            filepath_src=args.file_dir_src + '/' + file,
            file_dir_out=args.file_dir_out,
            filename_out=os.path.splitext(file)[0],
            device=args.device
        )

if __name__ == '__main__':
    main()

