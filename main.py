import argparse
from inference import inference_sahi

def main():
    parser = argparse.ArgumentParser(description='Run SAHI inference on an image.')
    parser.add_argument('filepath_src', type=str, help='Path to the source image file')
    parser.add_argument('--file_dir_out', type=str, default='./output', help='Output directory for results')
    parser.add_argument('--filename_out', type=str, default='out_sahi', help='Output filename (without extension)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda:0'], help='Device to run inference on')
    parser.add_argument('--save', type=bool,  help='Save the output image')

    args = parser.parse_args()

    inference_sahi(
        filepath_src=args.filepath_src,
        file_dir_out=args.file_dir_out,
        filename_out=args.filename_out,
        device=args.device,
        save=args.save
    )

if __name__ == '__main__':
    main()