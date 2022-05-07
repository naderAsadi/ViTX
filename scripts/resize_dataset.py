import argparse
from pathlib import Path
from PIL import Image


def main(args):
    # get image files path
    images_path = Path(args.images_path)
    image_files = [
        *images_path.glob("**/*.png"),
        *images_path.glob("**/*.jpg"),
        *images_path.glob("**/*.jpeg"),
        *images_path.glob("**/*.bmp"),
    ]
    image_files = {str(file.parts[-1].split(".")[0]): file for file in image_files}

    c = 0
    for key, image_path in image_files.items():
        try:
            img = Image.open(image_path)
            img = img.resize((args.image_size, args.image_size))
            img.save(f"{args.target_path}/{key}.jpg")
            c += 1
        except:
            continue
        
        print(f"Resized {c}/{len(image_files.keys())} images", end='\r')
    
    print(f"{len(image_files.keys()) - c} / {len(image_files.keys())} images were corrupted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, help='Path to the folder containing dataset image files.')
    parser.add_argument('--target_path', type=str, help='Path to the folder where resized images will be stored.')
    parser.add_argument('--image_size', type=int, default=336, help='Target image size.')
    args = parser.parse_args()

    main(args=args)