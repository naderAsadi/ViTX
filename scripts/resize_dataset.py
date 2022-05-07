import argparse
from pathlib import Path
from PIL import Image
from multiprocessing import Value
from multiprocessing.pool import ThreadPool


def resize(batch):
    key, image_path = batch
    try:
        img = Image.open(image_path)
        img = img.resize((args.image_size, args.image_size))
        img.save(f"{args.target_path}/{key}.jpg")

        global counter
        with counter.get_lock():
            counter.value += 1
    except:
        pass

    print(f"Resized {counter.value}/{num_images} images", end="\r")


def init(args):
    global counter
    counter = args


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

    global num_images
    num_images = len(image_files.keys())
    # initialize a cross-process counter
    counter = Value("i", 0)

    thread_pool = ThreadPool(
        processes=args.n_threads, initializer=init, initargs=(counter,)
    )
    pooled_output = thread_pool.map(resize, image_files.items())

    print(f"{num_images - counter.value} / {num_images} images were corrupted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_path", type=str)
    parser.add_argument("--target_path", type=str)
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--n_threads", type=int, default=8)
    args = parser.parse_args()

    main(args=args)
