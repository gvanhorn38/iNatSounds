import os
import tqdm
import librosa
import numpy as np
import soundfile as sf
import argparse
import cv2
from multiprocessing import Process
import warnings
warnings.filterwarnings("ignore")

def change_ext(fname):
    base = fname.split(".")[-2]
    return base + ".wav"

def convert_file_list(args, file_list):
    for dirname, fname in tqdm.tqdm(file_list):
        orig_fpath = os.path.join(args.raw_sound, dirname, fname)
        new_fpath = os.path.join(args.wav_sound, dirname, change_ext(fname))

        command = "~/ffmpeg/ffmpeg -y -hide_banner -loglevel error -i {} -ar 22050 -ac 1 {}".format(orig_fpath, new_fpath)
        os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_sound', type=str, default="./sound_files/raw_sound")
    parser.add_argument('--wav_sound', type=str, default="./sound_files/sound_wav_22050")
    args = parser.parse_args()

    os.makedirs(args.wav_sound, exist_ok=True)

    dirlist = [(os.path.join(args.raw_sound, d), d) for d in os.listdir(args.raw_sound)]
    dirlist = sorted(dirlist)
    file_list = []
    for dir_path, dirname in dirlist:
        if ".DS" in dirname: continue
        for fname in os.listdir(dir_path):
            if ".DS" in fname: continue

            file_list.append(
                (dirname, fname)
            )

    file_list = sorted(file_list)
    filt_file_list = []
    all_dirs = []
    for dirname, fname in tqdm.tqdm(file_list, desc="filtering files"):
        check_path = os.path.join(args.wav_sound, dirname, change_ext(fname))
        if os.path.exists(check_path):
            continue
        filt_file_list.append((dirname, fname))
        all_dirs.append(dirname)
    file_list = sorted(filt_file_list)
    all_dirs = list(set(all_dirs))

    for dirname in tqdm.tqdm(all_dirs):
        os.makedirs(os.path.join(args.wav_sound, dirname), exist_ok=True)


    n_threads = 64
    processes = []
    size_each = (len(file_list) + n_threads - 1) // n_threads
    for rank in range(n_threads):
        strt = rank*size_each
        end = min((rank+1)*size_each, len(file_list))
        entries = file_list[strt:end]
        p = Process(target=convert_file_list, args=(args, entries))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    