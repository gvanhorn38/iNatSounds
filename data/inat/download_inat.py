import csv
import tqdm
import json
from matplotlib import pyplot as plt
from urllib.request import urlretrieve
import os
from multiprocessing import Process
import time
from datetime import timedelta
import datetime
import numpy as np
import argparse

def get_sound_data(args):

    SOUND_DATA_PATH = "sound_data.json"

    if os.path.exists(SOUND_DATA_PATH):
        with open(SOUND_DATA_PATH, "r") as f:
            sound_data = json.load(f)
        return sound_data
    
    id2label = {}
    with open(args.obs_path, "r") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        for row in tqdm.tqdm(datareader, total = 334616):
            count += 1
            if count == 1:
                continue

            cur_id = int(row[0])
            cur_label = int(row[4])
            id2label[cur_id] = cur_label


    sound_data = []
    with open(args.media_path, "r") as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        for row in tqdm.tqdm(datareader, total = 360975):
            count += 1
            if count == 1:
                continue
            # ['observation_id', 'format', 'base_url', 'license', 'file_ext', 'asset_id']
            obs_id, fmt, base_url, _, file_ext, asset_id = row
            cur_dict = {
                "obs_id": obs_id,
                "url": base_url,
                "asset_id": asset_id,
                "label": id2label[int(obs_id)],
                "file_ext": file_ext,
            }
            sound_data.append(cur_dict)
    with open("sound_data.json", "w") as f:
        json.dump(sound_data, f, indent=4)
    return sound_data


def get_ext(fname):
    return fname.split(".")[-1].split("?")[0]
def download_file(args, entry):
    url = entry["url"]
    ext = entry["file_ext"]
    asset_id = entry["asset_id"]
    cur_dir = os.path.join(args.raw_sound_dir, str(entry["label"]))
    fname = os.path.join(cur_dir, "{}.{}".format(asset_id, ext))
    if os.path.exists(fname):
        return False
    try:
        urlretrieve(url, fname)
    except KeyboardInterrupt:
        exit()
    except:
        return False
    return True
def download_files(args, entries):
    for entry in tqdm.tqdm(entries, leave = False):
        new_download = download_file(args, entry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--media_path', type=str, default="./data_csv/media.csv")
    parser.add_argument('--obs_path', type=str, default="./data_csv/observations.csv")
    parser.add_argument('--raw_sound_dir', type=str, default="./sound_files/raw_sound")
    args = parser.parse_args()

    sound_data = get_sound_data(args)

    classes = {}
    for d in sound_data:
        cl = d["label"]
        if cl not in classes.keys():
            classes[cl] = 0
        classes[cl] += 1
    n_classes = len(classes.keys())

    all_classes = list(classes.keys())
    for c in all_classes:
        cur_dir = os.path.join(args.raw_sound_dir, str(c))
        os.makedirs(cur_dir, exist_ok=True)



    n_threads = 64
    processes = []
    size_each = (len(sound_data) + n_threads - 1) // n_threads
    for rank in range(n_threads):
        strt = rank*size_each
        end = min((rank+1)*size_each, len(sound_data))
        entries = sound_data[strt:end]
        p = Process(target=download_files, args=(args, entries, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
