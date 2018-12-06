import numpy as np
from PIL import Image
import glob
import random
import pickle

def load_file_list(mode):
    DATASET_ROOT = "/mnt/hdd1/dataset/signate"
    if mode=='train':
        train_dir = "/train{0:02d}"
        TC_flist = []
        nonTC_flist = []
        for i in range(16):
            TC_flist += glob.glob(DATASET_ROOT + train_dir.format(i) + "/TC/*.tif")
            nonTC_flist += glob.glob(DATASET_ROOT + train_dir.format(i) + "/nonTC/*.tif")
        return TC_flist, nonTC_flist

    elif mode=='test':
        test_dir = "/test{}"
        test_flist = []
        for i in range(3):
            test_flist += glob.glob(DATASET_ROOT + test_dir.format(i) + "/*.tif")
        return test_flist

def load_image(flist):
    img_list = []
    for fname in flist:
        img = np.array(Image.open(fname, 'r'))
        img_list.append(np.float32(img))
    return img_list

def Load(mode, new=False):
    if mode=='train':
        if new:
            print("Load File List")
            TC_flist, nonTC_flist = load_file_list('train')
            # n_neg = n_pos
            #nonTC_flist = random.sample(nonTC_flist, len(TC_flist)*4)
            print("Load Positive Image")
            pos = load_image(TC_flist)
            x = []
            x += pos
            for i, img in enumerate(pos):
                print("\r{}".format(i+1),end="")
                r90 = np.rot90(img).astype(np.float32)
                r180 = np.rot90(r90).astype(np.float32)
                r270 = np.rot90(r180).astype(np.float32)
                x += [r90,r180,r270]
            print("")
            print("Load Negative Image")
            neg = load_image(nonTC_flist)
            x += neg
            t = [1 if i<len(pos)*4 else 0 for i in range(len(x))]
            x = np.array(x).reshape(-1,1,64,64).astype(np.float32)
            t = np.array(t).astype(np.int32)
            rnd_idx = np.random.choice(range(len(t)), len(t), replace=False)
            x = x[rnd_idx]
            t = t[rnd_idx]
            with open("train_all_rot.pkl", "wb") as f:
                pickle.dump((x, t), f, protocol=4)
        else:
            with open("train.pkl", "rb") as f:
                x, t = pickle.load(f)
        return x, t

    elif mode=='test':
        if new:
            print("Load File List")
            flist = load_file_list('test')
            print("Load Test Image")
            x = load_image(flist)
            x = np.array(x).reshape(-1,1,64,64).astype(np.float32)
            t = [fname.split('/')[-1] for fname in flist]
            with open("test.pkl", "wb") as f:
                pickle.dump((x, t), f, protocol=4)
        else:
            with open("test.pkl", "rb") as f:
                x, t = pickle.load(f)
        return x, t