import os
import glob
import timeit
from PIL import Image as pil_image
from tqdm import tqdm

# SOME CODE HERE FOR TRAINING DIRECTORY SPECIFICATION

def get_file_list(ip_pth):
    # read all the file names in list
    pth_aud = []
    for path, subdirs_categ, files in os.walk(ip_pth):
        if len(files) != 0:
            a = glob.glob(path + "//*.png")
            pth_aud = pth_aud + a
    return pth_aud

def get_files(ip_path):
    pth_aud = glob.glob(ip_path + "//*.png")
    return pth_aud

def check_img(img_pth,cnt):
    img = None
    # print('processing '+ img_pth )
    try:
        img = pil_image.open(img_pth)  # open the image file
        img.verify()  # verify that it is, in fact an image
        cnt = cnt + 1
    except (IOError, SyntaxError) as e:
        cnt = cnt -1
        print('Bad file:', img_pth)
    finally:
        if img is not None:
            img.close()
    return

train_dir = 'C:/ML//env//tf//pycharm_areds3//areds//input//train'
tr_lst = get_file_list(train_dir)
subdir = os.listdir(train_dir)
print(" There are " + str(len(tr_lst))  + " files in training dir: " + train_dir)

# get each file in respective folder
for each_dir in subdir:
    pt = train_dir+ "//" + each_dir
    tr_lst_each = get_files(pt)
    cnt = 0
    for pth in tqdm(tr_lst_each):
        check_img(pth,cnt)
    print("sucessfully opened and closed " + str(len(tr_lst_each))  + " files in train subir " +
    each_dir)