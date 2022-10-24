from glob import glob
import pickle
import random
import os

# with open ('bv_file', 'rb') as fp:
#     itemlist = pickle.load(fp)
#     print(itemlist)
# test_bv = random.sample(itemlist, 20)
# with open('test_bv', 'wb') as fp:
#     pickle.dump(test_bv,fp)
with open ('test_bv', 'rb') as fp:
    itemlist = pickle.load(fp)
    print(itemlist)

for name in itemlist:
    files_path = os.path.join("/home/hyc/fangtan/train",name+'*')
    files = glob(files_path)
    # print(files)
    for file in files:
        save_path = os.path.join('/home/hyc/fangtan/test', os.path.basename(file))
        os.replace(file,save_path)
# os.replace("path/to/current/file.foo", "path/to/new/destination/for/file.foo")