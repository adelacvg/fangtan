from argparse import ArgumentParser
import os
import pickle
import subprocess
from tqdm import tqdm
DEVNULL = open(os.devnull, 'wb')
def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.bilibili.com/video/" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path

parser = ArgumentParser()
parser.add_argument("--video_folder", default='bili-fangtan', help='Path to bilibili videos')
parser.add_argument("--metadata", default='fangtan-metadata.csv', help='Path to metadata')
parser.add_argument("--out_folder", default='fangtan-png', help='Path to output')
parser.add_argument("--youtube", default='yt-dlp', help='Path to youtube-dl')
args = parser.parse_args()
if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
l=[]
with open ('bv_file', 'rb') as fp:
    l = pickle.load(fp)
for vid in tqdm(l):
    download(vid,args)