from itertools import cycle
from multiprocessing import Pool
import pickle
import subprocess
import face_alignment
import torch.nn as nn
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1,3'

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 1920:
        scale_factor =  max(frame.shape[0], frame.shape[1]) / 1920.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor



def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_bbox(start, end, fps, tube_bbox, frame_shape, inp, image_shape, increase_area=0.1):
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top

    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    h, w = bot - top, right - left

    start = start / fps
    end = end / fps
    time = end - start

    scale = f'{image_shape[0]}:{image_shape[1]}'

    return f'ffmpeg -i {inp} -ss {start} -t {time} -filter:v "crop={w}:{h}:{left}:{top}, scale={scale}" crop.mp4'


def compute_bbox_trajectories(trajectories, fps, frame_shape, args):
    commands = []
    for i, (bbox, tube_bbox, start, end) in enumerate(trajectories):
        if (end - start) > args.min_frames:
            tt=start+args.min_frames
            # print(start,end)
            while tt<end:
                # print(tt)
                command = compute_bbox(tt-args.min_frames, tt, fps, tube_bbox, frame_shape, inp=args.inp, image_shape=args.image_shape, increase_area=args.increase)
                commands.append(command)
                tt=tt+args.min_frames
    return commands


def process_video(params):
    filename,devices,args=params
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    # torch.multiprocessing.set_start_method('spawn')
    file_path = os.path.join(args.dataset_path,filename)
    args.inp=file_path

    device = 'cpu' if args.cpu else 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    # fa = nn.DataParallel(fa)

    video = imageio.get_reader(args.inp)

    trajectories = []
    previous_frame = None
    fps = video.get_meta_data()['fps']
    commands = []
    try:
        for i, frame in tqdm(enumerate(video)):
            frame_shape = frame.shape
            bboxes =  extract_bbox(frame, fa)
            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > args.iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)

            commands += compute_bbox_trajectories(not_valid_trajectories, fps, frame_shape, args)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection and current_intersection > args.iou_with_initial:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)


    except IndexError as e:
        raise (e)

    commands += compute_bbox_trajectories(trajectories, fps, frame_shape, args)
    return commands

def scheduler(data_list, fn, args):
    device_ids = args.device_ids.split(",")
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    # f = open(args.chunks_metadata, 'w')
    # line = "{video_id},{start},{end},{bbox},{fps},{width},{height},{partition}"
    # print (line.replace('{', '').replace('}', ''), file=f)
    for chunks_data in tqdm(pool.imap_unordered(fn, zip(data_list, cycle(device_ids), args_list))):
        # for data in chunks_data:
        pass
            # print (line.format(**data), file=f)
            # f.flush()
    # f.close()
def pscheduler(data_list, fn, args):
    device_ids = args.device_ids.split(",")
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    # f = open(args.chunks_metadata, 'w')
    # line = "{video_id},{start},{end},{bbox},{fps},{width},{height},{partition}"
    # print (line.replace('{', '').replace('}', ''), file=f)
    commands=[]
    for command in tqdm(pool.imap_unordered(fn, zip(data_list, cycle(device_ids), args_list))):
        commands = commands+command
        # for data in chunks_data:
            # pass
            # print (line.format(**data), file=f)
            # f.flush()
    with open('ffmpeg_commands', 'wb') as fp:
        pickle.dump(commands,fp)
    # f.close()
def run(params):
    # ffmpeg -i .\videos\id0001.mp4 -ss 4.533333333333333 -t 51.06666666666667 
    # -filter:v "crop=646:646:383:69, scale=1024:1024" crop.mp4
    cmd, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    # print(cmd)
    # return
    cmd_args = cmd.split(' ')
    cmd_args[-1]=os.path.split(cmd_args[2])[1].split('.')[0]+'_'+cmd_args[4]+'_'+cmd_args[6]+'.mp4'
    cmd_args[-1]=os.path.join(args.output_path, cmd_args[-1])
    # print(cmd_args[-1])
    # return
    cmd_args[8:10] = [' '.join(cmd_args[8:10])]
    cmd_args[8]=cmd_args[8].replace('"','')
    cmd_args.insert(1,'-y')
    # print(cmd_args)
    # return
    subprocess.call(cmd_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_shape", default=(1024, 1024), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--workers", default=8, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0,1,2,3", help="Names of the devices comma separated.")
    parser.add_argument("--inp", help='Input image or video')
    parser.add_argument("--dataset_path",default="bili-fangtan", help="Path to videos")
    parser.add_argument("--output_path",default="fangtan",help="Path to croped videos")
    # parser.add_argument("--dataset_path",default="videos_test", help="Path to videos")
    # parser.add_argument("--output_path",default="videos_crop_test",help="Path to croped videos")
    parser.add_argument("--min_frames", type=int, default=120,  help='Minimum number of frames')
    parser.add_argument("--use_cache", type=bool, default=True,  help='Use cached ffmpeg commands')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
    parser.add_argument("--ffmpeg", default='ffmpeg', help='Path to ffmpeg')

    args = parser.parse_args()
    # torch.multiprocessing.set_start_method('spawn')
    # print(args.dataset_path)
    _, _, filenames = next(os.walk(args.dataset_path), (None, None, []))
    # print(filenames)
    commands = []
    args.use_cache=False
    if args.use_cache == False:
        pscheduler(filenames,process_video,args)

    # for filename in tqdm(filenames):
    #     file_path = os.path.join(args.dataset_path,filename)
    #     # print(file_path)
    #     args.inp=file_path
    #     commands = commands+process_video(args)
    # with open('ffmpeg_commands', 'wb') as fp:
    #     pickle.dump(commands,fp)

    with open('ffmpeg_commands', 'rb') as fp:
        commands = pickle.load(fp)
        # print(commands)
    scheduler(commands,run,args)
    # for command in commands:
    #     # print(command)
    #     run(zip(command,args))
    #     break