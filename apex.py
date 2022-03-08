# THIS FILE IS PART OF Caesar PROJECT
# main_nonblock.py - The core part of the AI assistant
#
# THIS PROGRAM IS A FREE PROGRAM, WHICH IS LICENSED UNDER Caesar
# DO NOT FORWARD THIS PROGRAM TO ANYONE
from aim_csgo.apex_aim import lock, recoil_control
from aim_csgo.screen_inf import grab_screen_mss, grab_screen_win32, get_parameters
from aim_csgo.cs_model import load_model
import cv2
import win32gui
import win32con
import torch
import numpy as np

from aim_csgo.verify_args import verify_args
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
import pynput
from threading import Thread
import argparse
import time
import os
from simple_pid import PID

"鼠标控制使用ghub实现，使用前需先打开一次Logitech GHUB"
"游戏与桌面分辨率不一致时需要开启全屏模式，不能是无边框窗口"
"鼠标移动在fov为90的游戏中下最准确 其他fov能用，但可能效果没那么好"
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default='runs/train/exp14/weights/best.pt', help='模型地址')
parser.add_argument('--imgsz', type=int, default=640, help='和你训练模型时imgsz一样')
parser.add_argument('--conf-thres', type=float, default=0.1, help='置信阈值')
parser.add_argument('--iou-thres', type=float, default=0.65, help='交并比阈值')
parser.add_argument('--use-cuda', type=bool, default=True, help='是否使用cuda')

parser.add_argument('--show-window', type=bool, default=False, help='是否显示实时检测窗口(新版里改进了效率。若为True，不要去点右上角的X')
parser.add_argument('--top-most', type=bool, default=True, help='是否保持实时检测窗口置顶')
parser.add_argument('--resize-window', type=float, default=1/2, help='缩放实时检测窗口大小')
parser.add_argument('--thickness', type=int, default=5, help='画框粗细，必须大于1/resize-window')
parser.add_argument('--show-fps', type=bool, default=True, help='是否显示帧率')
parser.add_argument('--show-label', type=bool, default=False, help='是否显示标签')

parser.add_argument('--use_mss', type=str, default=False, help='是否使用mss截屏；为False时使用win32截屏，自行比对速度')

parser.add_argument('--region', type=tuple, default=(0.15, 0.4), help='检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')

parser.add_argument('--hold-lock', type=bool, default=True, help='lock模式；True为按住，False为切换')
parser.add_argument('--lock-sen', type=float, default= 3.0, help='lock幅度系数；若在桌面试用请调成1，在游戏中(csgo)则为灵敏度')
parser.add_argument('--lock-smooth', type=float, default=1.8, help='lock平滑系数；越大越平滑，最低1.0')#3.3
parser.add_argument('--lock-button', type=str, default='x2', help='lock按键；只支持鼠标按键')
parser.add_argument('--head-first', type=bool, default=True, help='是否优先瞄头')
parser.add_argument('--lock-tag', type=list, default=[0], help='对应标签；person')
parser.add_argument('--lock-choice', type=list, default=[0], help='目标选择；可自行决定锁定的目标，从自己的标签中选')

args = parser.parse_args()

'------------------------------------------------------------------------------------'

verify_args(args)

cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'

args.model_path = cur_dir + args.model_path
args.lock_tag = [str(i) for i in args.lock_tag]
args.lock_choice = [str(i) for i in args.lock_choice]

device = 'cuda' if args.use_cuda else 'cpu'
half = device != 'cpu'
imgsz = args.imgsz

conf_thres = args.conf_thres
iou_thres = args.iou_thres

top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))

monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}

model = load_model(args)
stride = int(model.stride.max())
names = model.module.names if hasattr(model, 'module') else model.names

lock_mode = False
team_mode = True
lock_button = eval('pynput.mouse.Button.' + args.lock_button)

mouse = pynput.mouse.Controller()

pidx = PID(1.2, 3, 0.0, setpoint=0, sample_time=0.001,)
pidy = PID(1.1, 0.08, 0.0, setpoint=0, sample_time=0.001,)
pidx.output_limits = (-5000 ,5000)
pidy.output_limits = (-2500 ,2500)
#pidx = PID(3, 0.05, 0, setpoint=0, sample_time=0.01,)
#pidy = PID(5, 0, 0, setpoint=0, sample_time=0.03,)
#pidx.output_limits = (-300,300)
#t = Thread(target=recoil_control, kwargs={'args': args})
#t.start()

if args.show_window:
    cv2.namedWindow('aim', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('aim', int(len_x * args.resize_window), int(len_y * args.resize_window))


def on_click(x, y, button, pressed):
    global lock_mode
    if button == lock_button:
        if args.hold_lock:
            if pressed:
                lock_mode = True
                print('locking...')
            else:
                lock_mode = False
                print('lock mode off')
        else:
            if pressed:
                lock_mode = not lock_mode
                print('lock mode', 'on' if lock_mode else 'off')

listener = pynput.mouse.Listener(on_click=on_click)
listener.start()

print('enjoy yourself!')
t0 = time.time()
cnt = 0

while True:
    '''
    if cnt % 20 == 0:
        top_x, top_y, x, y = get_parameters()
        len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
        top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
        monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}
        cnt = 0
    '''
    if args.use_mss:
        img0 = grab_screen_mss(monitor)
        img0 = cv2.resize(img0, (len_x, len_y))
    else:
        img0 = grab_screen_win32(region=(top_x, top_y, top_x + len_x, top_y + len_y))
        img0 = cv2.resize(img0, (len_x, len_y))

    img = letterbox(img0, imgsz, stride=stride)[0]

    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.

    if len(img.shape) == 3:
        img = img[None]

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)

    aims = []
    for i, det in enumerate(pred):
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # bbox:(tag, x_center, y_center, x_width, y_width)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                aim = ('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                aims.append(aim)

        if len(aims):
            if lock_mode:
                lock(aims, mouse, top_x, top_y, len_x, len_y, args, pidx, pidy)

        if args.show_window:
            for i, det in enumerate(aims):
                tag, x_center, y_center, width, height = det
                x_center, width = len_x * float(x_center), len_x * float(width)
                #print("width:" , width)
                #print("x_center:", x_center)
                y_center, height = len_y * float(y_center), len_y * float(height)
                top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                #print("top_left:", top_left)
                bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                #print("bottom_right:", bottom_right)
                cv2.rectangle(img0, top_left, bottom_right, (0, 255, 0), thickness=args.thickness)
                if args.show_label:
                    cv2.putText(img0, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 0, 0), 4)
    if args.show_fps:
        print(1. / (time.time() - t0))
        t0 = time.time()

    if args.show_window:
        if args.show_fps:
            cv2.putText(img0,"FPS:{:.1f}".format(1. / (time.time() - t0)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 235), 4)
            #cv2.putText(img0, "lock:{:.1f}".format(lock_mode), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 0, 235), 4)
            #cv2.putText(img0, "team:{:.1f}".format(team_mode), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 235), 4)
            print(1. / (time.time() - t0))
            t0 = time.time()

        cv2.imshow('aim', img0)

        if args.top_most:
            hwnd = win32gui.FindWindow(None, 'aim')
            CVRECT = cv2.getWindowImageRect('aim')
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        cv2.waitKey(1)
    pidx(0)
    pidy(0)
    #cnt += 1