import pynput
import csv
import time
import aim_csgo.ghub_mouse as ghub
from math import *

flag = 0


def lock(aims, mouse, top_x, top_y, len_x, len_y, args, pidx, pidy):
    mouse_pos_x, mouse_pos_y = mouse.position
    aims_copy = aims.copy()
    detect_arange = 15000
    aims_copy = [x for x in aims_copy if x[0] in args.lock_choice and (len_x * float(x[1]) + top_x - mouse_pos_x) ** 2 + (len_y * float(x[2]) + top_y - mouse_pos_y) ** 2 < detect_arange]
    k = 4.07 * (1 / args.lock_smooth)
    if len(aims_copy):
        dist_list = []
        tag_list = [x[0] for x in aims_copy]
        '''if args.head_first:
            if args.lock_tag[0] in tag_list:  # 有头
                aims_copy = [x for x in aims_copy if x[0] in [args.lock_tag[0]]]
                '''
        for det in aims_copy:
            _, x_c, y_c, _, _ = det
            dist = (len_x * float(x_c) + top_x - mouse_pos_x) ** 2 + (len_y * float(y_c) + top_y - mouse_pos_y) ** 2
            dist_list.append(dist)
            #print("dist", dist_list, "x_center", x_c, "y_center", y_c, "mouse", mouse.position)

        if dist_list:
            det = aims_copy[dist_list.index(min(dist_list))]
            tag, x_center, y_center, width, height = det
            # print(det)
            x_center, width = len_x * float(x_center) + top_x, len_x * float(width)
            y_center, height = len_y * float(y_center) + top_y, len_y * float(height)
            rel_x = int(k / args.lock_sen * atan((mouse_pos_x - x_center) / 640) * 640)
            rel_y = int(k / args.lock_sen * atan((mouse_pos_y - y_center + 1 / 4 * height) / 640) * 640)
            pid_movex = pidx(rel_x)
            pid_movey = pidy(rel_y)
            ghub.mouse_xy(round(pid_movex), round(pid_movey))
            # ghub.mouse_xy(-rel_x, -rel_y)


def recoil_control(args):
    global flag
    ak47_recoil = []
    m4a1_recoil = []
    m4a4_recoil = []
    galil_recoil = []
    famas_recoil = []
    aug_recoil = []
    bizon_recoil = []
    cz75_recoil = []
    m249_recoil = []
    mac10_recoil = []
    mp5sd_recoil = []
    mp7_recoil = []
    mp9_recoil = []
    p90_recoil = []
    sg553_recoil = []
    ump45_recoil = []

    for i in csv.reader(open('aim_csgo/ammo_path/ak47.csv', encoding='utf-8-sig')):
        ak47_recoil.append([float(x) for x in i])
        print(ak47_recoil)
    for i in csv.reader(open('./aim_csgo/ammo_path/m4a1.csv', encoding='utf-8-sig')):
        m4a1_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/m4a4.csv', encoding='utf-8-sig')):
        m4a4_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/galil.csv', encoding='utf-8-sig')):
        galil_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/famas.csv', encoding='utf-8-sig')):
        famas_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/aug.csv', encoding='utf-8-sig')):
        aug_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/bizon.csv', encoding='utf-8-sig')):
        bizon_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/cz75.csv', encoding='utf-8-sig')):
        cz75_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/m249.csv', encoding='utf-8-sig')):
        m249_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/mac10.csv', encoding='utf-8-sig')):
        mac10_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/mp5sd.csv', encoding='utf-8-sig')):
        mp5sd_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/mp7.csv', encoding='utf-8-sig')):
        mp7_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/mp9.csv', encoding='utf-8-sig')):
        mp9_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/p90.csv', encoding='utf-8-sig')):
        p90_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/sg553.csv', encoding='utf-8-sig')):
        sg553_recoil.append([float(x) for x in i])
    for i in csv.reader(open('./aim_csgo/ammo_path/ump45.csv', encoding='utf-8-sig')):
        ump45_recoil.append([float(x) for x in i])

    k = -args.recoil_sen
    recoil_mode = False

    with pynput.mouse.Events() as events:
        for event in events:
            if isinstance(event, pynput.mouse.Events.Click):
                if event.button == event.button.left:
                    if event.pressed:
                        flag = 1
                    else:
                        flag = 0
                if event.button == eval('event.button.' + args.recoil_button_ak47) and event.pressed:
                    recoil_mode = not recoil_mode
                    print('recoil mode', 'on' if recoil_mode else 'off')

            if flag and recoil_mode:
                i = 0
                a = next(events)
                while True:
                    ghub.mouse_xy(int(-ak47_recoil[i][0] * k), int(ak47_recoil[i][1] * k))
                    time.sleep(ak47_recoil[i][2] / 1000 - 0.01)
                    i += 1
                    if i == 30:
                        break
                    if a is not None and isinstance(a,
                                                    pynput.mouse.Events.Click) and a.button == a.button.left and not a.pressed:
                        break
                    a = next(events)
                    while a is not None and not isinstance(a, pynput.mouse.Events.Click):
                        a = next(events)
                flag = 0
