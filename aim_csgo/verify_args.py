import torch

def verify_args(args):
    if args.use_cuda and torch.cuda.is_available() == False:
        print("--use-cuda 无GPU环境，请改为False")
        exit(0)

    if args.thickness < 1 / args.resize_window:
        print("--thickness 请注意参数要求！")
        exit(0)

    if not (0 < args.region[0] <= 1) and not (0 < args.region[1] <= 1):
        print("--region 请输入0~1的数！")
        exit(0)

    if args.lock_button not in ['left', 'middle', 'right', 'x1', 'x2']:
        print("--lock-button 只支持鼠标按键:left, middle, right, x1, x2")
        exit(0)

    for i in args.lock_choice:
        if i not in args.lock_tag:
            print("--lock-choice 请注意参数要求！")
            exit(0)

    buttons = []
    buttons.append(args.lock_button)
    # if args.recoil_button_ak47 not in ['left', 'middle', 'right', 'x1', 'x2']:
    #    print("--recoil-button-ak47 只支持鼠标按键:left, middle, right, x1, x2")
    #    exit(0)
    # if args.recoil_button_ak47 in buttons:
    #   print("--recoil-button-ak47 与其他按键冲突")
    #  exit(0)
    #buttons.append(args.recoil_button_ak47)

    # if args.recoil_button_m4a1 not in ['left', 'middle', 'right', 'x1', 'x2']:
    #     print("--recoil-button-m4a1 只支持鼠标按键:left, middle, right, x1, x2")
    #     exit(0)
    # if args.recoil_button_m4a1 in buttons:
    #     print("--recoil-button-m4a1 与其他按键冲突")
    #     exit(0)
    # buttons.append(args.recoil_button_m4a1)
