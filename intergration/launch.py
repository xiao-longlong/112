import sys
import os
current_dir = os.getcwd()
sys.path.append(current_dir)

from cqx.main_cam import gesture_recognize
from ywj.client import speech_recognize
from zmj.wenben import context_recognition
import argparse

def make_parser():
    parser = argparse.ArgumentParser(description="projetc launch")
    parser.add_argument("--model", type=int, default=0, help="model") 
    return parser.parse_args()

def get_code():
    args = make_parser()
    # print(args.model)
    if args.model == 1:
        result_code = gesture_recognize()
    elif args.model == 2:
        result_code = speech_recognize()
    elif args.model == 3:
        result_code = context_recognition()
    else:
        print("model error")
    return result_code

if __name__ == "__main__":
    result_code = get_code()