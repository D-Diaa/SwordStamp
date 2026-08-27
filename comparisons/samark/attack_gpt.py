import json 
import os
import sys
import argparse
from tqdm import tqdm
from collections import Counter
from itertools import groupby
import re
from attack_utils import get_attacker


if __name__ == "__main__":      
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./logs/c4_logs/opt-1.3b_log")
    parser.add_argument("--sub_dir", type=str, default="KGW")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--attack_name", type=str, default="Doc-P I")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--prompt_attack", type=bool, default=True)
    args = parser.parse_args()

    log_dir=args.log_dir
    sub_dir=args.sub_dir
    attack_name=args.attack_name
    if attack_name in {"Doc-P I", "Doc-P II"}:
        _attack_dir_name = attack_name
    elif args.prompt_attack:
        _attack_dir_name=f"{attack_name}{args.ratio}-{int(args.prompt_attack)}"
    else:
        _attack_dir_name=f"{attack_name}{args.ratio}"

    # ── Early exit: skip if all samples in [start, end) already attacked ──
    _attack_out_dir = f"{log_dir}/{sub_dir}/attack/{_attack_dir_name}"
    all_done = all(
        os.path.exists(f"{_attack_out_dir}/{i}.json")
        for i in range(args.start, args.end)
    )
    if all_done:
        print(f"All samples [{args.start}, {args.end}) already attacked in {_attack_out_dir}, skipping.")
        sys.exit(0)

    attack=get_attacker(attack_name, args.ratio, "cuda")
    attack_name=_attack_dir_name
    print(args.start,args.end)
    for i in tqdm(range(args.start,args.end)):
        json_path=f"{log_dir}/{sub_dir}/{i}.json"
        if not os.path.exists(json_path):
            print(f"Sample {i} not found!")
            continue
        if os.path.exists(f"{log_dir}/{sub_dir}/attack/{attack_name}/{i}.json"):
            continue
        with open(json_path,"r",encoding="utf-8") as f:
            data=json.load(f)
        log=data["log"]
        text=data['generated_text']
        texts=[item['text'] for item in log]
        print(text)
        print("-----------------")
        data['unattacked_text']=data["generated_text"]
        edited=attack.edit(data["generated_text"])
        if "certain" in edited or "origin" in edited:
            edited=edited[edited.find(":"):]
        if edited is None or len(edited)<5:
            edited=data["generated_text"]
        else:
            edited=edited.replace("\n"," ")
        paraphrased_text=" "+edited
        if args.prompt_attack:# prompt attacked
            prompt_attacked=attack.edit(data['prompt'])
        data["generated_text"]=paraphrased_text
        print(data["generated_text"])
        para_path=f"{log_dir}/{sub_dir}/attack/{attack_name}/{i}.json"
        os.makedirs(os.path.dirname(para_path), exist_ok=True)
        with open(para_path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=4)
