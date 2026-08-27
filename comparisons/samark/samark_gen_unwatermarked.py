import os
import json
import argparse

from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer


def clean_sentence(sentence):
    sentence = sentence.replace("\n", "").replace("\u201c", '"').replace("\u201d", '"').rstrip()
    endings = {".", "?", "!", '"'}
    if not sentence or sentence[-1] not in endings:
        sentence += "."
    return sentence


def extract_first_new_sentence(full_text, prompt):
    sentences = sent_tokenize(full_text)
    prompt_sentences = sent_tokenize(prompt)
    if len(sentences) > len(prompt_sentences):
        return clean_sentence(sentences[len(prompt_sentences)])
    return None


def generate_candidates_vllm(llm, prompt, max_tokens=128, temperature=0.7, top_p=0.95):
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    outputs = llm.generate([prompt], sampling_params, use_tqdm=False)

    for output in outputs:
        for gen in output.outputs:
            full_text = prompt + gen.text
            sentence = extract_first_new_sentence(full_text, prompt)
            if sentence:
                return sentence
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Unwatermarked generation for pairwise judge reference")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_name", type=str, default="booksum", choices=["c4", "booksum"],
                    help="legacy shorthand (ignored when --data_path is set)")
    p.add_argument("--data_path", type=str, default=None,
                    help="path to a HuggingFace dataset saved to disk (takes precedence over --data_name)")
    p.add_argument("--log_dir", type=str, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None,
                    help="exclusive end row; defaults to the full dataset length")
    p.add_argument("--min_new_tokens", type=int, default=205)
    p.add_argument("--min_new_sentences", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.data_path:
        dataset = load_from_disk(args.data_path)
    else:
        dataset = load_from_disk("./data/c4-val-500" if args.data_name == "c4" else "./data/booksum-train-500")

    if args.end is None:
        args.end = len(dataset)
    args.end = min(args.end, len(dataset))
    args.start = max(0, args.start)

    all_done = all(os.path.exists(f"{args.log_dir}/{i}.json") for i in range(args.start, args.end))
    if all_done:
        print(f"All samples [{args.start}, {args.end}) already exist in {args.log_dir}, skipping generation.")
        _export_hf_dataset(args.log_dir)
        return

    from vllm import LLM

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        seed=42,
        dtype="bfloat16",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(f"{args.log_dir}/config.json", "w", encoding="utf-8") as f:
        json.dump({**args.__dict__, "generation_type": "unwatermarked_reference"}, f, ensure_ascii=False, indent=4)

    for i in tqdm(range(args.start, args.end)):
        out_path = f"{args.log_dir}/{i}.json"
        if os.path.exists(out_path):
            continue

        text = dataset[i]["text"]
        prompt = sent_tokenize(text)[0]

        generated_text = ""
        generation_log = {"prompt": prompt, "generated_text": "", "original_text": text, "log": []}

        step = 0
        while True:
            step += 1
            next_sentence = generate_candidates_vllm(llm, prompt)
            if not next_sentence:
                break

            prompt = prompt + " " + next_sentence
            generated_text = generated_text + " " + next_sentence
            generation_log["generated_text"] = generated_text
            generation_log["log"].append({"text": next_sentence, "step": step})

            gen_ids = tokenizer(generated_text, return_tensors="pt")["input_ids"]
            if gen_ids.shape[1] > args.min_new_tokens and step >= args.min_new_sentences:
                break

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(generation_log, f, ensure_ascii=False, indent=4)

    _export_hf_dataset(args.log_dir)
    print("Done!")


def _export_hf_dataset(log_dir):
    """Export the per-sample JSON logs as an HF dataset with a 'text' column, matching
    samark_gen.py, so this corpus can be passed straight to samark_detect.py as the
    unwatermarked negative class (the paper's null -- see Fig. 7 and the 'None' row)."""
    rows = []
    for fname in sorted(os.listdir(log_dir)):
        if not (fname.endswith(".json") and fname[:-5].isdigit()):
            continue
        with open(f"{log_dir}/{fname}", "r", encoding="utf-8") as f:
            log = json.load(f)
        rows.append({
            "samark_idx":    int(fname[:-5]),
            "text":          (log["prompt"] + log["generated_text"]).strip(),
            "original_text": log["original_text"],
            "prompt":        log["prompt"],
        })
    if rows:
        rows.sort(key=lambda r: r["samark_idx"])
        Dataset.from_list(rows).save_to_disk(log_dir)
        print(f"Saved HF dataset ({len(rows)} samples) -> {log_dir}")


if __name__ == "__main__":
    main()
