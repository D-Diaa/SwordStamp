"""Run sampling contract examples."""

import argparse

from transformers import GenerationConfig

from config.cli import add_config_args, resolve
from sampling.base_sampler import create_sampler, run_contract_examples
from segmentation import Segmenter


def parse_args():
    parser = argparse.ArgumentParser(description="Rejection-sampling contract self-test.")
    add_config_args(parser, positional_data=False)
    parser.add_argument("--prompt", type=str,
                        default="The astronaut stepped onto the surface of Mars and looked around in awe.")
    args = parser.parse_args()
    return resolve(args), args.prompt


def main():
    cfg, prompt = parse_args()
    gen = cfg.generation
    num_candidates = gen.num_candidates
    seg_type = cfg.segmentation.type
    seg_backend = cfg.segmentation.backend
    if seg_type == "semspan":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(cfg.watermark.embedder, device="cuda")
        segmenter = Segmenter.from_config(
            cfg.segmentation,
            encoder=encoder,
            encoder_id=cfg.watermark.embedder,
            batch_size=cfg.runtime.semcut_batch_size,
        )
    else:
        segmenter = Segmenter(seg_type, seg_backend)
    print(f"Initializing {gen.backend} sampler: {gen.model}")
    backend_kwargs = {"device": "cuda"} if gen.backend == "hf" else {}
    sampler = create_sampler(
        gen.backend,
        gen.model,
        num_candidates=num_candidates,
        chunk_tokens=gen.chunk_tokens,
        segmentation_type=seg_type,
        segmentation_backend=seg_backend,
        segmenter=segmenter,
        **backend_kwargs,
    )
    gen_config = GenerationConfig(
        max_new_tokens=gen.max_new_tokens,
        do_sample=gen.do_sample,
        temperature=gen.temperature,
        top_k=gen.top_k,
        top_p=gen.top_p,
        repetition_penalty=gen.rep_p,
        pad_token_id=sampler.tokenizer.pad_token_id,
    )
    run_contract_examples(sampler, gen_config, prompt)
    print("\nAll contract examples passed ✓")


if __name__ == "__main__":
    main()
