#!/usr/bin/env python3
"""
qwen_chat.py
============
Interactive chat with Qwen2.5 models using TurboQuant (weights + KV cache).

Usage
-----
# KV-cache TurboQuant only (no offline quantisation needed):
python examples/qwen_chat.py --model mlx-community/Qwen2.5-14B-Instruct-4bit

# Target: Qwen2.5-32B (best for 16 GB with TQ weights):
python examples/qwen_chat.py \\
    --model ~/tq_models/Qwen2.5-32B-tq4bit --tq-weights --kv-bits 3

# Full mode, full options:
python examples/qwen_chat.py \\
    --model mlx-community/Qwen2.5-14B-Instruct-4bit \\
    --kv-bits 3 --max-tokens 512 --temperature 0.7 --verbose

Memory guide (16 GB MacBook Air):
  14B (4-bit mlx + TQ 3-bit KV):  ~9 GB + 0.3 GB  -> fits easily
  32B (TQ4-bit W + TQ 3-bit KV):  ~13 GB + 0.5 GB -> fits with 2 GB headroom
  32B raw 4-bit mlx:               ~18 GB          -> too large without TQ weights
"""

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant MLX Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-14B-Instruct-4bit")
    parser.add_argument("--kv-bits",    type=int,   default=3)
    parser.add_argument("--tq-weights", action="store_true",
                        help="Load from a pre-quantised TurboQuant weight model")
    parser.add_argument("--weight-bits",type=int,   default=4)
    parser.add_argument("--max-tokens", type=int,   default=512)
    parser.add_argument("--temperature",type=float, default=0.7)
    parser.add_argument("--top-p",      type=float, default=0.9)
    parser.add_argument("--no-stream",  action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   TurboQuant MLX Chat               ║")
    print("  ╚══════════════════════════════════════╝\n")

    from mlx_lm import load as mlx_load
    from turboquant_mlx_full import (
        make_turboquant_cache, tq_generate, stream_tq_generate, print_memory_usage,
    )

    print(f"  Loading: {args.model}")
    if args.tq_weights:
        from turboquant_mlx_full.patch import load_turboquant_model
        model, tokenizer = load_turboquant_model(args.model, kv_bits=args.kv_bits)
        cache = getattr(model, "_tq_caches", None)
    else:
        model, tokenizer = mlx_load(args.model)
        print(f"  Attaching {args.kv_bits}-bit TurboQuant KV cache...")
        cache = make_turboquant_cache(model, kv_bits=args.kv_bits)
        print(f"  Ready: {len(cache)} layers | head_dim={cache[0].head_dim} | "
              f"n_kv_heads={cache[0].n_kv_heads}")

    print_memory_usage()
    print(f"\n  kv_bits={args.kv_bits}  max_tokens={args.max_tokens}  "
          f"temperature={args.temperature}")
    print("  Commands: /reset  /mem  /help  quit\n")
    print("  " + "─" * 40)

    history = []

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\n  Goodbye!")
            break
        if user_input == "/reset":
            history.clear()
            from turboquant_mlx_full.kv_cache import make_turboquant_cache as _mkc
            cache = _mkc(model, kv_bits=args.kv_bits)
            print("  [History and cache cleared]")
            continue
        if user_input == "/mem":
            print_memory_usage()
            continue
        if user_input == "/help":
            print("  /reset  clear chat history and KV cache")
            print("  /mem    show memory usage")
            print("  quit    exit")
            continue

        history.append({"role": "user", "content": user_input})

        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    history, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = user_input
        else:
            prompt = user_input

        print("\n  Assistant: ", end="", flush=True)

        if args.no_stream:
            response = tq_generate(
                model, tokenizer, prompt, cache=cache,
                max_tokens=args.max_tokens, temperature=args.temperature,
                top_p=args.top_p, verbose=args.verbose,
            )
            print(response)
            history.append({"role": "assistant", "content": response})
        else:
            full_response = ""
            for token in stream_tq_generate(
                model, tokenizer, prompt, cache=cache,
                max_tokens=args.max_tokens, temperature=args.temperature,
            ):
                print(token, end="", flush=True)
                full_response += token
            if args.verbose:
                print(); print_memory_usage()
            history.append({"role": "assistant", "content": full_response})

        print()


if __name__ == "__main__":
    main()
