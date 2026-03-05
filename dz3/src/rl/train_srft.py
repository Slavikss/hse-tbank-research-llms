"""Single-stage SRFT training with supervised and RL signals."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.base.data import Data
from src.rl.config_utils import DEFAULT_TRAIN_CONFIG, load_config
from src.rl.reward import build_reward_func, format_prompt
from src.rl.srft_objective import compute_srft_loss, entropy_from_logits


def _load_rows(train_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    items = Data.from_jsonl_file(train_path)
    if limit is not None:
        items = items[:limit]

    rows: list[dict[str, Any]] = []
    for item in items:
        if not item.gold_completion:
            continue
        rows.append(
            {
                "question": item.question,
                "answer": item.answer,
                "metadata": item.metadata or {},
                "gold_completion": item.gold_completion,
            }
        )
    return rows


def _completion_nll(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    completion_text: str,
    max_length: int,
):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        prompt_text + completion_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    labels = full_ids.clone()
    prompt_len = min(len(prompt_ids), int(full_ids.shape[1]))
    labels[:, :prompt_len] = -100
    outputs = model(input_ids=full_ids, labels=labels)
    return outputs.loss, outputs.logits


def _run_dry(config: dict[str, Any]) -> None:
    import torch

    rows = _load_rows(config["data"]["train_path"], limit=int(config["dry_run"]["num_examples"]))
    if not rows:
        raise RuntimeError("SRFT dry run failed: no rows with gold trajectories")

    entropy = torch.tensor(2.0)
    loss = compute_srft_loss(
        sft_ce_loss=torch.tensor(1.0),
        rl_demo_loss=torch.tensor(0.2),
        positive_nll=torch.tensor(0.4),
        negative_logprob=torch.tensor(-0.1),
        entropy=entropy,
        w_sft_coef=float(config["srft"]["w_sft_coef"]),
        w_rl_coef=float(config["srft"]["w_rl_coef"]),
    )
    if torch.isnan(loss.total):
        raise RuntimeError("SRFT dry run produced NaN")

    curve_path = Path(config["data"]["curves_dir"]) / "srft_dry_run.jsonl"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.write_text(
        json.dumps(
            {
                "step": 0,
                "loss_total": float(loss.total.item()),
                "reward_mean": 0.0,
                "entropy": float(entropy.item()),
                "gen_len_mean": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"SRFT dry run completed on {len(rows)} examples")


def _train(config: dict[str, Any]) -> None:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing dependencies for SRFT (torch/transformers/peft)") from exc

    model_cfg = config["model"]
    srft_cfg = config["srft"]
    train_cfg = config["training"]

    rows = _load_rows(config["data"]["train_path"], limit=None)
    if not rows:
        raise RuntimeError("No SRFT rows found. Run gold generation first.")

    random.seed(int(srft_cfg["seed"]))
    torch.manual_seed(int(srft_cfg["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=True,
        device_map="auto",
    )
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_cfg["lora_r"]),
        lora_alpha=int(model_cfg["lora_alpha"]),
        lora_dropout=float(model_cfg["lora_dropout"]),
        target_modules=list(model_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.train()

    reward_fn = build_reward_func(log_samples=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(srft_cfg["learning_rate"]))
    steps = int(srft_cfg["steps"])
    batch_size = int(srft_cfg["batch_size"])
    max_length = int(train_cfg["max_prompt_length"] + train_cfg["max_completion_length"])

    curve_path = Path(config["data"]["curves_dir"]) / "srft_train.jsonl"
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    with curve_path.open("w", encoding="utf-8") as curve_file:
        for step in range(1, steps + 1):
            batch = random.sample(rows, k=min(batch_size, len(rows)))

            optimizer.zero_grad(set_to_none=True)

            sft_losses = []
            demo_rl_terms = []
            entropies = []
            reward_values = []
            gen_lengths = []
            positive_nll_terms = []
            negative_logprob_terms = []

            for row in batch:
                prompt = format_prompt(str(row["question"]))
                gold = str(row["gold_completion"])

                demo_nll, demo_logits = _completion_nll(model, tokenizer, prompt, gold, max_length)
                sft_losses.append(demo_nll)
                entropy = entropy_from_logits(demo_logits)
                entropies.append(entropy)

                demo_logprob = -demo_nll
                old_logprob = demo_logprob.detach()
                ratio = torch.exp(demo_logprob - old_logprob)
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(srft_cfg["clip_eps"]),
                    1.0 + float(srft_cfg["clip_eps"]),
                )
                demo_rl_terms.append(-torch.min(ratio, clipped))

                prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        prompt_ids,
                        do_sample=True,
                        temperature=float(train_cfg.get("temperature", 0.7)),
                        top_p=float(train_cfg.get("top_p", 0.95)),
                        max_new_tokens=int(train_cfg.get("max_tokens", 96)),
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_text = tokenizer.decode(
                    generated_ids[0][prompt_ids.shape[1] :],
                    skip_special_tokens=True,
                )
                gen_lengths.append(len(gen_text.split()))

                rewards = reward_fn(
                    completions=[gen_text],
                    answer=[str(row["answer"])],
                    metadata=[row["metadata"]],
                )
                reward = float(rewards[0])
                reward_values.append(reward)

                gen_nll, _ = _completion_nll(model, tokenizer, prompt, gen_text, max_length)
                if reward > 0:
                    positive_nll_terms.append(gen_nll)
                else:
                    negative_logprob_terms.append(-gen_nll)

            sft_loss = torch.stack(sft_losses).mean()
            rl_demo_loss = torch.stack(demo_rl_terms).mean()
            mean_entropy = torch.stack(entropies).mean()

            if positive_nll_terms:
                positive_nll = torch.stack(positive_nll_terms).mean()
            else:
                positive_nll = torch.tensor(0.0, device=model.device)

            if negative_logprob_terms:
                negative_logprob = torch.stack(negative_logprob_terms).mean()
            else:
                negative_logprob = torch.tensor(0.0, device=model.device)

            components = compute_srft_loss(
                sft_ce_loss=sft_loss,
                rl_demo_loss=rl_demo_loss,
                positive_nll=positive_nll,
                negative_logprob=negative_logprob,
                entropy=mean_entropy,
                w_sft_coef=float(srft_cfg["w_sft_coef"]),
                w_rl_coef=float(srft_cfg["w_rl_coef"]),
            )

            components.total.backward()
            optimizer.step()

            record = {
                "step": step,
                "loss_total": float(components.total.detach().cpu().item()),
                "loss_sft_demo": float(components.sft_demo.detach().cpu().item()),
                "loss_rl_demo": float(components.rl_demo.detach().cpu().item()),
                "loss_rl_self": float(components.rl_self_rollout.detach().cpu().item()),
                "w_sft": float(components.w_sft.detach().cpu().item()),
                "w_rl": float(components.w_rl.detach().cpu().item()),
                "reward_mean": sum(reward_values) / len(reward_values) if reward_values else 0.0,
                "entropy": float(mean_entropy.detach().cpu().item()),
                "gen_len_mean": sum(gen_lengths) / len(gen_lengths) if gen_lengths else 0.0,
            }
            curve_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            curve_file.flush()

            if step % 10 == 0 or step == 1:
                print(
                    f"[srft] step={step} total={record['loss_total']:.4f} "
                    f"reward={record['reward_mean']:.3f} entropy={record['entropy']:.3f}"
                )

    output_dir = Path(srft_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(output_dir))
    else:
        model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"SRFT training completed. Saved model to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SRFT single-stage objective")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, DEFAULT_TRAIN_CONFIG)
    if args.dry_run:
        _run_dry(config)
    else:
        _train(config)


if __name__ == "__main__":
    main()
