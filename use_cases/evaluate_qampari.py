import hydra
from omegaconf import DictConfig
from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from utils.text_utils import normalize_answer
import numpy as np
import os
import string
import re
import logging
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich import print as rprint

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")
console = Console()


def escape_rich_markup(obj):
    if isinstance(obj, str):
        return escape(obj)
    elif isinstance(obj, list):
        return [escape_rich_markup(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: escape_rich_markup(v) for k, v in obj.items()}
    else:
        return obj


def print_debug_examples(data, max_items=3):
    console.print("\n[bold underline]--- DEBUG EXAMPLES ---[/bold underline]")

    for item in data[:max_items]:
        dbg = item.get("debug", {})
        if not dbg:
            continue

        console.rule("[bold blue]Debug Example[/bold blue]", style="bright_blue")
        console.print(
            f"[bold cyan]Response:[/bold cyan] {escape_rich_markup(item['response'])}"
        )
        console.print(f"[bold yellow]Preds:[/bold yellow] {dbg['preds']}")
        console.print(f"[bold green]Correct preds:[/bold green] {dbg['correct_preds']}")
        console.print(f"[bold red]Wrong preds:[/bold red] {dbg['wrong_preds']}")
        console.print(
            f"[bold magenta]Gold answer groups:[/bold magenta] {dbg['gold_answers']}"
        )
        console.print(
            f"[bold red]Missed gold groups:[/bold red] {dbg['missed_gold_groups']}"
        )


def summarize_scores_ci(scores, confidence=0.95, n_bootstrap=10000):
    """
    Compute mean and bootstrap confidence interval for a list of scores.

    Args:
        scores (list or array): List of 0/1 or float scores.
        confidence (float): Confidence level (e.g., 0.95 for 95% CI).
        n_bootstrap (int): Number of bootstrap resamples.

    Returns:
        (mean_pct, ci_half_width_pct): Tuple of mean (%) and half-width of CI (%)
    """
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)

    if n <= 1:
        return round(100 * mean, 1), 0.0  # not enough data for CI

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    ci_half_width = (upper - lower) / 2

    return round(100 * mean, 1), round(100 * ci_half_width, 1)


def display_metrics(metrics, final_fp, eval_upper_bound=False):
    rprint("\n[bold yellow]Evaluation Results:[/bold yellow]")

    # Group metrics by base name (removing _ci suffix)
    base_metrics = {}
    for metric_name, values in metrics.items():
        if not metric_name.endswith("_ci"):
            base_metrics[metric_name] = values

    # Display metrics in order
    for metric_name, value in base_metrics.items():
        # If this is a pre-computed mean (float)
        if isinstance(value, (int, float)):
            ci_value = metrics.get(f"{metric_name}_ci", (0, 0))[
                1
            ]  # Get CI from _ci tuple
            display_val = f"{value:.2f} ± {ci_value:.2f}"
        else:
            # If this is a list of scores that needs CI computation
            mean, ci = summarize_scores_ci(value)
            display_val = f"{mean:.2f} ± {ci:.2f}"

        rprint(f"{metric_name}: {display_val}")

    rprint(f"\n[bold magenta]Final output file:[/bold magenta] {final_fp}")
    if eval_upper_bound:
        rprint("[bold red]Note: This is an upper bound evaluation[/bold red]")


def extract_after_colon(response: str) -> str:
    if ":" in response:
        return response.split(":", 1)[1].strip()
    else:
        return response.strip()


def extract_preds(response: str, gold_answers) -> list[str]:
    preds = set()

    response = extract_after_colon(response)
    for chunk in re.split(r"(?:[,;\n]|(?:\d+\)))", response):
        chunk = chunk.strip().strip(string.punctuation)
        chunk = re.sub(r"^(and|or)\s+", "", chunk.strip(), flags=re.IGNORECASE)
        chunk_norm = normalize_answer(chunk.strip())

        trailing_split = re.split(
            r"\b(is|are)( all| both)? correct answer(s)?( to this question)?\b",
            chunk_norm,
            flags=re.IGNORECASE,
        )
        if len(trailing_split) > 1:
            preds.add(trailing_split[0].strip())
            continue

        leading_match = re.search(
            r"\b(?:one|two|three|multiple|several)?\s*correct answer(s)?( to this question)? (is|are) (.+)",
            chunk_norm,
            flags=re.IGNORECASE,
        )
        if leading_match:
            preds.add(leading_match.group(4).strip())
            continue
        preds.add(chunk_norm)

    # Also add gold alias matches found in full response
    resp_norm = normalize_answer(response)
    for alias_group in gold_answers:
        for alias in alias_group:
            alias_norm = normalize_answer(alias)
            if re.search(rf"\b{re.escape(alias_norm)}\b", resp_norm):
                preds.add(alias_norm)

    return list(preds)


def compute_qampari_f1(cfg, data):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    if cfg.eval_upper_bound:
        print("Evaluating upper bound")

    num_preds = []
    for item in data:
        if len(item["correct_entities"]) == 0:
            continue  # Skip if self-verification LM can't produce any correct entities
        if cfg.eval_upper_bound:
            o = ", ".join(item["correct_entities"])
        else:
            o = item["response"]
        preds = extract_preds(o, item["gold_answers"])
        unwanted_substrings = ["correct answer"]

        preds = [
            p
            for p in preds
            if p
            and not any(bad in p for bad in unwanted_substrings)
            and p not in ["and", "or", "they"]
        ]
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item["gold_answers"]]
        flat_answers = [item for sublist in answers for item in sublist]

        prec.append(
            sum([p in flat_answers for p in preds]) / len(preds)
            if len(preds) > 0
            else 0
        )
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(
            min(5, sum([any([x in preds for x in a]) for a in answers]))
            / min(5, len(answers))
        )
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

        # DEBUG
        correct_preds = [p for p in preds if p in flat_answers]
        wrong_preds = [p for p in preds if p not in flat_answers]

        missed_gold_groups = [
            group for group in answers if not any(p in group for p in preds)
        ]
        item["debug"] = {
            "preds": preds,
            "correct_preds": correct_preds,
            "wrong_preds": wrong_preds,
            "gold_answers": answers,
            "missed_gold_groups": missed_gold_groups,
        }

    debug_fp = os.path.join(os.path.dirname(cfg.final_fp), "qampari_debug.jsonl")
    save_file_jsonl(data, debug_fp)

    print_debug_examples(data)

    return {
        "num_preds": round(np.mean(num_preds), 2),
        "num_preds_ci": summarize_scores_ci(num_preds),
        "qampari_prec": round(100 * np.mean(prec), 1),
        "qampari_prec_ci": summarize_scores_ci(prec),
        "qampari_rec": round(100 * np.mean(rec), 1),
        "qampari_rec_ci": summarize_scores_ci(rec),
        "qampari_rec_top5": round(100 * np.mean(rec_top5), 1),
        "qampari_rec_top5_ci": summarize_scores_ci(rec_top5),
        "qampari_f1": round(100 * np.mean(f1), 1),
        "qampari_f1_ci": summarize_scores_ci(f1),
        "qampari_f1_top5": round(100 * np.mean(f1_top5), 1),
        "qampari_f1_top5_ci": summarize_scores_ci(f1_top5),
    }


@hydra.main(
    config_path="../conf/use_cases",
    config_name="default_pipeline_qampari",
    version_base="1.2",
)
def evaluate_qampari(cfg: DictConfig):
    final_fp = cfg.final_fp
    assert os.path.exists(final_fp), f"Final output file {final_fp} does not exist."
    data = load_jsonlines(final_fp)

    metrics = compute_qampari_f1(cfg, data)

    display_metrics(metrics, final_fp, cfg.eval_upper_bound)


if __name__ == "__main__":
    evaluate_qampari()
