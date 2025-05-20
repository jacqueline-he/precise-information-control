from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
import time
from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from utils.print_utils import fancy_print
from pic_bench.metrics import calculate_scores
from omegaconf import DictConfig
import hydra
from rich.logging import RichHandler
from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TextColumn,
)
from rich import box
import logging

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("rich")

is_slurm = "SLURM_JOB_ID" in os.environ
console = Console(force_terminal=not is_slurm)
_cfg = None


def set_cfg(cfg):
    global _cfg
    _cfg = cfg


def log_stage(stage_name: str, elapsed: float):
    msg = Text()
    msg.append(f"{stage_name} complete", style="bold green")
    msg.append(f" — elapsed: {elapsed:.2f}s", style="dim")
    console.print(msg)


def print_eval_summary(fp, data):
    f1s = [d["results"]["f1"] for d in data]
    precs = [d["results"]["precision"] for d in data]
    recalls = [d["results"]["recall"] for d in data]
    nums = [d["results"]["num_unsupported"] for d in data]

    perfect_prec_count = sum(1 for x in precs if x == 1)
    perfect_f1_count = sum(1 for x in f1s if x == 1)

    table = Table(
        title="📊 PIC-Bench Scores",
        title_style="plum4",
        box=box.ROUNDED,
        show_lines=True,
    )

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("Macro-Avg. Precision", f"{sum(precs)/len(precs)*100:.1f}%")
    table.add_row("Macro-Avg. Recall@K", f"{sum(recalls)/len(recalls)*100:.1f}%")
    table.add_row("Macro-Avg. F1@K", f"{sum(f1s)/len(f1s)*100:.1f}%")
    table.add_row(
        "Perfect Precision Prop. (%)", f"{perfect_prec_count / len(precs) * 100:.1f}%"
    )
    table.add_row("Perfect F1 Prop. (%)", f"{perfect_f1_count / len(f1s) * 100:.1f}%")
    table.add_row("Avg. Unsupported Claims", f"{sum(nums)/len(nums):.1f}")
    console.print(table)
    console.print(f"[bold green]Results loaded from:[/bold green] [dim]{fp}[/dim]\n")


def init_ce_worker():
    from .claim_extractor import ClaimExtractor

    global ce
    ce = ClaimExtractor(
        model_name=_cfg.claim_extraction.model_name,
        use_cache=_cfg.claim_extraction.use_cache,
        cache_dir=_cfg.claim_extraction.cache_dir,
    )


def init_cv_worker():
    global cv
    from .claim_verifier import ClaimVerifier

    cv = ClaimVerifier(
        model_name=_cfg.claim_verification.model_name,
        use_cache=_cfg.claim_verification.use_cache,
        cache_dir=_cfg.claim_verification.cache_dir,
    )


def extract_claims(data_chunk, response_key=None, *args, **kwargs):
    global ce
    if ce is None:
        init_ce_worker()  # Ensure ClaimExtractor is initialized
    for d in data_chunk:
        try:
            response = d[response_key]
            fancy_print(console, f"Response: {response}")
            snippet_lst, claim_list, all_claims, prompt_tok_cnt, response_tok_cnt = (
                ce.scanner_extractor_fast(response)
            )
            d["resp_claims"] = all_claims
        except Exception as e:
            d["resp_claims"] = []
            fancy_print(
                console,
                f"Failed to extract claims for {d.get('topic', 'unknown entity')}: {e}",
            )
    return data_chunk


def verify_claims(data_chunk, *args, **kwargs):
    global cv
    if cv is None:
        init_cv_worker()
    for d in data_chunk:
        try:
            resp_claims = d["resp_claims"]
            prompt_claims = d["claims"]
            claim_verify_res_dict, _, _ = cv.verifying_claim(
                resp_claims=resp_claims, prompt_claims=prompt_claims
            )
            d["claim_verify_res_dict"] = claim_verify_res_dict
        except Exception as e:
            d["claim_verify_res_dict"] = []
            fancy_print(
                console,
                f"Failed to verify claims for {d.get('topic', 'unknown entity')}: {e}",
            )
    return data_chunk


def parallel_process(
    data, wrapped_func, init_worker=None, chunksize=100, max_workers=1
):
    """Efficient parallel processing with rich progress tracking."""
    results = []
    total_chunks = (len(data) + chunksize - 1) // chunksize

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        if hasattr(wrapped_func, "func"):
            func_name = wrapped_func.func.__name__
        elif hasattr(wrapped_func, "__name__"):
            func_name = wrapped_func.__name__
        else:
            func_name = wrapped_func.__class__.__name__
        submit_task = progress.add_task(
            f"[cyan]Submitting chunks for {func_name}...", total=total_chunks
        )
        process_task = progress.add_task(
            f"[green]Processing chunks for '{func_name}'...", total=total_chunks
        )

        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=init_worker
        ) as executor:
            futures = {}
            for i in range(0, len(data), chunksize):
                future = executor.submit(wrapped_func, data[i : i + chunksize])
                futures[future] = i
                progress.advance(submit_task)

            progress.update(submit_task, visible=False)

            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    progress.console.print(
                        f"[red]Parallel processing failed:[/red] {e}"
                    )
                progress.advance(process_task)

    return results


@hydra.main(
    config_path="../conf/eval", config_name="default_scoring", version_base="1.2"
)
def score_evals(cfg: DictConfig):
    start_time = time.time()
    set_cfg(cfg)
    fp = cfg.filepath
    if cfg.claim_extraction.max_num_samples is not None:
        fancy_print(
            console,
            f"Loading {cfg.claim_extraction.max_num_samples} sample(s) from {fp}",
        )
        data = load_jsonlines(fp)[: cfg.claim_extraction.max_num_samples]
    else:
        data = load_jsonlines(fp)
    ce_fp = fp.replace(".jsonl", "_ce.jsonl")
    if not os.path.exists(ce_fp) or cfg.override_claim_extraction:
        wrapped_extract_claims = partial(extract_claims, response_key=cfg.resp_key)
        data = parallel_process(
            data,
            wrapped_extract_claims,
            init_ce_worker,
            chunksize=cfg.chunksize,
            max_workers=cfg.max_workers,
        )
        save_file_jsonl(data, ce_fp)
    else:
        data = load_jsonlines(ce_fp)

    ce_end_time = time.time()
    log_stage("Claim extraction", ce_end_time - start_time)

    cv_fp = fp.replace(".jsonl", "_cv.jsonl")
    if not os.path.exists(cv_fp) or cfg.override_claim_verification:
        data = parallel_process(
            data,
            verify_claims,
            init_cv_worker,
            chunksize=cfg.chunksize,
            max_workers=cfg.max_workers,
        )
        save_file_jsonl(data, cv_fp)
    else:
        data = load_jsonlines(cv_fp)

    cv_end_time = time.time()
    log_stage("Claim verification", cv_end_time - ce_end_time)

    sc_fp = fp.replace(".jsonl", "_sc.jsonl")
    if not os.path.exists(sc_fp) or cfg.override_score_calculation:
        data = parallel_process(
            data,
            calculate_scores,
            init_worker=None,
            chunksize=cfg.claim_scoring.chunk_size,
            max_workers=cfg.claim_scoring.max_workers,
        )
        save_file_jsonl(data, sc_fp)
    else:
        data = load_jsonlines(sc_fp)

    sc_end_time = time.time()
    log_stage("Score calculation", sc_end_time - cv_end_time)

    print_eval_summary(fp, data)


if __name__ == "__main__":
    score_evals()
