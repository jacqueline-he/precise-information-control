import hydra
from omegaconf import DictConfig
import os
from pic_bench.claim_verifier import ClaimVerifier
from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from utils.print_utils import fancy_print
from prompts.veriscore_prompts import VERISCORE_PROMPT
from use_cases.search_api import SearchAPI
import numpy as np
from rich.console import Console
from rich.progress import track
from rich import print as rprint

console = Console()


def debug_print(msg, data=None):
    rprint(f"[bold blue]DEBUG:[/bold blue] {msg}")
    if data is not None:
        if isinstance(data, (list, dict)):
            rprint(f"[dim]Data sample:[/dim]")
            rprint(data if isinstance(data, dict) else data[:2])
        else:
            rprint(f"[dim]Value:[/dim] {data}")


def get_ents_to_inds(data, eval_upper_bound=False):
    ents_to_inds = []
    for i, d in enumerate(data):
        if eval_upper_bound:
            entities = d["correct_entities"]
        else:
            entities = d["fast_eval"]["predicted_entities"]
        for _ in entities:
            ents_to_inds.append(i)
    return ents_to_inds


def build_prompts(data, eval_upper_bound=False):
    prompts = []
    for d in data:
        occupation = d["occupation"]
        loc = d["loc"]
        if eval_upper_bound:
            entities = d["correct_entities"]
        else:
            entities = d["fast_eval"]["predicted_entities"]
        for entity in entities:
            prompt = f"Is {entity} a {occupation} born in {loc}?"
            prompts.append(prompt)
    return prompts


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


@hydra.main(
    config_path="../conf/use_cases",
    config_name="default_pipeline_birthplace",
    version_base="1.2",
)
def evaluate_birthplace(cfg: DictConfig):
    final_fp = cfg.final_fp
    assert os.path.exists(final_fp), f"Final output file {final_fp} does not exist."

    debug_print("Loading data from", final_fp)
    data = load_jsonlines(final_fp)
    debug_print("Number of records loaded", len(data))

    ents_to_inds = get_ents_to_inds(
        data, eval_upper_bound=cfg.evaluate.eval_upper_bound
    )
    debug_print("Number of entities to evaluate", len(ents_to_inds))

    if cfg.evaluate.eval_upper_bound:
        fancy_print(console, "Evaluating upper bound...")
        verify_fp = final_fp.replace(".jsonl", "_verify_ub.jsonl")
        snippet_fp = final_fp.replace(".jsonl", "_snippet_ub.jsonl")
    else:
        verify_fp = final_fp.replace(".jsonl", "_verify.jsonl")
        snippet_fp = final_fp.replace(".jsonl", "_snippet.jsonl")

    if cfg.evaluate.search:
        prompts = build_prompts(data, eval_upper_bound=cfg.evaluate.eval_upper_bound)
        fancy_print(console, f"Example prompt: {prompts[0]}")
        api = SearchAPI(cache_file=cfg.evaluate.cache_file)
        results, failed_queries, empty_results = api.get_snippets(prompts, console)

        # Track which entities had search issues
        entities_with_issues = []
        for idx, _, _ in failed_queries:
            data_idx = ents_to_inds[idx]
            if cfg.evaluate.eval_upper_bound:
                entities = data[data_idx].get("correct_entities", [])
            else:
                entities = (
                    data[data_idx].get("fast_eval", {}).get("predicted_entities", [])
                )
            entities_with_issues.append(
                {
                    "index": idx,
                    "data_idx": data_idx,
                    "entities": entities,
                    "occupation": data[data_idx].get("occupation", "N/A"),
                    "location": data[data_idx].get("loc", "N/A"),
                    "issue": "failed_search",
                }
            )

        for idx, _ in empty_results:
            data_idx = ents_to_inds[idx]
            if cfg.evaluate.eval_upper_bound:
                entities = data[data_idx].get("correct_entities", [])
            else:
                entities = (
                    data[data_idx].get("fast_eval", {}).get("predicted_entities", [])
                )
            entities_with_issues.append(
                {
                    "index": idx,
                    "data_idx": data_idx,
                    "entities": entities,
                    "occupation": data[data_idx].get("occupation", "N/A"),
                    "location": data[data_idx].get("loc", "N/A"),
                    "issue": "no_search_results",
                }
            )

        if entities_with_issues:
            rprint("\n[bold red]Entities with Search Issues:[/bold red]")
            for entity in entities_with_issues:
                rprint(
                    f"\nEntity at index {entity['index']} (data_idx {entity['data_idx']}):"
                )
                rprint(f"- Issue: {entity['issue']}")
                rprint(f"- Entities: {entity['entities']}")
                rprint(f"- Occupation: {entity['occupation']}")
                rprint(f"- Location: {entity['location']}")

        save_file_jsonl([results], snippet_fp)

    if cfg.evaluate.verify:
        debug_print("Loading snippets from", snippet_fp)
        snippets = load_jsonlines(snippet_fp)[0]
        debug_print("Number of snippets loaded", len(snippets))

        cv = ClaimVerifier(
            model_name=cfg.evaluate.claim_verification.model_name,
            use_cache=cfg.evaluate.claim_verification.use_cache,
            cache_dir=cfg.evaluate.claim_verification.cache_dir,
        )

        claim_verify_res_dict, _, _ = cv.verifying_claim_snippets(
            snippets, search_res_num=10, prompt_initial_temp=VERISCORE_PROMPT
        )

        # Track verification process
        verification_stats = {
            "total_attempts": len(ents_to_inds),
            "successful_verifications": len(claim_verify_res_dict),
            "missing_verifications": len(ents_to_inds) - len(claim_verify_res_dict),
        }

        debug_print("Verification statistics", verification_stats)

        if verification_stats["missing_verifications"] > 0:
            rprint("\n[bold red]Warning: Missing verifications[/bold red]")
            rprint(f"Expected {verification_stats['total_attempts']} verifications")
            rprint(f"Got {verification_stats['successful_verifications']} results")
            rprint(
                f"Missing {verification_stats['missing_verifications']} verifications"
            )

            # Show which entities are missing verification
            missing_start_idx = verification_stats["successful_verifications"]
            for i in range(missing_start_idx, len(ents_to_inds)):
                data_idx = ents_to_inds[i]
                if cfg.evaluate.eval_upper_bound:
                    entities = data[data_idx].get("correct_entities", [])
                else:
                    entities = (
                        data[data_idx]
                        .get("fast_eval", {})
                        .get("predicted_entities", [])
                    )
                rprint(f"\nMissing verification for entity at index {i}:")
                rprint(f"- Data index: {data_idx}")
                rprint(f"- Entities: {entities}")
                rprint(f"- Occupation: {data[data_idx].get('occupation', 'N/A')}")
                rprint(f"- Location: {data[data_idx].get('loc', 'N/A')}")

        # Process only the verifications we have
        for i, data_idx in enumerate(ents_to_inds[: len(claim_verify_res_dict)]):
            try:
                if "verifications" not in data[data_idx]:
                    data[data_idx]["verifications"] = []
                data[data_idx]["verifications"].append(claim_verify_res_dict[i])
            except Exception as e:
                rprint(
                    f"[bold red]Error processing verification at index {i}:[/bold red]"
                )
                rprint(f"- Error: {str(e)}")
                rprint(f"- Data index: {data_idx}")
                pdb.set_trace()

        save_file_jsonl(data, verify_fp)
        fancy_print(console, f"Saved claim verification results to {verify_fp}")

    if cfg.evaluate.score:
        debug_print("Loading verification data from", verify_fp)
        verify_data = load_jsonlines(verify_fp)
        debug_print("Number of verification records", len(verify_data))

        # Compute atomic factual precision
        precs = []
        imperfect_samples = []
        for i, d in enumerate(
            track(verify_data, description="Scoring factual precision...")
        ):
            verifications = d.get("verifications", [])
            num_supported = sum(
                v["verification_result"] == "supported" for v in verifications
            )
            prec = num_supported / len(verifications) if verifications else 0
            precs.append(prec)

            # Collect samples with non-perfect precision
            if prec != 1.0:
                sample_info = {
                    "index": i,
                    "precision": prec,
                    "occupation": d.get("occupation", "N/A"),
                    "location": d.get("loc", "N/A"),
                    "entities": (
                        d.get("fast_eval", {}).get("predicted_entities", [])
                        if not cfg.evaluate.eval_upper_bound
                        else d.get("correct_entities", [])
                    ),
                    "num_supported": num_supported,
                    "total_verifications": len(verifications),
                    "verification_details": [
                        {
                            "result": v["verification_result"],
                        }
                        for v in verifications
                    ],
                }
                imperfect_samples.append(sample_info)

        avg_prec, ci_95 = summarize_scores_ci(precs)

        rprint(
            f"\n[bold yellow]Factual Precision:[/bold yellow] {avg_prec:.1f}% ± {ci_95:.1f}% (95% CI)"
        )
        rprint(f"[bold magenta]Final output file:[/bold magenta] {final_fp}")


if __name__ == "__main__":
    evaluate_birthplace()
