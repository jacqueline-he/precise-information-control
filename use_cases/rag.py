from omegaconf import DictConfig
import hydra
import os
from vllm import LLM, SamplingParams
from utils.jsonl_utils import load_jsonlines, save_file_jsonl
from utils.prompt_utils import format_claims, load_prompts
from utils.text_utils import exact_presence
from utils.print_utils import fancy_print
import numpy as np
import logging
from scipy.stats import sem, t
import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from rich import box

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")
console = Console()

# hard-coded oracle indices (885)
ORACLE_INDS = [
    0,
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    15,
    16,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    33,
    34,
    35,
    36,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    123,
    124,
    125,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    153,
    154,
    155,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
    168,
    169,
    170,
    172,
    173,
    175,
    177,
    178,
    179,
    180,
    181,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    208,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    230,
    231,
    233,
    234,
    235,
    236,
    237,
    238,
    239,
    240,
    241,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    250,
    251,
    252,
    253,
    256,
    257,
    258,
    259,
    260,
    261,
    263,
    265,
    266,
    267,
    269,
    270,
    271,
    272,
    273,
    274,
    275,
    276,
    277,
    278,
    279,
    280,
    282,
    283,
    284,
    285,
    286,
    287,
    288,
    289,
    290,
    291,
    292,
    293,
    294,
    295,
    296,
    297,
    298,
    299,
    300,
    302,
    304,
    305,
    306,
    307,
    308,
    309,
    310,
    311,
    312,
    313,
    314,
    316,
    317,
    318,
    319,
    320,
    321,
    322,
    323,
    324,
    325,
    326,
    328,
    329,
    330,
    331,
    332,
    333,
    334,
    335,
    336,
    337,
    338,
    340,
    341,
    342,
    343,
    344,
    345,
    346,
    347,
    348,
    349,
    350,
    351,
    352,
    353,
    354,
    355,
    356,
    357,
    358,
    359,
    361,
    362,
    363,
    364,
    365,
    366,
    367,
    369,
    370,
    371,
    372,
    373,
    375,
    376,
    377,
    378,
    379,
    380,
    381,
    382,
    383,
    384,
    385,
    386,
    387,
    388,
    389,
    390,
    391,
    392,
    394,
    395,
    396,
    397,
    398,
    399,
    400,
    401,
    402,
    403,
    404,
    405,
    406,
    407,
    408,
    409,
    410,
    411,
    412,
    413,
    414,
    415,
    416,
    417,
    418,
    419,
    420,
    421,
    422,
    423,
    424,
    425,
    426,
    428,
    429,
    430,
    431,
    432,
    433,
    434,
    435,
    436,
    437,
    438,
    439,
    440,
    441,
    442,
    443,
    444,
    445,
    446,
    447,
    448,
    449,
    450,
    451,
    453,
    454,
    455,
    456,
    457,
    458,
    460,
    461,
    462,
    463,
    464,
    465,
    466,
    467,
    469,
    470,
    471,
    472,
    473,
    474,
    475,
    476,
    477,
    478,
    479,
    480,
    481,
    482,
    483,
    484,
    485,
    487,
    488,
    489,
    490,
    491,
    492,
    493,
    494,
    495,
    496,
    497,
    499,
    500,
    501,
    502,
    503,
    504,
    505,
    506,
    507,
    508,
    509,
    510,
    511,
    512,
    513,
    514,
    515,
    516,
    517,
    518,
    519,
    520,
    521,
    522,
    523,
    524,
    525,
    526,
    527,
    528,
    529,
    531,
    533,
    534,
    535,
    536,
    537,
    538,
    539,
    540,
    541,
    542,
    543,
    544,
    545,
    546,
    547,
    548,
    549,
    550,
    551,
    552,
    553,
    554,
    555,
    556,
    557,
    558,
    560,
    561,
    562,
    563,
    564,
    565,
    566,
    567,
    568,
    569,
    570,
    571,
    572,
    573,
    574,
    575,
    576,
    577,
    579,
    580,
    581,
    582,
    583,
    584,
    585,
    586,
    587,
    588,
    589,
    590,
    591,
    592,
    593,
    594,
    595,
    597,
    598,
    599,
    600,
    601,
    602,
    603,
    604,
    605,
    606,
    607,
    609,
    610,
    611,
    612,
    613,
    614,
    615,
    616,
    617,
    618,
    619,
    620,
    621,
    622,
    623,
    624,
    625,
    626,
    627,
    628,
    629,
    630,
    631,
    632,
    633,
    634,
    635,
    636,
    637,
    638,
    639,
    640,
    641,
    642,
    643,
    644,
    645,
    646,
    647,
    648,
    649,
    650,
    651,
    652,
    653,
    654,
    655,
    656,
    657,
    658,
    659,
    661,
    662,
    663,
    664,
    665,
    666,
    667,
    668,
    669,
    670,
    671,
    672,
    673,
    674,
    675,
    677,
    678,
    679,
    680,
    681,
    683,
    684,
    685,
    686,
    687,
    688,
    689,
    690,
    691,
    692,
    693,
    694,
    695,
    697,
    698,
    699,
    700,
    701,
    702,
    703,
    704,
    705,
    707,
    708,
    709,
    710,
    711,
    712,
    713,
    714,
    716,
    717,
    718,
    719,
    720,
    721,
    722,
    723,
    724,
    726,
    727,
    728,
    729,
    730,
    731,
    732,
    733,
    734,
    735,
    736,
    737,
    738,
    739,
    740,
    741,
    742,
    743,
    744,
    745,
    746,
    747,
    748,
    749,
    750,
    751,
    752,
    753,
    754,
    755,
    756,
    757,
    758,
    759,
    760,
    761,
    762,
    763,
    764,
    766,
    768,
    769,
    770,
    771,
    772,
    773,
    774,
    775,
    776,
    777,
    778,
    779,
    780,
    781,
    782,
    783,
    784,
    785,
    786,
    787,
    788,
    789,
    790,
    791,
    792,
    793,
    794,
    795,
    796,
    797,
    798,
    799,
    800,
    802,
    803,
    804,
    805,
    806,
    807,
    808,
    809,
    810,
    811,
    812,
    813,
    814,
    815,
    816,
    817,
    818,
    819,
    820,
    821,
    822,
    823,
    824,
    826,
    827,
    829,
    830,
    831,
    832,
    833,
    835,
    837,
    838,
    839,
    840,
    841,
    842,
    843,
    844,
    845,
    846,
    847,
    848,
    849,
    850,
    851,
    852,
    853,
    854,
    855,
    856,
    857,
    858,
    859,
    860,
    862,
    863,
    864,
    865,
    866,
    867,
    868,
    869,
    870,
    871,
    872,
    873,
    875,
    876,
    877,
    878,
    879,
    880,
    881,
    882,
    883,
    884,
    885,
    886,
    887,
    888,
    889,
    890,
    891,
    892,
    893,
    895,
    896,
    897,
    898,
    899,
    900,
    901,
    902,
    904,
    905,
    906,
    907,
    908,
    909,
    910,
    911,
    912,
    913,
    914,
    915,
    916,
    918,
    919,
    920,
    921,
    922,
    923,
    924,
    925,
    926,
    927,
    928,
    929,
    930,
    931,
    932,
    933,
    934,
    935,
    936,
    937,
    938,
    939,
    940,
    941,
    942,
    943,
    944,
    945,
    947,
]


def display_str_em_results(
    output_em, output_em_hit, oracle_em, oracle_em_hit, model_name
):
    console = Console()

    table = Table(
        title="📊 RAG STR-EM",
        title_style="plum4",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric", style="magenta", justify="left")
    table.add_column("Mean ± 95% CI", style="green", justify="right")

    # Unpack values
    em_mean, em_ci = output_em
    hit_mean, hit_ci = output_em_hit
    oracle_em_mean, oracle_em_ci = oracle_em
    oracle_hit_mean, oracle_hit_ci = oracle_em_hit

    # Add rows
    table.add_row("Output EM", f"{em_mean:.1f} ± {em_ci:.1f}")
    table.add_row("Output EM Hit", f"{hit_mean:.1f} ± {hit_ci:.1f}")
    table.add_row("Oracle Output EM", f"{oracle_em_mean:.1f} ± {oracle_em_ci:.1f}")
    table.add_row(
        "Oracle Output EM Hit", f"{oracle_hit_mean:.1f} ± {oracle_hit_ci:.1f}"
    )

    console.print(table)
    fancy_print(console, f"Model name: {model_name}")


def summarize_scores_ci(scores, confidence=0.95):
    scores = np.array(scores)
    n = len(scores)
    mean = np.mean(scores)
    h = sem(scores) * t.ppf((1 + confidence) / 2.0, n - 1)
    return round(100 * mean, 1), round(100 * h, 1)


def compute_str_em(data, debug_path=None):
    if "qa_pairs" not in data[0] or data[0]["qa_pairs"] is None:
        return 0, 0

    acc = []
    hit = []

    for idx, item in enumerate(track(data, description="Processing items...")):
        loc_acc = []
        debug_info = []

        resp = item["output"].lower()  # Case normalization
        context = item.get("instruction", "")
        for i, qa_pair in enumerate(item["qa_pairs"]):
            short_answers = [
                ans.lower() for ans in qa_pair["short_answers"]
            ]  # Case normalization
            match = exact_presence(short_answers, resp)

            context_presence = any(ans in context for ans in short_answers)

            loc_acc.append(match)

            debug_info.append(
                {
                    "short_answers": short_answers,
                    "matched": bool(match),
                    "present_in_context": context_presence,
                    "resp": resp,
                }
            )

        item_score = np.mean(loc_acc)
        acc.append(item_score)
        hit_val = int(item_score == 1)
        hit.append(hit_val)

        # Save detailed match info into item
        item["str_em_debug"] = {
            "score": item_score,
            "hit": hit_val,
            "matches": debug_info,
        }

    fancy_print(console, f"\nProcessed {len(acc)} items.")

    # Save debug info if requested
    if debug_path:
        save_file_jsonl(data, debug_path)
        fancy_print(console, f"Saved detailed STR-EM debug info to {debug_path}")

    mean_acc, ci_acc = summarize_scores_ci(acc)
    mean_hit, ci_hit = summarize_scores_ci(hit)

    return (mean_acc, ci_acc), (mean_hit, ci_hit)


def compute_str_em_oracle(data):
    acc = []
    hit = []

    for i, item in enumerate(track(data, description="Processing items...")):
        if i not in ORACLE_INDS:
            continue

        resp = item["output"].lower()  # Case normalization
        loc_acc = [
            exact_presence(
                [ans.lower() for ans in qa["short_answers"]], resp
            )  # Case normalization
            for qa in item["qa_pairs"]
        ]
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    mean_acc, ci_acc = summarize_scores_ci(acc)
    mean_hit, ci_hit = summarize_scores_ci(hit)

    return (mean_acc, ci_acc), (mean_hit, ci_hit)


def load_vllm(cfg):
    with console.status("[plum4]Loading LLM...", spinner="aesthetic"):
        llm = LLM(
            model=cfg.model_name_or_path,
            download_dir=cfg.download_dir,
            tensor_parallel_size=cfg.tensor_parallel_size,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_batched_tokens=cfg.max_num_batched_tokens,
            swap_space=cfg.swap_space,
            dtype=torch.bfloat16,
            disable_custom_all_reduce=cfg.disable_custom_all_reduce,
        )

        tokenizer = llm.get_tokenizer()

        sampling_params = SamplingParams(
            max_tokens=cfg.sampling.max_tokens,
            temperature=cfg.sampling.temperature,
            repetition_penalty=cfg.sampling.repetition_penalty,
            seed=cfg.sampling.seed,
            stop=cfg.sampling.stop_tokens + [tokenizer.eos_token],
        )
    return llm, sampling_params


def create_rag_prompts(cfg, data):
    PROMPT_DICT = load_prompts()
    prompts = []
    all_qa_pairs = []
    template = "".join(PROMPT_DICT[f"pic_{cfg.pic_type}_prompt"])
    sfx = cfg.instr_sfx
    for d in track(data):
        instruction = d["instruction"] + sfx
        if cfg.passage_level:
            formatted_claims = d["context"]
        else:
            formatted_claims = format_claims(d["claims"])
        prompt = template.format(instruction=instruction, claims=formatted_claims)
        prompts.append(prompt)
        all_qa_pairs.append(d["qa_pairs"])
    return prompts, all_qa_pairs


@hydra.main(
    config_path="../conf/use_cases", config_name="default_rag_asqa", version_base="1.2"
)
def rag(cfg: DictConfig):
    mode = cfg.mode
    if mode not in ["generate", "evaluate", "all"]:
        raise ValueError(
            f"Invalid mode: {mode}. Choose from 'generate', 'evaluate', or 'all'."
        )

    if mode == "generate" or mode == "all":
        # rag generations:
        llm, sampling_params = load_vllm(cfg)

        assert os.path.exists(cfg.data_path)
        data = load_jsonlines(cfg.data_path)

        prompts, all_qa_pairs = create_rag_prompts(cfg, data)

        fancy_print(console, f"Ex. Prompt: {prompts[0]}")

        outputs = llm.generate(prompts, sampling_params)
        generations = [o.outputs[0].text for o in outputs]

        all_claims = [d["claims"] for d in data]

        generated_data = [
            {
                "instruction": prompt,
                "claims": claim,
                "output": generation,
                "qa_pairs": qa_pairs,
                "generator": cfg.model_name_or_path,
            }
            for prompt, claim, generation, qa_pairs in zip(
                prompts, all_claims, generations, all_qa_pairs
            )
        ]

        os.makedirs(os.path.dirname(cfg.save_fp), exist_ok=True)
        save_file_jsonl(generated_data, cfg.save_fp)
        fancy_print(console, f"Saved to {cfg.save_fp}")

    if mode == "evaluate" or mode == "all":
        try:
            data = load_jsonlines(cfg.save_fp)
        except:
            raise ValueError(f"Check that {cfg.save_fp} contains valid generations")

        # Print first example generation
        fancy_print(console, f"Ex. generation: {data[0]['output']}")

        (output_em_mean, output_em_ci), (output_hit_mean, output_hit_ci) = (
            compute_str_em(data)
        )
        (oracle_em_mean, oracle_em_ci), (oracle_hit_mean, oracle_hit_ci) = (
            compute_str_em_oracle(data)
        )

        display_str_em_results(
            (output_em_mean, output_em_ci),
            (output_hit_mean, output_hit_ci),
            (oracle_em_mean, oracle_em_ci),
            (oracle_hit_mean, oracle_hit_ci),
            cfg.model_name_or_path,
        )

        if cfg.save_results is True:
            results_fp = cfg.save_fp.replace(".jsonl", "_results.jsonl")
            results = [
                {
                    "output_em_mean": output_em_mean,
                    "output_em_ci": output_em_ci,
                    "output_hit_mean": output_hit_mean,
                    "output_hit_ci": output_hit_ci,
                    "oracle_em_mean": oracle_em_mean,
                    "oracle_em_ci": oracle_em_ci,
                    "oracle_hit_mean": oracle_hit_mean,
                    "oracle_hit_ci": oracle_hit_ci,
                    "model_name": cfg.model_name_or_path,
                }
            ]
            save_file_jsonl(results, results_fp)
            fancy_print(console, f"Saved ASQA results to {results_fp}")


if __name__ == "__main__":
    rag()
