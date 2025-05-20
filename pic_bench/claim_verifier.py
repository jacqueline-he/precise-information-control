import os
from .get_response import GetResponse
import re
from rich.progress import track


# Code adapted from: https://github.com/Yixiao-Song/VeriScore
class ClaimVerifier:
    def __init__(self, model_name, use_cache=False, cache_dir="../cache/"):
        self.model_name = model_name
        self.use_cache = use_cache
        if self.use_cache:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "claim_verification_cache.json")
        else:
            self.cache_file = None
        self.get_model_response = GetResponse(
            cache_file=self.cache_file,
            model_name=model_name,
            max_tokens=1000,
            temperature=0,
            use_cache=use_cache,
        )
        self.system_message = "You are a helpful assistant who can verify the truthfulness of a claim against provided contextual knowledge."

    def get_instruction_template(self):
        prompt_temp = (
            "Your task is to judge whether a claim is supported by a series of atomic claims. Do not use any external knowledge. Evaluate the claim strictly based on the provided input claims. Return either 'supported' or 'unsupported', followed by the number(s) of the supporting atomic claim(s) if necessary, comma-separated and enclosed in angle brackets (<>).\nExample Output: `supported <2,3>`\n`unsupported`\n\n"
            + "Below are the definitions of the two categories:\n"
            + "- Supported: A claim is supported by the atomic claims if one or more claims directly support the claim. There may be cases in which some of the atomic claims are not fully related to the claim, but none of the claims should directly contradict the claim. All parts of the claim should be supported by the atomic claims. If there is a part of the claim that is not directly supported, the claim should be marked as unsupported."
            + "- Unsupported: If a claim is not supported by any of the atomic claims, mark it as unsupported.\n\n"
            + "Claim: {claim}\n Atomic Claims: {atomic_claims}"
        )
        return prompt_temp

    # Find the content within angle brackets
    def extract_indices(self, s):
        match = re.search(r"<(.*?)>", s)
        return [int(num) for num in match.group(1).split(",")] if match else []

    def verifying_claim(self, resp_claims, prompt_claims):
        prompt_tok_cnt, response_tok_cnt = 0, 0
        out_lst = []
        prompt = self.get_instruction_template()
        for resp_claim in resp_claims:
            formatted_prompt_claims = "\n".join(
                f"{i + 1}. {item}" for i, item in enumerate(prompt_claims)
            )
            formatted_prompt = prompt.format(
                claim=resp_claim, atomic_claims=formatted_prompt_claims
            )
            response, prompt_tok_num, response_tok_num = (
                self.get_model_response.get_response(
                    self.system_message, formatted_prompt
                )
            )
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if "unsupported" in response.lower():
                clean_output = "unsupported"
                supporting_inds = []
            else:
                clean_output = "supported"
                supporting_inds = self.extract_indices(response)
            claim_verify_res_dict = {
                "resp_claim": resp_claim,
                "prompt_claims": prompt_claims,
                "verification_result": clean_output,
                "supporting_inds": supporting_inds,
            }
            out_lst.append(claim_verify_res_dict)
        return out_lst, prompt_tok_cnt, response_tok_cnt

    def verifying_claim_snippets(
        self, claim_snippets_dict, search_res_num=5, prompt_initial_temp=None
    ):
        """
        search_snippet_lst = [{"title": title, "snippet": snippet, "link": link}, ...]
        """
        your_task = "Your task:\n\nClaim: {claim}\n\n{search_results}\n\nYour decision:"

        system_message = "You are a helpful assistant who can judge whether a claim is supported by the search results or not."
        prompt_tok_cnt, response_tok_cnt = 0, 0
        out_lst = []
        claim_verify_res_dict = {}
        for claim in track(claim_snippets_dict, description="Verifying claims..."):
            search_snippet_lst = claim_snippets_dict[claim]
            search_res_str = ""
            search_cnt = 1
            for search_dict in search_snippet_lst[:search_res_num]:
                search_res_str += f'Search result {search_cnt}\nTitle: {search_dict["title"].strip()}\nLink: {search_dict["link"].strip()}\nContent: {search_dict["snippet"].strip()}\n\n'
                search_cnt += 1

            prompt_tail = your_task.format(
                claim=claim,
                search_results=search_res_str.strip(),
            )
            prompt = f"{prompt_initial_temp}\n\n{prompt_tail}"
            response, prompt_tok_num, response_tok_num = (
                self.get_model_response.get_response(system_message, prompt)
            )
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            clean_output = (
                response.replace("#", "").split(".")[0].lower()
                if response is not None
                else None
            )
            claim_verify_res_dict = {
                "claim": claim,
                "search_results": search_res_str,
                "verification_result": clean_output,
            }
            out_lst.append(claim_verify_res_dict)
        return out_lst, prompt_tok_cnt, response_tok_cnt
