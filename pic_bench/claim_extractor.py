import os
import regex
import spacy
from .get_response import GetResponse
from prompts.claim_extraction_template import CLAIM_EXTRACTION_TEMPLATE
from concurrent.futures import ThreadPoolExecutor


# Code heavily adapted from: https://github.com/Yixiao-Song/VeriScore
class ClaimExtractor:
    def __init__(self, model_name, use_cache=True, cache_dir=None):
        self.use_cache = use_cache
        if self.use_cache:
            cache_dir = os.path.join(cache_dir, model_name)
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f"claim_extraction_cache.json")
        else:
            self.cache_file = None
        self.get_model_response = GetResponse(
            cache_file=self.cache_file,
            model_name=model_name,
            max_tokens=1000,
            temperature=0,
            use_cache=self.use_cache,
        )
        self.system_message = "You are a helpful assistant who can extract verifiable atomic claims from a piece of text. Each extracted claim should be verifiable against some knowledge source. "
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def generate_snippets(self, sentences):
        """Generate snippets in batch."""
        lead_sent = sentences[0]

        def get_context(i):
            context1 = " ".join(sentences[max(0, i - 3) : i])
            context2 = " ".join(sentences[i + 1 : i + 2])
            return context1, context2

        return [
            f"{lead_sent} {context1} <SOS>{sent} <EOS> {context2}".strip()
            for i, sent in enumerate(sentences)
            for context1, context2 in [get_context(i)]
        ]

    def extract_claims_parallel(self, snippets, sentences, claim_extractor):
        """Run claim extraction in parallel using threads."""
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() * 5, 32)) as executor:
            futures = [
                executor.submit(claim_extractor, snippet, sentence.strip())
                for snippet, sentence in zip(snippets, sentences)
            ]
            return [f.result() for f in futures]

    def scanner_extractor_fast(self, response):
        sentences = self.get_sentence(response)
        snippet_lst = self.generate_snippets(sentences)
        claim_results = self.extract_claims_parallel(
            snippet_lst, sentences, self.claim_extractor
        )
        # Process extracted claims
        all_claims_lst = []
        claim_lst_lst = []
        prompt_tok_cnt, response_tok_cnt = 0, 0
        seen_claims = set()

        for claims, prompt_tok_num, response_tok_num in claim_results:
            prompt_tok_cnt += prompt_tok_num
            response_tok_cnt += response_tok_num

            if claims is None:
                claim_lst_lst.append([None])
                continue

            claim_lst = list(
                set(
                    claim.strip()
                    for claim in claims
                    if claim.strip() and not claim.startswith("Note:")
                )
            )

            unique_claims = [claim for claim in claim_lst if claim not in seen_claims]
            seen_claims.update(unique_claims)  # Add to global seen claims

            all_claims_lst.extend(unique_claims)
            claim_lst_lst.append(unique_claims)

        return (
            snippet_lst,
            claim_lst_lst,
            all_claims_lst,
            prompt_tok_cnt,
            response_tok_cnt,
        )

    def get_sentence(self, text):
        # use spaCy to split the text into sentences
        return [x.text.strip() for x in self.spacy_nlp(text).sents]

    def claim_extractor(self, snippet, sentence):
        """
        snippet = (context1) <SOS>sentence<EOS> (context2)
        sentence = the sentence to be focused on
        """

        ### prompting base approach via API call
        # with open("../prompts/claim_extraction_template.txt", "r") as f:
        #     prompt_template = f.read()
        prompt_text = CLAIM_EXTRACTION_TEMPLATE.format(
            snippet=snippet, sentence=sentence
        )
        response, prompt_tok_cnt, response_tok_cnt = (
            self.get_model_response.get_response(self.system_message, prompt_text)
        )
        if not response or "no verifiable claim" in response.lower().strip():
            return None, prompt_tok_cnt, response_tok_cnt
        else:
            # remove itemized list
            claims = [x.strip().replace("- ", "") for x in response.split("\n")]
            # remove numbers in the beginning
            claims = [regex.sub(r"^\d+\.?\s", "", x) for x in claims]
            return claims, prompt_tok_cnt, response_tok_cnt
