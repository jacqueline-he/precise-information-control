import os
from ast import literal_eval
import json
import requests
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)


class SearchAPI:
    def __init__(self, cache_file=None):
        # invariant variables
        self.serper_key = os.getenv("SERPER_KEY_PRIVATE")
        self.url = "https://google.serper.dev/search"
        self.headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json",
        }
        # cache related
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.add_n = 0
        self.save_interval = 10

    def get_snippets(self, claim_lst, console):
        text_claim_snippets_dict = {}
        failed_queries = []
        empty_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Getting search results...", total=len(claim_lst))

            for i, query in enumerate(claim_lst):
                try:
                    search_result = self.get_search_res(query)

                    if "statusCode" in search_result:
                        console.print(
                            f"[bold red]Error for query {i}: {search_result['message']}[/bold red]"
                        )
                        failed_queries.append((i, query, search_result["message"]))
                        # Instead of exit, continue with empty results
                        search_res_lst = []
                    else:
                        organic_res = search_result.get("organic", [])
                        if not organic_res:
                            empty_results.append((i, query))

                        search_res_lst = []
                        for item in organic_res:
                            title = item.get("title", "")
                            snippet = item.get("snippet", "")
                            link = item.get("link", "")
                            search_res_lst.append(
                                {"title": title, "snippet": snippet, "link": link}
                            )

                    text_claim_snippets_dict[query] = search_res_lst
                except Exception as e:
                    console.print(
                        f"[bold red]Unexpected error for query {i}: {str(e)}[/bold red]"
                    )
                    failed_queries.append((i, query, str(e)))
                    text_claim_snippets_dict[query] = []

                progress.update(task, advance=1)

                # Save cache more frequently during errors
                if failed_queries and self.add_n % 5 == 0:
                    self.save_cache()

        # Print summary of issues
        if failed_queries:
            console.print("\n[bold red]Failed Queries:[/bold red]")
            for idx, query, error in failed_queries:
                console.print(f"Query {idx}: {query}\nError: {error}\n")

        if empty_results:
            console.print("\n[bold yellow]Queries with No Results:[/bold yellow]")
            for idx, query in empty_results:
                console.print(f"Query {idx}: {query}\n")

        return text_claim_snippets_dict, failed_queries, empty_results

    def get_search_res(self, query):
        # check if prompt is in cache; if so, return from cache
        cache_key = query.strip()
        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        payload = json.dumps({"q": query})
        response = requests.request(
            "POST", self.url, headers=self.headers, data=payload
        )
        response_json = literal_eval(response.text)

        # update cache
        self.cache_dict[query.strip()] = response_json
        self.add_n += 1

        # save cache every save_interval times
        if self.add_n % self.save_interval == 0:
            self.save_cache()

        return response_json

    def save_cache(self):
        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        cache = self.load_cache().items()
        for k, v in cache:
            self.cache_dict[k] = v
        print(f"Saving search cache ...")
        with open(self.cache_file, "w") as f:
            json.dump(self.cache_dict, f, indent=4)

    def load_cache(self):
        if self.cache_file is not None and os.path.exists(self.cache_file):
            print(f"Loading cache from {self.cache_file}...")
            with open(self.cache_file, "r") as f:
                cache = json.load(f)
        else:
            cache = {}
        return cache
