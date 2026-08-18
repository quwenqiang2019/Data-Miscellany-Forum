from arxiv import Search

def arxiv_search(query: str):
    search = Search(query=query, max_results=3, sort_by="relevance")
    results = []
    for r in search.results():
        results.append({
            "title": r.title,
            "summary": r.summary,
            "url": r.entry_id
        })
    return results