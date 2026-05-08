"""
Async web-fetching tools for the 0G docs agent.

fetch_page           — single URL
fetch_pages_parallel — comma-separated URLs fetched concurrently with asyncio.gather

Using str instead of list[str] for the parallel tool because several
OpenAI-compatible endpoints do not reliably serialise array-type tool
parameters — the model silently omits the field, causing a ValidationError.
A plain string is universally safe.

Page content is cached in a SQLite database at ./data/page_cache.db with a
configurable TTL (default 24 hours). Use clear_page_cache() to invalidate.
"""

import asyncio
import json
import os
import sqlite3
import time
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# SSRF guard — only permit requests to approved 0G domains
# ---------------------------------------------------------------------------

_ALLOWED_HOSTS = frozenset({
    "0g.ai",
    "docs.0g.ai",
    "build.0g.ai",
    "pc.0g.ai",
    "app.0g.ai",
})


def _is_allowed_url(url: str) -> bool:
    """Return True only for http(s) URLs on approved 0G domains."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        return host in _ALLOWED_HOSTS or host.endswith(".0g.ai")
    except Exception:
        return False


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; 0GDocsAgent/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
_TIMEOUT = 30.0
_MAX_CHARS = 5_000  # per page, to keep token usage reasonable

# ---------------------------------------------------------------------------
# Cache warm-up — pre-fetch known URLs into the SQLite cache at startup
# ---------------------------------------------------------------------------

# INDEXABLE_URLS is the canonical list of stable doc pages (defined in core/rag.py).
# _WARM_URLS extends it with dynamic pages (blog, sitemap) that are fetched but
# never indexed into ChromaDB.
from core.rag import INDEXABLE_URLS  # noqa: E402 (after stdlib/third-party)

_WARM_URLS: list = sorted(INDEXABLE_URLS) + [
    "https://0g.ai/blog",
    "https://0g.ai/sitemap.xml",
]


async def warm_cache() -> None:
    """Pre-fetch all known URLs into the SQLite cache and update the RAG index.

    Runs as a background task at startup. Errors are silently ignored —
    a failed warm-up is not critical; the agent will fetch live on demand.
    """
    async with httpx.AsyncClient(headers=_HEADERS) as client:
        results = await asyncio.gather(
            *[_fetch_one(client, url) for url in _WARM_URLS],
            return_exceptions=True,
        )

    # Build a map of successfully fetched pages for RAG indexing.
    # return_exceptions=True means failed coroutines return an Exception object.
    content_map = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        url, content = result
        if content and not content.startswith(("### Fetch failed", "### Blocked")):
            content_map[url] = content

    if content_map:
        try:
            from core.rag import index_urls  # noqa: PLC0415
            await asyncio.to_thread(index_urls, list(content_map.keys()), content_map)
        except Exception as e:
            print(f"[warm_cache] RAG indexing failed: {e}")

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours
_CACHE_DB_PATH = "./data/page_cache.db"
_cache_initialised = False


def _init_cache() -> None:
    """Create ./data/ directory and the page_cache table if they don't exist.

    Idempotent — safe to call multiple times.
    """
    global _cache_initialised
    if _cache_initialised:
        return

    db_dir = os.path.dirname(_CACHE_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(_CACHE_DB_PATH, check_same_thread=False, timeout=5)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS page_cache (
                url        TEXT PRIMARY KEY,
                content    TEXT NOT NULL,
                cached_at  REAL NOT NULL
            )
            """
        )
        conn.commit()
        conn.execute(
            "DELETE FROM page_cache WHERE length(content) > ?",
            (_MAX_CHARS + 500,),
        )
        conn.commit()
    finally:
        conn.close()

    _cache_initialised = True


def _cache_get(url: str) -> Optional[str]:
    """Return cached content for *url* if it exists and has not expired, else None."""
    _init_cache()
    conn = sqlite3.connect(_CACHE_DB_PATH, check_same_thread=False, timeout=5)
    try:
        cursor = conn.execute(
            "SELECT content, cached_at FROM page_cache WHERE url = ?",
            (url,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        content, cached_at = row
        if (time.time() - cached_at) > _CACHE_TTL_SECONDS:
            return None
        return content
    finally:
        conn.close()


def _cache_set(url: str, content: str) -> None:
    """Upsert *url* → *content* into the cache with the current timestamp."""
    _init_cache()
    conn = sqlite3.connect(_CACHE_DB_PATH, check_same_thread=False, timeout=5)
    try:
        conn.execute(
            """
            INSERT INTO page_cache (url, content, cached_at)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                content   = excluded.content,
                cached_at = excluded.cached_at
            """,
            (url, content, time.time()),
        )
        conn.commit()
    finally:
        conn.close()


def clear_page_cache(url: Optional[str] = None) -> str:
    """Clear cached page content.

    - url=None  -> clear entire cache, return count of deleted rows as message
    - url=str   -> clear that specific URL only, return confirmation message
    """
    _init_cache()
    conn = sqlite3.connect(_CACHE_DB_PATH, check_same_thread=False, timeout=5)
    try:
        if url is None:
            cursor = conn.execute("DELETE FROM page_cache")
            conn.commit()
            return f"Page cache cleared: {cursor.rowcount} entries removed."
        else:
            cursor = conn.execute("DELETE FROM page_cache WHERE url = ?", (url,))
            conn.commit()
            if cursor.rowcount:
                return f"Cache entry cleared for: {url}"
            return f"No cache entry found for: {url}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def _parse_xml(url: str, content: str) -> str:
    """Parse XML documents (sitemaps, RSS feeds) and return readable text.

    Extracts all ``<loc>`` values from sitemaps and ``<title>``/``<link>``
    pairs from RSS feeds, stripping XML namespace prefixes automatically.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        return f"### Parse error: {url}\nError: failed to parse XML — {e}"

    def tag(elem) -> str:
        """Strip ``{namespace}`` prefix from an element tag."""
        t = elem.tag
        return t.split("}")[-1] if "}" in t else t

    # Collect all <loc> entries (sitemaps and sitemap indexes)
    locs = [elem.text.strip() for elem in root.iter() if tag(elem) == "loc" and elem.text]

    if locs:
        return (
            f"### Sitemap: {url}\nURL: {url}\n\n"
            f"Found {len(locs)} URLs:\n\n" + "\n".join(locs)
        )

    # RSS / Atom fallback — collect <title> + <link> pairs
    items = []
    for item in root.iter():
        if tag(item) in ("item", "entry"):
            title = next((c.text for c in item if tag(c) == "title"), "")
            link = next((c.text for c in item if tag(c) == "link"), "")
            if title or link:
                items.append(f"{title.strip()} — {link.strip()}")
    if items:
        return (
            f"### Feed: {url}\nURL: {url}\n\n"
            + "\n".join(items)
        )

    # Generic fallback — dump all text nodes
    texts = [elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()]
    return f"### XML: {url}\nURL: {url}\n\n" + "\n".join(texts)


def _extract_json_strings(obj, depth: int = 0) -> list:
    """Recursively collect meaningful string values from a JSON structure.

    Skips short strings, URLs, and values that look like code/keys so that
    the resulting list contains human-readable prose only.
    """
    if depth > 10:
        return []
    if isinstance(obj, str):
        s = obj.strip()
        # Keep readable prose (long strings without URL/code noise)
        is_prose = len(s) > 40 and not s.startswith(("http", "data:", "/", "#", "{"))
        # Also keep blog/page paths so the agent can construct fetch URLs
        is_page_path = s.startswith("/blog/") and len(s) > 7
        if is_prose or is_page_path:
            return [s]
        return []
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_extract_json_strings(item, depth + 1))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_extract_json_strings(v, depth + 1))
        return out
    return []


def _parse(url: str, html: str) -> str:
    """Strip noise from HTML and return cleaned plain text.

    Routes XML documents (sitemaps, RSS) to ``_parse_xml``.  For HTML pages,
    extracts Next.js ``__NEXT_DATA__`` JSON before stripping scripts so that
    JavaScript-rendered content is still readable.
    """
    # Detect XML before touching BeautifulSoup to avoid XMLParsedAsHTMLWarning
    stripped = html.lstrip()
    if url.endswith(".xml") or stripped.startswith("<?xml") or stripped.startswith("<rss") or stripped.startswith("<feed"):
        return _parse_xml(url, html)

    soup = BeautifulSoup(html, "html.parser")

    # 1. Extract Next.js page data BEFORE stripping script tags
    next_data_text = ""
    next_data_tag = soup.find("script", {"id": "__NEXT_DATA__"})
    if next_data_tag and next_data_tag.string:
        try:
            data = json.loads(next_data_tag.string)
            strings = _extract_json_strings(data)
            next_data_text = "\n".join(strings)
        except Exception:
            pass

    # 2. Meta description fallback (og:description or plain description)
    meta_desc = ""
    for attr in ("property", "name"):
        for val in ("og:description", "description"):
            tag = soup.find("meta", {attr: val})
            if tag and tag.get("content"):
                meta_desc = tag["content"].strip()
                break
        if meta_desc:
            break

    # 3. Strip noise and grab visible body text
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url

    body = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.body
    )
    raw = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    text = "\n".join(lines)

    # 4. If the visible body is sparse, fall back to richer sources
    if len(text) < 300:
        if next_data_text:
            text = next_data_text
        elif meta_desc:
            text = f"[Page content is client-side rendered — meta description only]\n\n{meta_desc}"

    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n\n... [content truncated — fetch a more specific sub-page if needed]"

    return f"### {title}\nURL: {url}\n\n{text}"


# ---------------------------------------------------------------------------
# Internal fetch helper (with cache)
# ---------------------------------------------------------------------------

async def _fetch_one(client: httpx.AsyncClient, url: str) -> Tuple[str, str]:
    """Return (url, parsed_text_or_error_message).

    Checks the SQLite cache before making a network request. Successful
    responses are stored in the cache; error responses are not.
    """
    if not _is_allowed_url(url):
        return url, (
            f"### Blocked: {url}\n"
            f"Error: URL is not on an approved 0G domain. "
            f"Only 0g.ai and its subdomains are permitted."
        )

    cached = await asyncio.to_thread(_cache_get, url)
    if cached is not None:
        return url, cached

    try:
        r = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
        r.raise_for_status()
        content = _parse(url, r.text)
        await asyncio.to_thread(_cache_set, url, content)
        return url, content
    except httpx.TimeoutException:
        return url, f"### Fetch failed: {url}\nError: request timed out after {_TIMEOUT}s"
    except httpx.HTTPStatusError as e:
        return url, f"### Fetch failed: {url}\nError: HTTP {e.response.status_code}"
    except Exception as e:
        return url, f"### Fetch failed: {url}\nError: {e}"


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class FetchPageInput(BaseModel):
    url: str = Field(description="The full URL of the documentation page to fetch")


class FetchPagesParallelInput(BaseModel):
    # Plain str avoids list-type serialisation issues on OpenAI-compatible endpoints.
    urls: str = Field(
        description=(
            "Comma-separated full URLs to fetch simultaneously. "
            "Example: \"https://docs.0g.ai/,https://build.0g.ai/\""
        )
    )


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool("fetch_page", args_schema=FetchPageInput)
async def fetch_page(url: str) -> str:
    """Fetch a single documentation page and return its text content."""
    url = url.strip()
    if not url:
        return "Error: no URL provided."
    async with httpx.AsyncClient(headers=_HEADERS) as client:
        _, content = await _fetch_one(client, url)
    return content


async def _fetch_and_join(url_list: list) -> str:
    """Fetch *url_list* in parallel and return pages joined by a separator."""
    async with httpx.AsyncClient(headers=_HEADERS) as client:
        results = await asyncio.gather(*[_fetch_one(client, url) for url in url_list])
    separator = "\n" + "=" * 72 + "\n"
    return separator.join(content for _, content in results)


@tool("fetch_pages_parallel", args_schema=FetchPagesParallelInput)
async def fetch_pages_parallel(urls: str) -> str:
    """
    Fetch multiple documentation pages in parallel and return their combined text.
    Pass URLs as a comma-separated string, e.g.:
      "https://docs.0g.ai/,https://build.0g.ai/"
    Prefer this over repeated fetch_page calls — it is much faster.
    """
    url_list = [u.strip() for u in urls.split(",") if u.strip()]
    if not url_list:
        return "Error: no valid URLs provided. Pass a comma-separated list of URLs."
    return await _fetch_and_join(url_list)


async def prefetch_urls(url_list: list) -> str:
    """Fetch a list of URLs and return their combined content.

    Used by the pre-routing layer in agent.py to populate context before the
    first LLM call, eliminating the planning round-trip entirely.  Results
    are served from the SQLite cache when available (warm cache = <300ms).
    """
    return await _fetch_and_join(url_list)
