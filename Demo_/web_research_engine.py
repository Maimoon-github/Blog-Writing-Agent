# #!/usr/bin/env python3
# """
# Web Research Engine - Production-Grade Implementation

# A comprehensive web-based research tool that performs deep research by:
# 1. Querying search engines (DuckDuckGo)
# 2. Retrieving and processing multiple relevant results
# 3. Extracting substantial content from each source
# 4. Aggregating information into structured outputs
# 5. Producing clear, verifiable research reports

# Features:
# - Multi-query research workflows
# - Full-page content extraction
# - Intelligent rate limiting and retry logic
# - Multiple output formats (JSON, Markdown, HTML)
# - Robust error handling and logging
# - Progress tracking and metrics
# """

# import asyncio
# import json
# import logging
# import re
# import time
# from dataclasses import dataclass, field, asdict
# from datetime import datetime
# from pathlib import Path
# from typing import List, Dict, Optional, Set
# from urllib.parse import urlparse, urljoin
# from html.parser import HTMLParser
# import hashlib

# # Third-party imports
# try:
#     from duckduckgo_search import DDGS
#     DDGS_AVAILABLE = True
# except ImportError:
#     DDGS_AVAILABLE = False
#     print("Warning: duckduckgo-search not installed. Install with: pip install duckduckgo-search")

# try:
#     import requests
#     from requests.adapters import HTTPAdapter
#     from urllib3.util.retry import Retry
#     REQUESTS_AVAILABLE = True
# except ImportError:
#     REQUESTS_AVAILABLE = False
#     print("Warning: requests not installed. Install with: pip install requests")

# try:
#     from bs4 import BeautifulSoup
#     BS4_AVAILABLE = True
# except ImportError:
#     BS4_AVAILABLE = False
#     print("Warning: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")

# try:
#     from readability import Document
#     READABILITY_AVAILABLE = True
# except ImportError:
#     READABILITY_AVAILABLE = False
#     # Readability is optional - we have fallback extraction

# # ----------------------------------------------------------------------
# # Logging Configuration
# # ----------------------------------------------------------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("web_research.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


# # ----------------------------------------------------------------------
# # Data Models
# # ----------------------------------------------------------------------
# @dataclass
# class SearchResult:
#     """Represents a single search result."""
#     title: str
#     url: str
#     snippet: str
#     position: int
#     query: str
#     timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# @dataclass
# class ExtractedContent:
#     """Represents extracted content from a web page."""
#     url: str
#     title: str
#     content: str
#     word_count: int
#     extraction_method: str
#     metadata: Dict[str, any] = field(default_factory=dict)
#     timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
#     error: Optional[str] = None
    

# @dataclass
# class ResearchQuery:
#     """Configuration for a research query."""
#     query: str
#     max_results: int = 10
#     time_range: Optional[str] = None  # 'd' (day), 'w' (week), 'm' (month), 'y' (year)
#     region: str = 'wt-wt'  # worldwide


# @dataclass
# class ResearchReport:
#     """Complete research report with all findings."""
#     topic: str
#     queries: List[str]
#     search_results: List[SearchResult]
#     extracted_content: List[ExtractedContent]
#     metadata: Dict[str, any]
#     timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
#     def to_dict(self) -> dict:
#         """Convert to dictionary for JSON serialization."""
#         return {
#             'topic': self.topic,
#             'queries': self.queries,
#             'search_results': [asdict(r) for r in self.search_results],
#             'extracted_content': [asdict(c) for c in self.extracted_content],
#             'metadata': self.metadata,
#             'timestamp': self.timestamp
#         }


# # ----------------------------------------------------------------------
# # HTML Content Extractor
# # ----------------------------------------------------------------------
# class ContentExtractor:
#     """Extract meaningful text content from HTML."""
    
#     # Tags that typically contain main content
#     CONTENT_TAGS = ['article', 'main', 'section', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
#     # Tags to remove
#     SKIP_TAGS = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']
    
#     @staticmethod
#     def clean_text(text: str) -> str:
#         """Clean and normalize extracted text."""
#         # Remove extra whitespace
#         text = re.sub(r'\s+', ' ', text)
#         # Remove multiple newlines
#         text = re.sub(r'\n\s*\n', '\n\n', text)
#         return text.strip()
    
#     @staticmethod
#     def extract_with_readability(html: str, url: str) -> tuple[str, str, str]:
#         """Extract content using readability library."""
#         if not READABILITY_AVAILABLE:
#             raise ImportError("Readability library not available")
        
#         doc = Document(html)
#         title = doc.title()
#         content_html = doc.summary()
        
#         # Convert HTML to text
#         soup = BeautifulSoup(content_html, 'html.parser')
#         content = soup.get_text(separator='\n', strip=True)
        
#         return title, content, "readability"
    
#     @staticmethod
#     def extract_with_beautifulsoup(html: str, url: str) -> tuple[str, str, str]:
#         """Extract content using BeautifulSoup with heuristics."""
#         if not BS4_AVAILABLE:
#             raise ImportError("BeautifulSoup library not available")
        
#         soup = BeautifulSoup(html, 'html.parser')
        
#         # Remove unwanted tags
#         for tag in soup.find_all(ContentExtractor.SKIP_TAGS):
#             tag.decompose()
        
#         # Try to find title
#         title = ""
#         title_tag = soup.find('title')
#         if title_tag:
#             title = title_tag.get_text(strip=True)
#         elif soup.find('h1'):
#             title = soup.find('h1').get_text(strip=True)
        
#         # Try to find main content container
#         content_container = None
        
#         # Priority order for content containers
#         for selector in ['article', 'main', '[role="main"]', '.content', '#content', '.post', '.article']:
#             content_container = soup.select_one(selector)
#             if content_container:
#                 break
        
#         # Fall back to body
#         if not content_container:
#             content_container = soup.find('body')
        
#         if not content_container:
#             content_container = soup
        
#         # Extract text
#         content = content_container.get_text(separator='\n', strip=True)
        
#         return title, content, "beautifulsoup"
    
#     @staticmethod
#     def extract_basic(html: str, url: str) -> tuple[str, str, str]:
#         """Basic HTML tag stripping as fallback."""
#         # Simple HTML tag removal
#         title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
#         title = title_match.group(1).strip() if title_match else urlparse(url).netloc
        
#         # Remove scripts and styles
#         html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
#         html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
#         # Remove all HTML tags
#         content = re.sub(r'<[^>]+>', ' ', html)
        
#         # Decode HTML entities
#         content = content.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
#         return title, content, "basic"
    
#     @classmethod
#     def extract(cls, html: str, url: str) -> tuple[str, str, str]:
#         """
#         Extract content using best available method.
#         Returns: (title, content, method_used)
#         """
#         methods = []
        
#         if READABILITY_AVAILABLE:
#             methods.append(cls.extract_with_readability)
#         if BS4_AVAILABLE:
#             methods.append(cls.extract_with_beautifulsoup)
#         methods.append(cls.extract_basic)
        
#         for method in methods:
#             try:
#                 title, content, method_name = method(html, url)
#                 content = cls.clean_text(content)
#                 if content and len(content) > 100:  # Minimum content threshold
#                     return title, content, method_name
#             except Exception as e:
#                 logger.debug(f"Extraction method {method.__name__} failed: {e}")
#                 continue
        
#         # Final fallback
#         return urlparse(url).netloc, "", "failed"


# # ----------------------------------------------------------------------
# # Web Fetcher with Retry Logic
# # ----------------------------------------------------------------------
# class WebFetcher:
#     """Robust web page fetcher with retry logic and rate limiting."""
    
#     def __init__(self, timeout: int = 30, max_retries: int = 3, rate_limit_delay: float = 1.0):
#         """
#         Initialize web fetcher.
        
#         Args:
#             timeout: Request timeout in seconds
#             max_retries: Maximum number of retry attempts
#             rate_limit_delay: Delay between requests in seconds
#         """
#         self.timeout = timeout
#         self.rate_limit_delay = rate_limit_delay
#         self.last_request_time = 0
        
#         if not REQUESTS_AVAILABLE:
#             raise ImportError("requests library is required")
        
#         # Create session with retry logic
#         self.session = requests.Session()
        
#         retry_strategy = Retry(
#             total=max_retries,
#             backoff_factor=1,
#             status_forcelist=[429, 500, 502, 503, 504],
#             allowed_methods=["GET", "HEAD"]
#         )
        
#         adapter = HTTPAdapter(max_retries=retry_strategy)
#         self.session.mount("http://", adapter)
#         self.session.mount("https://", adapter)
        
#         # Set user agent to avoid blocking
#         self.session.headers.update({
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#             'Accept-Language': 'en-US,en;q=0.5',
#             'Accept-Encoding': 'gzip, deflate',
#             'DNT': '1',
#             'Connection': 'keep-alive',
#             'Upgrade-Insecure-Requests': '1'
#         })
    
#     def _rate_limit(self):
#         """Implement rate limiting between requests."""
#         current_time = time.time()
#         time_since_last = current_time - self.last_request_time
        
#         if time_since_last < self.rate_limit_delay:
#             sleep_time = self.rate_limit_delay - time_since_last
#             logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
#             time.sleep(sleep_time)
        
#         self.last_request_time = time.time()
    
#     def fetch(self, url: str) -> tuple[Optional[str], Optional[str]]:
#         """
#         Fetch a URL and return HTML content.
        
#         Returns:
#             Tuple of (html_content, error_message)
#         """
#         self._rate_limit()
        
#         try:
#             logger.info(f"Fetching: {url}")
#             response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
#             response.raise_for_status()
            
#             # Check content type
#             content_type = response.headers.get('Content-Type', '')
#             if 'text/html' not in content_type.lower():
#                 return None, f"Non-HTML content type: {content_type}"
            
#             return response.text, None
            
#         except requests.exceptions.Timeout:
#             error = f"Timeout after {self.timeout}s"
#             logger.warning(f"Failed to fetch {url}: {error}")
#             return None, error
        
#         except requests.exceptions.HTTPError as e:
#             error = f"HTTP {e.response.status_code}"
#             logger.warning(f"Failed to fetch {url}: {error}")
#             return None, error
        
#         except requests.exceptions.RequestException as e:
#             error = str(e)
#             logger.warning(f"Failed to fetch {url}: {error}")
#             return None, error
        
#         except Exception as e:
#             error = f"Unexpected error: {str(e)}"
#             logger.error(f"Failed to fetch {url}: {error}")
#             return None, error


# # ----------------------------------------------------------------------
# # Search Engine Interface
# # ----------------------------------------------------------------------
# class SearchEngine:
#     """Interface for web search using DuckDuckGo."""
    
#     def __init__(self):
#         """Initialize search engine."""
#         if not DDGS_AVAILABLE:
#             raise ImportError("duckduckgo-search library is required")
        
#         self.ddgs = DDGS()
    
#     def search(self, query_config: ResearchQuery) -> List[SearchResult]:
#         """
#         Perform a search and return results.
        
#         Args:
#             query_config: Research query configuration
            
#         Returns:
#             List of SearchResult objects
#         """
#         results = []
        
#         try:
#             logger.info(f"Searching: '{query_config.query}' (max_results={query_config.max_results})")
            
#             # Perform search
#             search_results = self.ddgs.text(
#                 keywords=query_config.query,
#                 region=query_config.region,
#                 max_results=query_config.max_results,
#                 timelimit=query_config.time_range
#             )
            
#             for idx, result in enumerate(search_results):
#                 search_result = SearchResult(
#                     title=result.get('title', 'No Title'),
#                     url=result.get('href', result.get('link', '')),
#                     snippet=result.get('body', result.get('snippet', '')),
#                     position=idx + 1,
#                     query=query_config.query
#                 )
#                 results.append(search_result)
#                 logger.debug(f"  [{idx+1}] {search_result.title}")
            
#             logger.info(f"Found {len(results)} results for '{query_config.query}'")
            
#         except Exception as e:
#             logger.error(f"Search failed for '{query_config.query}': {e}")
        
#         return results


# # ----------------------------------------------------------------------
# # Research Engine - Main Orchestrator
# # ----------------------------------------------------------------------
# class ResearchEngine:
#     """Main research engine that orchestrates search, extraction, and aggregation."""
    
#     def __init__(
#         self,
#         max_pages_per_query: int = 10,
#         timeout: int = 30,
#         rate_limit_delay: float = 1.0,
#         min_content_words: int = 100
#     ):
#         """
#         Initialize research engine.
        
#         Args:
#             max_pages_per_query: Maximum pages to fetch per search query
#             timeout: Request timeout in seconds
#             rate_limit_delay: Delay between requests in seconds
#             min_content_words: Minimum words for valid content
#         """
#         self.max_pages_per_query = max_pages_per_query
#         self.min_content_words = min_content_words
        
#         self.search_engine = SearchEngine()
#         self.fetcher = WebFetcher(timeout=timeout, rate_limit_delay=rate_limit_delay)
#         self.extractor = ContentExtractor()
        
#         self.visited_urls: Set[str] = set()
    
#     def _normalize_url(self, url: str) -> str:
#         """Normalize URL for deduplication."""
#         parsed = urlparse(url)
#         # Remove fragment and common tracking parameters
#         normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
#         return normalized.lower()
    
#     def _should_fetch(self, url: str) -> bool:
#         """Check if URL should be fetched (not already visited)."""
#         normalized = self._normalize_url(url)
        
#         if normalized in self.visited_urls:
#             logger.debug(f"Skipping duplicate: {url}")
#             return False
        
#         return True
    
#     def _mark_visited(self, url: str):
#         """Mark URL as visited."""
#         normalized = self._normalize_url(url)
#         self.visited_urls.add(normalized)
    
#     def research_query(self, query_config: ResearchQuery) -> tuple[List[SearchResult], List[ExtractedContent]]:
#         """
#         Research a single query: search and extract content.
        
#         Returns:
#             Tuple of (search_results, extracted_content)
#         """
#         # Step 1: Search
#         search_results = self.search_engine.search(query_config)
        
#         if not search_results:
#             logger.warning(f"No search results for: {query_config.query}")
#             return [], []
        
#         # Step 2: Extract content from each result
#         extracted_content = []
        
#         for idx, result in enumerate(search_results[:self.max_pages_per_query]):
#             if not self._should_fetch(result.url):
#                 continue
            
#             self._mark_visited(result.url)
            
#             logger.info(f"Extracting content [{idx+1}/{len(search_results)}]: {result.url}")
            
#             # Fetch page
#             html, error = self.fetcher.fetch(result.url)
            
#             if error:
#                 extracted = ExtractedContent(
#                     url=result.url,
#                     title=result.title,
#                     content="",
#                     word_count=0,
#                     extraction_method="failed",
#                     error=error
#                 )
#                 extracted_content.append(extracted)
#                 continue
            
#             # Extract content
#             try:
#                 title, content, method = self.extractor.extract(html, result.url)
#                 word_count = len(content.split())
                
#                 # Extract metadata
#                 metadata = {
#                     'search_position': result.position,
#                     'search_query': result.query,
#                     'original_title': result.title,
#                     'snippet': result.snippet
#                 }
                
#                 extracted = ExtractedContent(
#                     url=result.url,
#                     title=title or result.title,
#                     content=content,
#                     word_count=word_count,
#                     extraction_method=method,
#                     metadata=metadata
#                 )
                
#                 if word_count >= self.min_content_words:
#                     extracted_content.append(extracted)
#                     logger.info(f"  ✓ Extracted {word_count} words using {method}")
#                 else:
#                     logger.warning(f"  ✗ Content too short ({word_count} words), skipping")
#                     extracted.error = f"Content too short ({word_count} < {self.min_content_words} words)"
#                     extracted_content.append(extracted)
                
#             except Exception as e:
#                 logger.error(f"  ✗ Extraction failed: {e}")
#                 extracted = ExtractedContent(
#                     url=result.url,
#                     title=result.title,
#                     content="",
#                     word_count=0,
#                     extraction_method="failed",
#                     error=str(e)
#                 )
#                 extracted_content.append(extracted)
        
#         return search_results, extracted_content
    
#     def research(
#         self,
#         topic: str,
#         queries: Optional[List[str]] = None,
#         max_results_per_query: int = 10
#     ) -> ResearchReport:
#         """
#         Perform comprehensive research on a topic.
        
#         Args:
#             topic: Main research topic
#             queries: List of search queries (auto-generated if None)
#             max_results_per_query: Maximum results per query
            
#         Returns:
#             ResearchReport with all findings
#         """
#         start_time = time.time()
        
#         logger.info(f"="*80)
#         logger.info(f"Starting research on: {topic}")
#         logger.info(f"="*80)
        
#         # Auto-generate queries if not provided
#         if not queries:
#             queries = [topic]
#             logger.info(f"Using single query: {topic}")
#         else:
#             logger.info(f"Using {len(queries)} queries")
        
#         all_search_results = []
#         all_extracted_content = []
        
#         # Research each query
#         for query_idx, query in enumerate(queries):
#             logger.info(f"\n--- Query {query_idx + 1}/{len(queries)}: {query} ---")
            
#             query_config = ResearchQuery(
#                 query=query,
#                 max_results=max_results_per_query
#             )
            
#             search_results, extracted_content = self.research_query(query_config)
            
#             all_search_results.extend(search_results)
#             all_extracted_content.extend(extracted_content)
        
#         # Calculate statistics
#         elapsed_time = time.time() - start_time
#         successful_extractions = sum(1 for c in all_extracted_content if c.word_count >= self.min_content_words)
#         total_words = sum(c.word_count for c in all_extracted_content)
        
#         metadata = {
#             'total_queries': len(queries),
#             'total_search_results': len(all_search_results),
#             'total_pages_fetched': len(all_extracted_content),
#             'successful_extractions': successful_extractions,
#             'total_words_extracted': total_words,
#             'elapsed_time_seconds': round(elapsed_time, 2),
#             'min_content_words': self.min_content_words
#         }
        
#         logger.info(f"\n{'='*80}")
#         logger.info(f"Research Complete!")
#         logger.info(f"{'='*80}")
#         logger.info(f"Queries: {metadata['total_queries']}")
#         logger.info(f"Search Results: {metadata['total_search_results']}")
#         logger.info(f"Pages Fetched: {metadata['total_pages_fetched']}")
#         logger.info(f"Successful Extractions: {metadata['successful_extractions']}")
#         logger.info(f"Total Words: {metadata['total_words_extracted']:,}")
#         logger.info(f"Time Elapsed: {metadata['elapsed_time_seconds']}s")
#         logger.info(f"{'='*80}\n")
        
#         return ResearchReport(
#             topic=topic,
#             queries=queries,
#             search_results=all_search_results,
#             extracted_content=all_extracted_content,
#             metadata=metadata
#         )


# # ----------------------------------------------------------------------
# # Output Formatters
# # ----------------------------------------------------------------------
# class OutputFormatter:
#     """Format research reports in various formats."""
    
#     @staticmethod
#     def to_json(report: ResearchReport, filepath: Path):
#         """Save report as JSON."""
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
#         logger.info(f"JSON report saved: {filepath}")
    
#     @staticmethod
#     def to_markdown(report: ResearchReport, filepath: Path):
#         """Save report as Markdown."""
#         md = []
        
#         # Header
#         md.append(f"# Research Report: {report.topic}\n")
#         md.append(f"**Generated:** {report.timestamp}\n")
#         md.append(f"---\n")
        
#         # Metadata
#         md.append("## Research Metadata\n")
#         for key, value in report.metadata.items():
#             md.append(f"- **{key.replace('_', ' ').title()}:** {value}")
#         md.append("\n---\n")
        
#         # Queries
#         md.append("## Search Queries\n")
#         for idx, query in enumerate(report.queries, 1):
#             md.append(f"{idx}. {query}")
#         md.append("\n---\n")
        
#         # Search Results Summary
#         md.append("## Search Results Summary\n")
#         for idx, result in enumerate(report.search_results, 1):
#             md.append(f"### [{idx}] {result.title}\n")
#             md.append(f"- **URL:** {result.url}")
#             md.append(f"- **Query:** {result.query}")
#             md.append(f"- **Snippet:** {result.snippet[:200]}...")
#             md.append("")
#         md.append("\n---\n")
        
#         # Extracted Content
#         md.append("## Extracted Content\n")
        
#         successful_content = [c for c in report.extracted_content if c.word_count >= report.metadata['min_content_words']]
        
#         for idx, content in enumerate(successful_content, 1):
#             md.append(f"### Source {idx}: {content.title}\n")
#             md.append(f"**URL:** {content.url}\n")
#             md.append(f"**Word Count:** {content.word_count:,}\n")
#             md.append(f"**Extraction Method:** {content.extraction_method}\n")
            
#             if content.metadata:
#                 md.append(f"**Search Query:** {content.metadata.get('search_query', 'N/A')}\n")
            
#             md.append("#### Content\n")
#             md.append("```")
#             # Truncate very long content for readability
#             if len(content.content) > 5000:
#                 md.append(content.content[:5000] + "\n\n... [Content truncated] ...")
#             else:
#                 md.append(content.content)
#             md.append("```\n")
#             md.append("---\n")
        
#         # Failed extractions
#         failed_content = [c for c in report.extracted_content if c.error]
#         if failed_content:
#             md.append("## Failed Extractions\n")
#             for content in failed_content:
#                 md.append(f"- **{content.url}:** {content.error}")
#             md.append("")
        
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(md))
#         logger.info(f"Markdown report saved: {filepath}")
    
#     @staticmethod
#     def to_html(report: ResearchReport, filepath: Path):
#         """Save report as HTML."""
#         html = []
        
#         html.append("<!DOCTYPE html>")
#         html.append("<html lang='en'>")
#         html.append("<head>")
#         html.append("  <meta charset='UTF-8'>")
#         html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
#         html.append(f"  <title>Research: {report.topic}</title>")
#         html.append("  <style>")
#         html.append("    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }")
#         html.append("    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
#         html.append("    h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }")
#         html.append("    h3 { color: #7f8c8d; }")
#         html.append("    .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }")
#         html.append("    .metadata-item { margin: 5px 0; }")
#         html.append("    .source { background: #fff; border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }")
#         html.append("    .url { color: #3498db; word-break: break-all; }")
#         html.append("    .content { background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 10px 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.9em; }")
#         html.append("    .error { color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 3px; }")
#         html.append("    .stat { display: inline-block; margin: 10px 15px 10px 0; }")
#         html.append("    .query-list { background: #fff; padding: 15px; border-radius: 5px; }")
#         html.append("  </style>")
#         html.append("</head>")
#         html.append("<body>")
        
#         # Header
#         html.append(f"  <h1>Research Report: {report.topic}</h1>")
#         html.append(f"  <p><strong>Generated:</strong> {report.timestamp}</p>")
        
#         # Metadata
#         html.append("  <div class='metadata'>")
#         html.append("    <h2>Research Statistics</h2>")
#         for key, value in report.metadata.items():
#             html.append(f"    <div class='metadata-item'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>")
#         html.append("  </div>")
        
#         # Queries
#         html.append("  <h2>Search Queries</h2>")
#         html.append("  <div class='query-list'>")
#         html.append("    <ol>")
#         for query in report.queries:
#             html.append(f"      <li>{query}</li>")
#         html.append("    </ol>")
#         html.append("  </div>")
        
#         # Extracted Content
#         html.append("  <h2>Extracted Content</h2>")
        
#         successful_content = [c for c in report.extracted_content if c.word_count >= report.metadata['min_content_words']]
        
#         for idx, content in enumerate(successful_content, 1):
#             html.append("  <div class='source'>")
#             html.append(f"    <h3>Source {idx}: {content.title}</h3>")
#             html.append(f"    <p><strong>URL:</strong> <a href='{content.url}' class='url' target='_blank'>{content.url}</a></p>")
#             html.append(f"    <div class='stat'><strong>Word Count:</strong> {content.word_count:,}</div>")
#             html.append(f"    <div class='stat'><strong>Method:</strong> {content.extraction_method}</div>")
            
#             if content.metadata:
#                 html.append(f"    <div class='stat'><strong>Query:</strong> {content.metadata.get('search_query', 'N/A')}</div>")
            
#             html.append("    <h4>Content</h4>")
#             html.append("    <div class='content'>")
#             # Escape HTML and truncate if needed
#             escaped_content = content.content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
#             if len(escaped_content) > 10000:
#                 html.append(escaped_content[:10000] + "\n\n... [Content truncated for display] ...")
#             else:
#                 html.append(escaped_content)
#             html.append("    </div>")
#             html.append("  </div>")
        
#         # Failed extractions
#         failed_content = [c for c in report.extracted_content if c.error]
#         if failed_content:
#             html.append("  <h2>Failed Extractions</h2>")
#             for content in failed_content:
#                 html.append(f"  <div class='error'><strong>{content.url}:</strong> {content.error}</div>")
        
#         html.append("</body>")
#         html.append("</html>")
        
#         with open(filepath, 'w', encoding='utf-8') as f:
#             f.write('\n'.join(html))
#         logger.info(f"HTML report saved: {filepath}")


# # ----------------------------------------------------------------------
# # CLI Interface
# # ----------------------------------------------------------------------
# def main():
#     """Command-line interface for the research engine."""
#     import argparse
    
#     parser = argparse.ArgumentParser(
#         description="Web Research Engine - Comprehensive web-based research tool",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Research a single topic
#   python web_research_engine.py "artificial intelligence trends 2024"
  
#   # Research with multiple specific queries
#   python web_research_engine.py "machine learning" --queries "deep learning frameworks" "neural network architectures" "ML deployment best practices"
  
#   # Customize output and limits
#   python web_research_engine.py "climate change" --max-results 15 --output climate_research --format all
  
#   # Quick research with fewer pages
#   python web_research_engine.py "python async programming" --max-results 5 --max-pages 3
#         """
#     )
    
#     parser.add_argument(
#         "topic",
#         help="Main research topic"
#     )
    
#     parser.add_argument(
#         "--queries",
#         nargs='+',
#         help="Specific search queries (default: use topic as single query)"
#     )
    
#     parser.add_argument(
#         "--max-results",
#         type=int,
#         default=10,
#         help="Maximum search results per query (default: 10)"
#     )
    
#     parser.add_argument(
#         "--max-pages",
#         type=int,
#         default=10,
#         help="Maximum pages to fetch per query (default: 10)"
#     )
    
#     parser.add_argument(
#         "--timeout",
#         type=int,
#         default=30,
#         help="Request timeout in seconds (default: 30)"
#     )
    
#     parser.add_argument(
#         "--rate-limit",
#         type=float,
#         default=1.0,
#         help="Delay between requests in seconds (default: 1.0)"
#     )
    
#     parser.add_argument(
#         "--min-words",
#         type=int,
#         default=100,
#         help="Minimum words for valid content (default: 100)"
#     )
    
#     parser.add_argument(
#         "--output",
#         default="research_report",
#         help="Output filename (without extension, default: research_report)"
#     )
    
#     parser.add_argument(
#         "--format",
#         choices=['json', 'markdown', 'html', 'all'],
#         default='all',
#         help="Output format (default: all)"
#     )
    
#     args = parser.parse_args()
    
#     # Check dependencies
#     if not DDGS_AVAILABLE:
#         logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
#         return 1
    
#     if not REQUESTS_AVAILABLE:
#         logger.error("requests not installed. Run: pip install requests")
#         return 1
    
#     if not BS4_AVAILABLE:
#         logger.warning("beautifulsoup4 not installed (recommended). Run: pip install beautifulsoup4")
    
#     # Initialize research engine
#     engine = ResearchEngine(
#         max_pages_per_query=args.max_pages,
#         timeout=args.timeout,
#         rate_limit_delay=args.rate_limit,
#         min_content_words=args.min_words
#     )
    
#     # Perform research
#     try:
#         report = engine.research(
#             topic=args.topic,
#             queries=args.queries,
#             max_results_per_query=args.max_results
#         )
        
#         # Save outputs
#         output_dir = Path("research_outputs")
#         output_dir.mkdir(exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         base_filename = f"{args.output}_{timestamp}"
        
#         if args.format in ['json', 'all']:
#             OutputFormatter.to_json(report, output_dir / f"{base_filename}.json")
        
#         if args.format in ['markdown', 'all']:
#             OutputFormatter.to_markdown(report, output_dir / f"{base_filename}.md")
        
#         if args.format in ['html', 'all']:
#             OutputFormatter.to_html(report, output_dir / f"{base_filename}.html")
        
#         print(f"\n{'='*80}")
#         print("Research completed successfully!")
#         print(f"{'='*80}")
#         print(f"Topic: {report.topic}")
#         print(f"Queries: {len(report.queries)}")
#         print(f"Search Results: {report.metadata['total_search_results']}")
#         print(f"Successful Extractions: {report.metadata['successful_extractions']}")
#         print(f"Total Words: {report.metadata['total_words_extracted']:,}")
#         print(f"Output Directory: {output_dir.absolute()}")
#         print(f"{'='*80}\n")
        
#         return 0
        
#     except KeyboardInterrupt:
#         logger.info("\nResearch interrupted by user")
#         return 130
    
#     except Exception as e:
#         logger.error(f"Research failed: {e}", exc_info=True)
#         return 1


# if __name__ == "__main__":
#     exit(main())























# --------------------------------------------------------------------------
# --------------------------------------------------------------------------





















#!/usr/bin/env python3
"""
Web Research Engine - Production-Grade Implementation

A comprehensive web-based research tool that performs deep research by:
1. Querying search engines (DuckDuckGo)
2. Retrieving and processing multiple relevant results
3. Extracting substantial content from each source
4. Aggregating information into structured outputs
5. Producing clear, verifiable research reports

Features:
- Multi-query research workflows
- Full-page content extraction
- Intelligent rate limiting and retry logic
- Multiple output formats (JSON, Markdown, HTML)
- Robust error handling and logging
- Progress tracking and metrics
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse, urljoin
from html.parser import HTMLParser
import hashlib

# Third-party imports
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old package name
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
        import warnings
        warnings.warn(
            "duckduckgo_search is deprecated. Please upgrade: pip uninstall duckduckgo-search && pip install ddgs",
            DeprecationWarning,
            stacklevel=2
        )
    except ImportError:
        DDGS_AVAILABLE = False
        print("Warning: ddgs not installed. Install with: pip install ddgs")

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not installed. Install with: pip install requests")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    # Readability is optional - we have fallback extraction

# ----------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------
# Configure logging with UTF-8 encoding for cross-platform compatibility
import sys

# Create file handler with UTF-8 encoding
file_handler = logging.FileHandler("web_research.log", encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Create console handler with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    try:
        # Reconfigure stdout to use UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        # Fallback for older Python versions or if reconfigure fails
        # Replace Unicode characters with ASCII equivalents in messages
        pass

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Unicode-safe logging helper
# ----------------------------------------------------------------------
def safe_log(log_func, message, use_unicode=True):
    """
    Safely log messages with Unicode characters.
    Falls back to ASCII on Windows if Unicode fails.
    """
    if sys.platform == 'win32' and not use_unicode:
        # Replace Unicode symbols with ASCII equivalents
        message = message.replace('✓', '[OK]').replace('✗', '[X]')
    
    try:
        log_func(message)
    except UnicodeEncodeError:
        # Fallback: replace Unicode with ASCII
        ascii_message = message.replace('✓', '[OK]').replace('✗', '[X]')
        log_func(ascii_message)


# ----------------------------------------------------------------------
# Data Models
# ----------------------------------------------------------------------
@dataclass
class SearchResult:
    """Represents a single search result."""
    title: str
    url: str
    snippet: str
    position: int
    query: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExtractedContent:
    """Represents extracted content from a web page."""
    url: str
    title: str
    content: str
    word_count: int
    extraction_method: str
    metadata: Dict[str, any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    

@dataclass
class ResearchQuery:
    """Configuration for a research query."""
    query: str
    max_results: int = 10
    time_range: Optional[str] = None  # 'd' (day), 'w' (week), 'm' (month), 'y' (year)
    region: str = 'wt-wt'  # worldwide


@dataclass
class ResearchReport:
    """Complete research report with all findings."""
    topic: str
    queries: List[str]
    search_results: List[SearchResult]
    extracted_content: List[ExtractedContent]
    metadata: Dict[str, any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'topic': self.topic,
            'queries': self.queries,
            'search_results': [asdict(r) for r in self.search_results],
            'extracted_content': [asdict(c) for c in self.extracted_content],
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


# ----------------------------------------------------------------------
# HTML Content Extractor
# ----------------------------------------------------------------------
class ContentExtractor:
    """Extract meaningful text content from HTML."""
    
    # Tags that typically contain main content
    CONTENT_TAGS = ['article', 'main', 'section', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    # Tags to remove
    SKIP_TAGS = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    @staticmethod
    def extract_with_readability(html: str, url: str) -> tuple[str, str, str]:
        """Extract content using readability library."""
        if not READABILITY_AVAILABLE:
            raise ImportError("Readability library not available")
        
        doc = Document(html)
        title = doc.title()
        content_html = doc.summary()
        
        # Convert HTML to text
        soup = BeautifulSoup(content_html, 'html.parser')
        content = soup.get_text(separator='\n', strip=True)
        
        return title, content, "readability"
    
    @staticmethod
    def extract_with_beautifulsoup(html: str, url: str) -> tuple[str, str, str]:
        """Extract content using BeautifulSoup with heuristics."""
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup library not available")
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted tags
        for tag in soup.find_all(ContentExtractor.SKIP_TAGS):
            tag.decompose()
        
        # Try to find title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        elif soup.find('h1'):
            title = soup.find('h1').get_text(strip=True)
        
        # Try to find main content container
        content_container = None
        
        # Priority order for content containers
        for selector in ['article', 'main', '[role="main"]', '.content', '#content', '.post', '.article']:
            content_container = soup.select_one(selector)
            if content_container:
                break
        
        # Fall back to body
        if not content_container:
            content_container = soup.find('body')
        
        if not content_container:
            content_container = soup
        
        # Extract text
        content = content_container.get_text(separator='\n', strip=True)
        
        return title, content, "beautifulsoup"
    
    @staticmethod
    def extract_basic(html: str, url: str) -> tuple[str, str, str]:
        """Basic HTML tag stripping as fallback."""
        # Simple HTML tag removal
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else urlparse(url).netloc
        
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all HTML tags
        content = re.sub(r'<[^>]+>', ' ', html)
        
        # Decode HTML entities
        content = content.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        return title, content, "basic"
    
    @classmethod
    def extract(cls, html: str, url: str) -> tuple[str, str, str]:
        """
        Extract content using best available method.
        Returns: (title, content, method_used)
        """
        methods = []
        
        if READABILITY_AVAILABLE:
            methods.append(cls.extract_with_readability)
        if BS4_AVAILABLE:
            methods.append(cls.extract_with_beautifulsoup)
        methods.append(cls.extract_basic)
        
        for method in methods:
            try:
                title, content, method_name = method(html, url)
                content = cls.clean_text(content)
                if content and len(content) > 100:  # Minimum content threshold
                    return title, content, method_name
            except Exception as e:
                logger.debug(f"Extraction method {method.__name__} failed: {e}")
                continue
        
        # Final fallback
        return urlparse(url).netloc, "", "failed"


# ----------------------------------------------------------------------
# Web Fetcher with Retry Logic
# ----------------------------------------------------------------------
class WebFetcher:
    """Robust web page fetcher with retry logic and rate limiting."""
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, rate_limit_delay: float = 1.0):
        """
        Initialize web fetcher.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_delay: Delay between requests in seconds
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required")
        
        # Create session with retry logic
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch(self, url: str) -> tuple[Optional[str], Optional[str]]:
        """
        Fetch a URL and return HTML content.
        
        Returns:
            Tuple of (html_content, error_message)
        """
        self._rate_limit()
        
        try:
            logger.info(f"Fetching: {url}")
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                return None, f"Non-HTML content type: {content_type}"
            
            return response.text, None
            
        except requests.exceptions.Timeout:
            error = f"Timeout after {self.timeout}s"
            logger.warning(f"Failed to fetch {url}: {error}")
            return None, error
        
        except requests.exceptions.HTTPError as e:
            error = f"HTTP {e.response.status_code}"
            logger.warning(f"Failed to fetch {url}: {error}")
            return None, error
        
        except requests.exceptions.RequestException as e:
            error = str(e)
            logger.warning(f"Failed to fetch {url}: {error}")
            return None, error
        
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to fetch {url}: {error}")
            return None, error


# ----------------------------------------------------------------------
# Search Engine Interface
# ----------------------------------------------------------------------
class SearchEngine:
    """Interface for web search using DuckDuckGo."""
    
    def __init__(self):
        """Initialize search engine."""
        if not DDGS_AVAILABLE:
            raise ImportError("ddgs library is required. Install with: pip install ddgs")
        
        self.ddgs = DDGS()
    
    def search(self, query_config: ResearchQuery) -> List[SearchResult]:
        """
        Perform a search and return results with retry logic.
        
        Args:
            query_config: Research query configuration
            
        Returns:
            List of SearchResult objects
        """
        results = []
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Searching: '{query_config.query}' (max_results={query_config.max_results})")
                
                # Try new API signature first, then fall back to old
                try:
                    search_results = self.ddgs.text(
                        query_config.query,
                        max_results=query_config.max_results
                    )
                except (TypeError, AttributeError):
                    # Fallback for older package API
                    search_results = self.ddgs.text(
                        keywords=query_config.query,
                        region=query_config.region,
                        max_results=query_config.max_results
                    )
                
                # Consume iterator carefully
                result_list = []
                if search_results:
                    try:
                        for item in search_results:
                            if item:
                                result_list.append(item)
                            if len(result_list) >= query_config.max_results:
                                break
                    except StopIteration:
                        pass
                    except Exception as e:
                        logger.debug(f"Iterator error: {e}")
                
                # Process results
                for idx, result in enumerate(result_list):
                    if not result or not isinstance(result, dict):
                        continue
                    
                    # Extract URL with multiple fallbacks
                    url = (result.get('href') or 
                           result.get('link') or 
                           result.get('url') or '')
                    
                    if not url:
                        continue
                    
                    search_result = SearchResult(
                        title=result.get('title', result.get('name', 'No Title')),
                        url=url,
                        snippet=result.get('body', result.get('snippet', result.get('description', ''))),
                        position=idx + 1,
                        query=query_config.query
                    )
                    
                    results.append(search_result)
                    logger.debug(f"  [{idx+1}] {search_result.title}")
                
                logger.info(f"Found {len(results)} results for '{query_config.query}'")
                
                # If we got results, return them
                if results:
                    return results
                
                # If no results and we have retries left, wait and try again
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"No results on attempt {attempt + 1}/{max_retries}. Waiting {delay}s before retry...")
                    time.sleep(delay)
                else:
                    logger.warning(f"No results after {max_retries} attempts")
                
            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{max_retries} failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Search failed after {max_retries} attempts")
        
        return results


# ----------------------------------------------------------------------
# Research Engine - Main Orchestrator
# ----------------------------------------------------------------------
class ResearchEngine:
    """Main research engine that orchestrates search, extraction, and aggregation."""
    
    def __init__(
        self,
        max_pages_per_query: int = 10,
        timeout: int = 30,
        rate_limit_delay: float = 1.0,
        min_content_words: int = 100
    ):
        """
        Initialize research engine.
        
        Args:
            max_pages_per_query: Maximum pages to fetch per search query
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between requests in seconds
            min_content_words: Minimum words for valid content
        """
        self.max_pages_per_query = max_pages_per_query
        self.min_content_words = min_content_words
        
        self.search_engine = SearchEngine()
        self.fetcher = WebFetcher(timeout=timeout, rate_limit_delay=rate_limit_delay)
        self.extractor = ContentExtractor()
        
        self.visited_urls: Set[str] = set()
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urlparse(url)
        # Remove fragment and common tracking parameters
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return normalized.lower()
    
    def _should_fetch(self, url: str) -> bool:
        """Check if URL should be fetched (not already visited)."""
        normalized = self._normalize_url(url)
        
        if normalized in self.visited_urls:
            logger.debug(f"Skipping duplicate: {url}")
            return False
        
        return True
    
    def _mark_visited(self, url: str):
        """Mark URL as visited."""
        normalized = self._normalize_url(url)
        self.visited_urls.add(normalized)
    
    def research_query(self, query_config: ResearchQuery) -> tuple[List[SearchResult], List[ExtractedContent]]:
        """
        Research a single query: search and extract content.
        
        Returns:
            Tuple of (search_results, extracted_content)
        """
        # Step 1: Search
        search_results = self.search_engine.search(query_config)
        
        if not search_results:
            logger.warning(f"No search results for: {query_config.query}")
            return [], []
        
        # Step 2: Extract content from each result
        extracted_content = []
        
        for idx, result in enumerate(search_results[:self.max_pages_per_query]):
            if not self._should_fetch(result.url):
                continue
            
            self._mark_visited(result.url)
            
            logger.info(f"Extracting content [{idx+1}/{len(search_results)}]: {result.url}")
            
            # Fetch page
            html, error = self.fetcher.fetch(result.url)
            
            if error:
                extracted = ExtractedContent(
                    url=result.url,
                    title=result.title,
                    content="",
                    word_count=0,
                    extraction_method="failed",
                    error=error
                )
                extracted_content.append(extracted)
                continue
            
            # Extract content
            try:
                title, content, method = self.extractor.extract(html, result.url)
                word_count = len(content.split())
                
                # Extract metadata
                metadata = {
                    'search_position': result.position,
                    'search_query': result.query,
                    'original_title': result.title,
                    'snippet': result.snippet
                }
                
                extracted = ExtractedContent(
                    url=result.url,
                    title=title or result.title,
                    content=content,
                    word_count=word_count,
                    extraction_method=method,
                    metadata=metadata
                )
                
                if word_count >= self.min_content_words:
                    extracted_content.append(extracted)
                    safe_log(logger.info, f"  ✓ Extracted {word_count} words using {method}")
                else:
                    safe_log(logger.warning, f"  ✗ Content too short ({word_count} words), skipping")
                    extracted.error = f"Content too short ({word_count} < {self.min_content_words} words)"
                    extracted_content.append(extracted)
                
            except Exception as e:
                safe_log(logger.error, f"  ✗ Extraction failed: {e}")
                extracted = ExtractedContent(
                    url=result.url,
                    title=result.title,
                    content="",
                    word_count=0,
                    extraction_method="failed",
                    error=str(e)
                )
                extracted_content.append(extracted)
        
        return search_results, extracted_content
    
    def research(
        self,
        topic: str,
        queries: Optional[List[str]] = None,
        max_results_per_query: int = 10
    ) -> ResearchReport:
        """
        Perform comprehensive research on a topic.
        
        Args:
            topic: Main research topic
            queries: List of search queries (auto-generated if None)
            max_results_per_query: Maximum results per query
            
        Returns:
            ResearchReport with all findings
        """
        start_time = time.time()
        
        logger.info(f"="*80)
        logger.info(f"Starting research on: {topic}")
        logger.info(f"="*80)
        
        # Auto-generate queries if not provided
        if not queries:
            queries = [topic]
            logger.info(f"Using single query: {topic}")
        else:
            logger.info(f"Using {len(queries)} queries")
        
        all_search_results = []
        all_extracted_content = []
        
        # Research each query
        for query_idx, query in enumerate(queries):
            logger.info(f"\n--- Query {query_idx + 1}/{len(queries)}: {query} ---")
            
            query_config = ResearchQuery(
                query=query,
                max_results=max_results_per_query
            )
            
            search_results, extracted_content = self.research_query(query_config)
            
            all_search_results.extend(search_results)
            all_extracted_content.extend(extracted_content)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        successful_extractions = sum(1 for c in all_extracted_content if c.word_count >= self.min_content_words)
        total_words = sum(c.word_count for c in all_extracted_content)
        
        metadata = {
            'total_queries': len(queries),
            'total_search_results': len(all_search_results),
            'total_pages_fetched': len(all_extracted_content),
            'successful_extractions': successful_extractions,
            'total_words_extracted': total_words,
            'elapsed_time_seconds': round(elapsed_time, 2),
            'min_content_words': self.min_content_words
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Research Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Queries: {metadata['total_queries']}")
        logger.info(f"Search Results: {metadata['total_search_results']}")
        logger.info(f"Pages Fetched: {metadata['total_pages_fetched']}")
        logger.info(f"Successful Extractions: {metadata['successful_extractions']}")
        logger.info(f"Total Words: {metadata['total_words_extracted']:,}")
        logger.info(f"Time Elapsed: {metadata['elapsed_time_seconds']}s")
        logger.info(f"{'='*80}\n")
        
        return ResearchReport(
            topic=topic,
            queries=queries,
            search_results=all_search_results,
            extracted_content=all_extracted_content,
            metadata=metadata
        )


# ----------------------------------------------------------------------
# Output Formatters
# ----------------------------------------------------------------------
class OutputFormatter:
    """Format research reports in various formats."""
    
    @staticmethod
    def to_json(report: ResearchReport, filepath: Path):
        """Save report as JSON."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"JSON report saved: {filepath}")
    
    @staticmethod
    def to_markdown(report: ResearchReport, filepath: Path):
        """Save report as Markdown."""
        md = []
        
        # Header
        md.append(f"# Research Report: {report.topic}\n")
        md.append(f"**Generated:** {report.timestamp}\n")
        md.append(f"---\n")
        
        # Metadata
        md.append("## Research Metadata\n")
        for key, value in report.metadata.items():
            md.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        md.append("\n---\n")
        
        # Queries
        md.append("## Search Queries\n")
        for idx, query in enumerate(report.queries, 1):
            md.append(f"{idx}. {query}")
        md.append("\n---\n")
        
        # Search Results Summary
        md.append("## Search Results Summary\n")
        for idx, result in enumerate(report.search_results, 1):
            md.append(f"### [{idx}] {result.title}\n")
            md.append(f"- **URL:** {result.url}")
            md.append(f"- **Query:** {result.query}")
            md.append(f"- **Snippet:** {result.snippet[:200]}...")
            md.append("")
        md.append("\n---\n")
        
        # Extracted Content
        md.append("## Extracted Content\n")
        
        successful_content = [c for c in report.extracted_content if c.word_count >= report.metadata['min_content_words']]
        
        for idx, content in enumerate(successful_content, 1):
            md.append(f"### Source {idx}: {content.title}\n")
            md.append(f"**URL:** {content.url}\n")
            md.append(f"**Word Count:** {content.word_count:,}\n")
            md.append(f"**Extraction Method:** {content.extraction_method}\n")
            
            if content.metadata:
                md.append(f"**Search Query:** {content.metadata.get('search_query', 'N/A')}\n")
            
            md.append("#### Content\n")
            md.append("```")
            # Truncate very long content for readability
            if len(content.content) > 5000:
                md.append(content.content[:5000] + "\n\n... [Content truncated] ...")
            else:
                md.append(content.content)
            md.append("```\n")
            md.append("---\n")
        
        # Failed extractions
        failed_content = [c for c in report.extracted_content if c.error]
        if failed_content:
            md.append("## Failed Extractions\n")
            for content in failed_content:
                md.append(f"- **{content.url}:** {content.error}")
            md.append("")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))
        logger.info(f"Markdown report saved: {filepath}")
    
    @staticmethod
    def to_html(report: ResearchReport, filepath: Path):
        """Save report as HTML."""
        html = []
        
        html.append("<!DOCTYPE html>")
        html.append("<html lang='en'>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html.append(f"  <title>Research: {report.topic}</title>")
        html.append("  <style>")
        html.append("    body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }")
        html.append("    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }")
        html.append("    h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }")
        html.append("    h3 { color: #7f8c8d; }")
        html.append("    .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }")
        html.append("    .metadata-item { margin: 5px 0; }")
        html.append("    .source { background: #fff; border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }")
        html.append("    .url { color: #3498db; word-break: break-all; }")
        html.append("    .content { background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; margin: 10px 0; white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.9em; }")
        html.append("    .error { color: #e74c3c; background: #fadbd8; padding: 10px; border-radius: 3px; }")
        html.append("    .stat { display: inline-block; margin: 10px 15px 10px 0; }")
        html.append("    .query-list { background: #fff; padding: 15px; border-radius: 5px; }")
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")
        
        # Header
        html.append(f"  <h1>Research Report: {report.topic}</h1>")
        html.append(f"  <p><strong>Generated:</strong> {report.timestamp}</p>")
        
        # Metadata
        html.append("  <div class='metadata'>")
        html.append("    <h2>Research Statistics</h2>")
        for key, value in report.metadata.items():
            html.append(f"    <div class='metadata-item'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>")
        html.append("  </div>")
        
        # Queries
        html.append("  <h2>Search Queries</h2>")
        html.append("  <div class='query-list'>")
        html.append("    <ol>")
        for query in report.queries:
            html.append(f"      <li>{query}</li>")
        html.append("    </ol>")
        html.append("  </div>")
        
        # Extracted Content
        html.append("  <h2>Extracted Content</h2>")
        
        successful_content = [c for c in report.extracted_content if c.word_count >= report.metadata['min_content_words']]
        
        for idx, content in enumerate(successful_content, 1):
            html.append("  <div class='source'>")
            html.append(f"    <h3>Source {idx}: {content.title}</h3>")
            html.append(f"    <p><strong>URL:</strong> <a href='{content.url}' class='url' target='_blank'>{content.url}</a></p>")
            html.append(f"    <div class='stat'><strong>Word Count:</strong> {content.word_count:,}</div>")
            html.append(f"    <div class='stat'><strong>Method:</strong> {content.extraction_method}</div>")
            
            if content.metadata:
                html.append(f"    <div class='stat'><strong>Query:</strong> {content.metadata.get('search_query', 'N/A')}</div>")
            
            html.append("    <h4>Content</h4>")
            html.append("    <div class='content'>")
            # Escape HTML and truncate if needed
            escaped_content = content.content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if len(escaped_content) > 10000:
                html.append(escaped_content[:10000] + "\n\n... [Content truncated for display] ...")
            else:
                html.append(escaped_content)
            html.append("    </div>")
            html.append("  </div>")
        
        # Failed extractions
        failed_content = [c for c in report.extracted_content if c.error]
        if failed_content:
            html.append("  <h2>Failed Extractions</h2>")
            for content in failed_content:
                html.append(f"  <div class='error'><strong>{content.url}:</strong> {content.error}</div>")
        
        html.append("</body>")
        html.append("</html>")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))
        logger.info(f"HTML report saved: {filepath}")


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------
def main():
    """Command-line interface for the research engine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Web Research Engine - Comprehensive web-based research tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Research a single topic
  python web_research_engine.py "artificial intelligence trends 2024"
  
  # Research with multiple specific queries
  python web_research_engine.py "machine learning" --queries "deep learning frameworks" "neural network architectures" "ML deployment best practices"
  
  # Customize output and limits
  python web_research_engine.py "climate change" --max-results 15 --output climate_research --format all
  
  # Quick research with fewer pages
  python web_research_engine.py "python async programming" --max-results 5 --max-pages 3
        """
    )
    
    parser.add_argument(
        "topic",
        help="Main research topic"
    )
    
    parser.add_argument(
        "--queries",
        nargs='+',
        help="Specific search queries (default: use topic as single query)"
    )
    
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum search results per query (default: 10)"
    )
    
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum pages to fetch per query (default: 10)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--min-words",
        type=int,
        default=100,
        help="Minimum words for valid content (default: 100)"
    )
    
    parser.add_argument(
        "--output",
        default="research_report",
        help="Output filename (without extension, default: research_report)"
    )
    
    parser.add_argument(
        "--format",
        choices=['json', 'markdown', 'html', 'all'],
        default='all',
        help="Output format (default: all)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not DDGS_AVAILABLE:
        logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return 1
    
    if not REQUESTS_AVAILABLE:
        logger.error("requests not installed. Run: pip install requests")
        return 1
    
    if not BS4_AVAILABLE:
        logger.warning("beautifulsoup4 not installed (recommended). Run: pip install beautifulsoup4")
    
    # Initialize research engine
    engine = ResearchEngine(
        max_pages_per_query=args.max_pages,
        timeout=args.timeout,
        rate_limit_delay=args.rate_limit,
        min_content_words=args.min_words
    )
    
    # Perform research
    try:
        report = engine.research(
            topic=args.topic,
            queries=args.queries,
            max_results_per_query=args.max_results
        )
        
        # Save outputs
        output_dir = Path("research_outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{args.output}_{timestamp}"
        
        if args.format in ['json', 'all']:
            OutputFormatter.to_json(report, output_dir / f"{base_filename}.json")
        
        if args.format in ['markdown', 'all']:
            OutputFormatter.to_markdown(report, output_dir / f"{base_filename}.md")
        
        if args.format in ['html', 'all']:
            OutputFormatter.to_html(report, output_dir / f"{base_filename}.html")
        
        print(f"\n{'='*80}")
        print("Research completed successfully!")
        print(f"{'='*80}")
        print(f"Topic: {report.topic}")
        print(f"Queries: {len(report.queries)}")
        print(f"Search Results: {report.metadata['total_search_results']}")
        print(f"Successful Extractions: {report.metadata['successful_extractions']}")
        print(f"Total Words: {report.metadata['total_words_extracted']:,}")
        print(f"Output Directory: {output_dir.absolute()}")
        print(f"{'='*80}\n")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\nResearch interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Research failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())