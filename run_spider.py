#!/usr/bin/env python3
import sys
import json
import logging
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
from spider import LinkCollectorSpider
from urllib.parse import urlparse
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_spider.py <start_url>")
        sys.exit(1)
    
    start_url = sys.argv[1]
    max_urls = int(sys.argv[2]) if len(sys.argv) > 2 else 20  # Allow custom limit
    
    # Completely disable Scrapy logging
    configure_logging({"LOG_ENABLED": False})
    logging.getLogger('scrapy').setLevel(logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Clear any existing URLs
    LinkCollectorSpider.collected_urls = []
    
    # Debug information
    print(f"[DEBUG] Starting spider for: {start_url}", file=sys.stderr)
    print(f"[DEBUG] Max URLs to collect: {max_urls}", file=sys.stderr)
    
    # Determine site type for optimal settings
    parsed_url = urlparse(start_url)
    is_wikipedia = 'wikipedia.org' in parsed_url.netloc
    is_news_site = any(news in parsed_url.netloc.lower() for news in ['news', 'cnn', 'bbc', 'reuters'])
    
    print(f"[DEBUG] Site type - Wikipedia: {is_wikipedia}, News: {is_news_site}", file=sys.stderr)
    
    # Adaptive settings based on website type
    if is_wikipedia:
        settings = {
            'LOG_ENABLED': False,
            'LOG_LEVEL': 'ERROR',
            'DEPTH_LIMIT': 3,  # Increased depth for Wikipedia
            'DOWNLOAD_DELAY': 0.3,  # Slower for Wikipedia
            'ROBOTSTXT_OBEY': False,
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'COOKIES_ENABLED': True,
            'RETRY_ENABLED': True,
            'RETRY_TIMES': 3,
            'REDIRECT_ENABLED': True,
            'CONCURRENT_REQUESTS': 1,
            'DOWNLOAD_TIMEOUT': 45,  # Longer timeout for Wikipedia
            'DNSCACHE_ENABLED': True,
            'DNSCACHE_SIZE': 10000,
        }
    elif is_news_site:
        settings = {
            'LOG_ENABLED': False,
            'LOG_LEVEL': 'ERROR',
            'DEPTH_LIMIT': 2,
            'DOWNLOAD_DELAY': 0.2,
            'ROBOTSTXT_OBEY': False,
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'COOKIES_ENABLED': True,
            'RETRY_ENABLED': True,
            'RETRY_TIMES': 2,
            'REDIRECT_ENABLED': True,
            'CONCURRENT_REQUESTS': 2,
            'DOWNLOAD_TIMEOUT': 30,
        }
    else:
        # Regular websites (like aapgs.com)
        settings = {
            'LOG_ENABLED': False,
            'LOG_LEVEL': 'ERROR',
            'DEPTH_LIMIT': 3,  # Increased depth
            'DOWNLOAD_DELAY': 0.1,  # Faster for regular sites
            'ROBOTSTXT_OBEY': False,
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'COOKIES_ENABLED': False,
            'RETRY_ENABLED': True,  # Enable retries for regular sites too
            'RETRY_TIMES': 2,
            'REDIRECT_ENABLED': True,
            'CONCURRENT_REQUESTS': 4,  # More concurrent requests
            'DOWNLOAD_TIMEOUT': 20,
        }
    
    # Set up the crawler process
    process = CrawlerProcess(settings)
    
    try:
        print(f"[DEBUG] Starting crawler process...", file=sys.stderr)
        
        # Start the crawler
        process.crawl(LinkCollectorSpider, start_url=start_url, max_urls=max_urls)
        process.start()
        
        # Get collected URLs and ensure we have the start_url
        urls = LinkCollectorSpider.collected_urls
        print(f"[DEBUG] Spider collected {len(urls)} URLs", file=sys.stderr)
        
        if start_url not in urls:
            urls.insert(0, start_url)
            print(f"[DEBUG] Added start URL to collection", file=sys.stderr)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        print(f"[DEBUG] After deduplication: {len(unique_urls)} unique URLs", file=sys.stderr)
        
        # Apply limit
        final_urls = unique_urls[:max_urls]
        print(f"[DEBUG] Final URL count after limit: {len(final_urls)}", file=sys.stderr)
        
        # Debug: Print first few URLs
        for i, url in enumerate(final_urls[:5]):
            print(f"[DEBUG] URL {i+1}: {url}", file=sys.stderr)
        
        if len(final_urls) > 5:
            print(f"[DEBUG] ... and {len(final_urls) - 5} more URLs", file=sys.stderr)
        
        # Output ONLY the JSON to stdout - no debug messages to stdout
        print(json.dumps(final_urls))
        
    except Exception as e:
        print(f"[ERROR] Spider failed: {e}", file=sys.stderr)
        # If spider fails, return at least the start URL
        print(json.dumps([start_url]))

if __name__ == "__main__":
    main()