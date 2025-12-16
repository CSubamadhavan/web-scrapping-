from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import scrapy
from urllib.parse import urljoin, urlparse
import logging
import re

MAX_PAGES = 20  # Increased limit for better coverage

class LinkCollectorSpider(CrawlSpider):
    name = 'link_collector'
    custom_settings = {
        'LOG_ENABLED': False,
        'LOG_LEVEL': 'ERROR', 
        'DEPTH_LIMIT': 2,
        'DOWNLOAD_DELAY': 0.1,  # Slightly slower for Wikipedia
        'CONCURRENT_REQUESTS': 2,  # Reduced for stability
        'ROBOTSTXT_OBEY': False,  # Ignore robots.txt
        'COOKIES_ENABLED': True,   # Enable cookies for Wikipedia
        'RETRY_ENABLED': True,     # Enable retries
        'RETRY_TIMES': 2,
        'REDIRECT_ENABLED': True,
        'DOWNLOAD_TIMEOUT': 20,    # Increased timeout
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    collected_urls = []

    def __init__(self, start_url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [start_url]
        
        # Extract domain for allowed_domains
        parsed_url = urlparse(start_url)
        self.allowed_domains = [parsed_url.netloc]
        self.base_domain = parsed_url.netloc
        
        # Disable all logging for this spider instance
        logging.getLogger('scrapy').setLevel(logging.CRITICAL)
        
        # Enhanced rules for different website types
        if 'wikipedia.org' in self.base_domain:
            # Special rules for Wikipedia
            self.rules = (
                Rule(
                    LinkExtractor(
                        allow_domains=self.allowed_domains,
                        allow=[r'/wiki/[^:]+$'],  # Only main Wikipedia articles
                        deny=[
                            r'/wiki/File:',
                            r'/wiki/Category:',
                            r'/wiki/Template:',
                            r'/wiki/Help:',
                            r'/wiki/Special:',
                            r'/wiki/Talk:',
                            r'/wiki/User:',
                            r'/wiki/Wikipedia:',
                            r'/wiki/Portal:',
                            r'#',
                            r'\?',
                            r'/w/',
                        ],
                        deny_extensions=['pdf', 'zip', 'tar', 'gz', 'jpg', 'jpeg', 'png', 'gif', 'svg', 'ico', 'css', 'js'],
                        unique=True
                    ), 
                    callback='parse_item', 
                    follow=True
                ),
            )
        else:
            # Rules for regular websites
            deny_patterns = [r'/tag/', r'/category/', r'/author/', r'#', r'\?page=\d+$']
            
            # Add common CMS patterns to deny
            if any(cms in start_url.lower() for cms in ['wordpress', 'wp-content', 'admin']):
                deny_patterns.extend([r'/wp-admin/', r'/wp-content/'])
            
            self.rules = (
                Rule(
                    LinkExtractor(
                        allow_domains=self.allowed_domains,
                        deny_extensions=['pdf', 'zip', 'tar', 'gz', 'jpg', 'jpeg', 'png', 'gif', 'svg', 'ico', 'css', 'js'],
                        deny=deny_patterns,
                        unique=True
                    ), 
                    callback='parse_item', 
                    follow=True
                ),
            )
        
        super(LinkCollectorSpider, self)._compile_rules()

    def parse_start_url(self, response):
        """Parse the start URL and add it to collected URLs"""
        if response.url not in LinkCollectorSpider.collected_urls:
            LinkCollectorSpider.collected_urls.append(response.url)
        
        # Continue with normal parsing
        return self.parse_item(response)

    def parse_item(self, response):
        # Add current URL if not already collected
        if len(LinkCollectorSpider.collected_urls) < MAX_PAGES:
            if response.url not in LinkCollectorSpider.collected_urls:
                LinkCollectorSpider.collected_urls.append(response.url)
        
        # Continue following links until we reach MAX_PAGES
        if len(LinkCollectorSpider.collected_urls) < MAX_PAGES:
            return self.extract_links(response)
    
    def extract_links(self, response):
        """Extract additional links from the current page"""
        if 'wikipedia.org' in self.base_domain:
            # Special link extraction for Wikipedia
            links = LinkExtractor(
                allow_domains=self.allowed_domains,
                allow=[r'/wiki/[^:]+$'],
                deny=[
                    r'/wiki/File:', r'/wiki/Category:', r'/wiki/Template:',
                    r'/wiki/Help:', r'/wiki/Special:', r'/wiki/Talk:',
                    r'/wiki/User:', r'/wiki/Wikipedia:', r'/wiki/Portal:',
                    r'#', r'\?', r'/w/',
                ],
                deny_extensions=['pdf', 'zip', 'tar', 'gz', 'jpg', 'jpeg', 'png', 'gif', 'svg', 'ico', 'css', 'js'],
                unique=True
            ).extract_links(response)
        else:
            # Regular link extraction
            links = LinkExtractor(
                allow_domains=self.allowed_domains,
                deny_extensions=['pdf', 'zip', 'tar', 'gz', 'jpg', 'jpeg', 'png', 'gif', 'svg', 'ico', 'css', 'js'],
                deny=[r'/tag/', r'/category/', r'/author/', r'#', r'\?page=\d+$'],
                unique=True
            ).extract_links(response)
        
        for link in links:
            if len(LinkCollectorSpider.collected_urls) >= MAX_PAGES:
                break
            if link.url not in LinkCollectorSpider.collected_urls:
                yield scrapy.Request(
                    link.url, 
                    callback=self.parse_item, 
                    dont_filter=False,
                    meta={'download_timeout': 20}  # Per-request timeout
                )

class URLCollector:
    def __init__(self, target_url):
        self.target_url = target_url

    def collect(self):
        # Clear any existing URLs
        LinkCollectorSpider.collected_urls = []
        
        # Disable all logging
        logging.getLogger('scrapy').setLevel(logging.CRITICAL)
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Determine if this is Wikipedia or similar site
        parsed_url = urlparse(self.target_url)
        is_wikipedia = 'wikipedia.org' in parsed_url.netloc
        
        # Adjust settings based on site type
        settings = {
            'LOG_ENABLED': False,
            'LOG_LEVEL': 'ERROR',
            'DEPTH_LIMIT': 2 if is_wikipedia else 1,
            'DOWNLOAD_DELAY': 0.2 if is_wikipedia else 0.1,
            'ROBOTSTXT_OBEY': False,
            'COOKIES_ENABLED': True if is_wikipedia else False,
            'RETRY_ENABLED': True,
            'RETRY_TIMES': 2,
            'CONCURRENT_REQUESTS': 1 if is_wikipedia else 2,
            'DOWNLOAD_TIMEOUT': 25,  # Increased timeout
            'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        process = CrawlerProcess(settings)
        
        try:
            process.crawl(LinkCollectorSpider, start_url=self.target_url)
            process.start()
            
            urls = LinkCollectorSpider.collected_urls
            if self.target_url not in urls:
                urls.insert(0, self.target_url)
            
            return urls[:MAX_PAGES]  # Ensure we don't exceed limit
        except Exception as e:
            print(f"URLCollector error: {e}")
            return [self.target_url]