from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup

class SeleniumScraper:
    def __init__(self, driver_path):
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        chrome_options.add_argument("--ignore-certificate-errors")  # ✅ ignore SSL issues
        chrome_options.add_argument("--ignore-ssl-errors=yes")      # ✅ extra SSL ignore
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # avoid detection
        chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
        )

        service = Service(driver_path)
        self.driver = webdriver.Chrome(service=service, options=chrome_options)

    def scrape(self, urls):
        data = {}
        for i, url in enumerate(urls, 1):
            try:
                print(f"[INFO] Scraping page {i}/{len(urls)}: {url}")
                self.driver.get(url)

                # wait for page load (max 10 seconds)
                WebDriverWait(self.driver, 10).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete'
                )

                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                title = soup.title.string.strip() if soup.title else 'No Title'
                paragraphs = soup.find_all('p')
                content = ' '.join(p.get_text(strip=True) for p in paragraphs) or 'No Content'

                data[f'page {i}'] = {
                    'url': url,
                    'title': title,
                    'content': content
                }

            except TimeoutException:
                print(f"[⏳] Timeout loading: {url}")
                data[f'page {i}'] = {'url': url, 'title': 'Timeout', 'content': ''}
            except WebDriverException as e:
                print(f"[✗] WebDriver Error for {url}: {e}")
                data[f'page {i}'] = {'url': url, 'title': 'WebDriver Error', 'content': ''}
            except Exception as e:
                print(f"[✗] Failed: {url} — {e}")
                data[f'page {i}'] = {'url': url, 'title': 'Error', 'content': ''}

        self.driver.quit()
        return data
