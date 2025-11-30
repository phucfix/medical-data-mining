"""
Base Crawler class cho việc thu thập dữ liệu y tế
"""
import requests
from bs4 import BeautifulSoup
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Generator
from abc import ABC, abstractmethod
from loguru import logger
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    USER_AGENT, CRAWLER_DELAY, CRAWLER_TIMEOUT, MAX_RETRIES, RAW_DATA_DIR
)


class BaseCrawler(ABC):
    """Base class cho tất cả các crawler"""
    
    def __init__(self, source_name: str, base_url: str, delay: float = CRAWLER_DELAY):
        self.source_name = source_name
        self.base_url = base_url
        self.delay = delay
        self.session = self._create_session()
        self.collected_data = []
        
        # Setup logging
        logger.add(
            f"logs/{source_name}_crawler.log",
            rotation="10 MB",
            level="INFO"
        )
    
    def _create_session(self) -> requests.Session:
        """Tạo session với headers phù hợp"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
        })
        return session
    
    def _get_page(self, url: str, retry_count: int = 0) -> Optional[BeautifulSoup]:
        """Lấy nội dung trang web với retry logic"""
        try:
            # Random delay để tránh bị block
            time.sleep(self.delay + random.uniform(0, 0.5))
            
            response = self.session.get(url, timeout=CRAWLER_TIMEOUT)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'lxml')
            
        except requests.RequestException as e:
            if retry_count < MAX_RETRIES:
                logger.warning(f"Retry {retry_count + 1}/{MAX_RETRIES} for {url}: {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self._get_page(url, retry_count + 1)
            else:
                logger.error(f"Failed to fetch {url}: {e}")
                return None
    
    @abstractmethod
    def get_category_urls(self, category: str) -> List[str]:
        """Lấy danh sách URLs của một category (bệnh/triệu chứng/thuốc)"""
        pass
    
    @abstractmethod
    def parse_item(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Parse một item từ trang chi tiết"""
        pass
    
    def crawl_category(self, category: str) -> List[Dict]:
        """Crawl tất cả items trong một category"""
        logger.info(f"Starting crawl for category: {category}")
        
        urls = self.get_category_urls(category)
        logger.info(f"Found {len(urls)} URLs to crawl")
        
        results = []
        for url in tqdm(urls, desc=f"Crawling {category}"):
            soup = self._get_page(url)
            if soup:
                item = self.parse_item(url, soup)
                if item:
                    item['source'] = self.source_name
                    item['source_url'] = url
                    item['category'] = category
                    results.append(item)
        
        logger.info(f"Crawled {len(results)} items for {category}")
        return results
    
    def save_data(self, data: List[Dict], category: str):
        """Lưu dữ liệu đã crawl"""
        output_dir = RAW_DATA_DIR / category
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{self.source_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {output_file}")
    
    def run(self, categories: List[str] = None):
        """Chạy crawler cho tất cả categories"""
        if categories is None:
            categories = ['diseases', 'symptoms', 'drugs']
        
        for category in categories:
            try:
                data = self.crawl_category(category)
                if data:
                    self.save_data(data, category)
            except Exception as e:
                logger.error(f"Error crawling {category}: {e}")


class VinmecCrawler(BaseCrawler):
    """Crawler cho Vinmec.com"""
    
    def __init__(self):
        super().__init__(
            source_name="vinmec",
            base_url="https://www.vinmec.com"
        )
        
        self.category_paths = {
            'diseases': '/vi/benh/',
            'symptoms': '/vi/tin-tuc/thong-tin-suc-khoe/',
        }
    
    def get_category_urls(self, category: str) -> List[str]:
        """Lấy URLs từ sitemap hoặc listing pages"""
        urls = []
        
        if category not in self.category_paths:
            return urls
        
        # Crawl các trang listing
        base_path = self.category_paths[category]
        
        # Lấy danh sách các bệnh từ trang danh mục
        for page in range(1, 50):  # Crawl 50 trang đầu
            list_url = f"{self.base_url}{base_path}?page={page}"
            soup = self._get_page(list_url)
            
            if not soup:
                break
            
            # Tìm các links đến trang chi tiết
            links = soup.find_all('a', href=True)
            page_urls = []
            
            for link in links:
                href = link['href']
                if base_path in href and href != base_path:
                    full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                    if full_url not in urls:
                        page_urls.append(full_url)
            
            if not page_urls:
                break
                
            urls.extend(page_urls)
            logger.info(f"Page {page}: Found {len(page_urls)} URLs")
        
        return list(set(urls))
    
    def parse_item(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Parse thông tin bệnh/triệu chứng từ Vinmec"""
        try:
            # Lấy tiêu đề
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else None
            
            if not title:
                return None
            
            # Lấy nội dung chính
            content_div = soup.find('div', class_='article-content') or soup.find('article')
            content = ""
            
            if content_div:
                # Lấy tất cả các đoạn văn
                paragraphs = content_div.find_all(['p', 'li', 'h2', 'h3'])
                content = "\n".join([p.get_text(strip=True) for p in paragraphs])
            
            # Trích xuất các phần cụ thể
            sections = self._extract_sections(soup)
            
            return {
                'name': title,
                'content': content,
                'description': sections.get('description', ''),
                'symptoms': sections.get('symptoms', []),
                'causes': sections.get('causes', []),
                'treatment': sections.get('treatment', ''),
                'prevention': sections.get('prevention', ''),
            }
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None
    
    def _extract_sections(self, soup: BeautifulSoup) -> Dict:
        """Trích xuất các phần từ bài viết"""
        sections = {}
        
        # Tìm các heading và nội dung tương ứng
        headings = soup.find_all(['h2', 'h3'])
        
        for heading in headings:
            heading_text = heading.get_text(strip=True).lower()
            
            # Lấy nội dung sau heading
            content_parts = []
            for sibling in heading.find_next_siblings():
                if sibling.name in ['h2', 'h3']:
                    break
                if sibling.name in ['p', 'ul', 'ol']:
                    content_parts.append(sibling.get_text(strip=True))
            
            content = "\n".join(content_parts)
            
            # Phân loại theo heading
            if 'triệu chứng' in heading_text or 'dấu hiệu' in heading_text:
                sections['symptoms'] = content.split('\n')
            elif 'nguyên nhân' in heading_text:
                sections['causes'] = content.split('\n')
            elif 'điều trị' in heading_text or 'chữa' in heading_text:
                sections['treatment'] = content
            elif 'phòng ngừa' in heading_text or 'phòng tránh' in heading_text:
                sections['prevention'] = content
            elif 'là gì' in heading_text or 'định nghĩa' in heading_text:
                sections['description'] = content
        
        return sections


class WikipediaCrawler(BaseCrawler):
    """Crawler cho Wikipedia tiếng Việt - Danh mục y tế"""
    
    def __init__(self):
        super().__init__(
            source_name="wikipedia",
            base_url="https://vi.wikipedia.org"
        )
        
        self.category_urls = {
            'diseases': [
                '/wiki/Thể_loại:Bệnh',
                '/wiki/Thể_loại:Bệnh_truyền_nhiễm',
                '/wiki/Thể_loại:Bệnh_tim_mạch',
                '/wiki/Thể_loại:Bệnh_ung_thư',
                '/wiki/Thể_loại:Bệnh_hô_hấp',
                '/wiki/Thể_loại:Bệnh_tiêu_hóa',
                '/wiki/Thể_loại:Bệnh_thần_kinh',
            ],
            'drugs': [
                '/wiki/Thể_loại:Thuốc',
                '/wiki/Thể_loại:Thuốc_kháng_sinh',
                '/wiki/Thể_loại:Thuốc_giảm_đau',
            ],
            'symptoms': [
                '/wiki/Thể_loại:Triệu_chứng_y_học',
            ]
        }
    
    def get_category_urls(self, category: str) -> List[str]:
        """Lấy URLs từ các category của Wikipedia"""
        urls = []
        
        if category not in self.category_urls:
            return urls
        
        for cat_url in self.category_urls[category]:
            full_url = f"{self.base_url}{cat_url}"
            self._crawl_category_page(full_url, urls, depth=2)
        
        return list(set(urls))
    
    def _crawl_category_page(self, url: str, urls: List[str], depth: int = 2):
        """Crawl recursive các subcategories"""
        if depth <= 0:
            return
        
        soup = self._get_page(url)
        if not soup:
            return
        
        # Tìm các articles trong category
        content_div = soup.find('div', id='mw-pages')
        if content_div:
            links = content_div.find_all('a', href=True)
            for link in links:
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = f"{self.base_url}{href}"
                    urls.append(full_url)
        
        # Crawl subcategories
        subcat_div = soup.find('div', id='mw-subcategories')
        if subcat_div and depth > 1:
            subcat_links = subcat_div.find_all('a', href=True)
            for link in subcat_links[:5]:  # Giới hạn số subcategories
                href = link['href']
                if 'Thể_loại:' in href:
                    subcat_url = f"{self.base_url}{href}"
                    self._crawl_category_page(subcat_url, urls, depth - 1)
    
    def parse_item(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Parse bài viết Wikipedia"""
        try:
            # Lấy tiêu đề
            title_tag = soup.find('h1', id='firstHeading')
            title = title_tag.get_text(strip=True) if title_tag else None
            
            if not title:
                return None
            
            # Lấy nội dung
            content_div = soup.find('div', id='mw-content-text')
            if not content_div:
                return None
            
            # Lấy đoạn mở đầu
            intro_paragraphs = []
            for p in content_div.find_all('p', recursive=False)[:3]:
                text = p.get_text(strip=True)
                if text:
                    intro_paragraphs.append(text)
            
            description = " ".join(intro_paragraphs)
            
            # Lấy toàn bộ nội dung
            all_text = []
            for element in content_div.find_all(['p', 'li']):
                text = element.get_text(strip=True)
                if text and len(text) > 20:
                    all_text.append(text)
            
            content = "\n".join(all_text)
            
            # Trích xuất infobox nếu có
            infobox = self._extract_infobox(soup)
            
            return {
                'name': title,
                'description': description,
                'content': content,
                'infobox': infobox,
            }
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None
    
    def _extract_infobox(self, soup: BeautifulSoup) -> Dict:
        """Trích xuất thông tin từ infobox"""
        infobox = {}
        
        infobox_table = soup.find('table', class_='infobox')
        if infobox_table:
            rows = infobox_table.find_all('tr')
            for row in rows:
                header = row.find('th')
                data = row.find('td')
                if header and data:
                    key = header.get_text(strip=True)
                    value = data.get_text(strip=True)
                    infobox[key] = value
        
        return infobox


class HelloBacSiCrawler(BaseCrawler):
    """Crawler cho HelloBacsi.com"""
    
    def __init__(self):
        super().__init__(
            source_name="hellobacsi",
            base_url="https://hellobacsi.com"
        )
        
        self.category_paths = {
            'diseases': '/benh/',
            'symptoms': '/trieu-chung/',
            'drugs': '/thuoc/',
        }
    
    def get_category_urls(self, category: str) -> List[str]:
        """Lấy URLs từ HelloBacsi"""
        urls = []
        
        if category not in self.category_paths:
            return urls
        
        base_path = self.category_paths[category]
        
        # Crawl sitemap hoặc listing pages
        for page in range(1, 100):
            list_url = f"{self.base_url}{base_path}page/{page}/"
            soup = self._get_page(list_url)
            
            if not soup:
                break
            
            # Tìm các article links
            articles = soup.find_all('article') or soup.find_all('div', class_='post')
            
            if not articles:
                break
            
            for article in articles:
                link = article.find('a', href=True)
                if link:
                    href = link['href']
                    if base_path in href:
                        urls.append(href)
            
            logger.info(f"Page {page}: Found {len(articles)} articles")
        
        return list(set(urls))
    
    def parse_item(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Parse bài viết từ HelloBacsi"""
        try:
            # Lấy tiêu đề
            title_tag = soup.find('h1')
            title = title_tag.get_text(strip=True) if title_tag else None
            
            if not title:
                return None
            
            # Lấy nội dung chính
            content_div = soup.find('div', class_='entry-content') or soup.find('article')
            
            if not content_div:
                return None
            
            # Lấy tất cả paragraphs
            paragraphs = content_div.find_all(['p', 'li', 'h2', 'h3'])
            content = "\n".join([p.get_text(strip=True) for p in paragraphs])
            
            # Lấy mô tả ngắn
            meta_desc = soup.find('meta', {'name': 'description'})
            description = meta_desc['content'] if meta_desc else ""
            
            return {
                'name': title,
                'description': description,
                'content': content,
            }
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None


class DrugBankVNCrawler(BaseCrawler):
    """Crawler cho DrugBank.vn - Thông tin thuốc"""
    
    def __init__(self):
        super().__init__(
            source_name="drugbank_vn",
            base_url="https://drugbank.vn"
        )
    
    def get_category_urls(self, category: str) -> List[str]:
        """Lấy danh sách URLs thuốc"""
        if category != 'drugs':
            return []
        
        urls = []
        
        # Crawl theo alphabet
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        for letter in alphabet:
            for page in range(1, 20):
                list_url = f"{self.base_url}/thuoc?search={letter}&page={page}"
                soup = self._get_page(list_url)
                
                if not soup:
                    break
                
                # Tìm links đến trang chi tiết thuốc
                links = soup.find_all('a', href=True)
                page_urls = []
                
                for link in links:
                    href = link['href']
                    if '/thuoc/' in href and href != '/thuoc/':
                        full_url = href if href.startswith('http') else f"{self.base_url}{href}"
                        if full_url not in urls:
                            page_urls.append(full_url)
                
                if not page_urls:
                    break
                
                urls.extend(page_urls)
            
            logger.info(f"Letter {letter}: Total {len(urls)} URLs")
        
        return list(set(urls))
    
    def parse_item(self, url: str, soup: BeautifulSoup) -> Optional[Dict]:
        """Parse thông tin thuốc"""
        try:
            # Lấy tên thuốc
            title_tag = soup.find('h1')
            name = title_tag.get_text(strip=True) if title_tag else None
            
            if not name:
                return None
            
            # Lấy các thông tin chi tiết
            info = {
                'name': name,
                'active_ingredient': '',
                'dosage_form': '',
                'manufacturer': '',
                'indication': '',
                'contraindication': '',
                'side_effects': '',
                'dosage': '',
                'content': '',
            }
            
            # Tìm các section thông tin
            sections = soup.find_all(['div', 'section'], class_=True)
            
            for section in sections:
                section_text = section.get_text(strip=True).lower()
                
                if 'hoạt chất' in section_text or 'thành phần' in section_text:
                    info['active_ingredient'] = section.get_text(strip=True)
                elif 'chỉ định' in section_text:
                    info['indication'] = section.get_text(strip=True)
                elif 'chống chỉ định' in section_text:
                    info['contraindication'] = section.get_text(strip=True)
                elif 'tác dụng phụ' in section_text:
                    info['side_effects'] = section.get_text(strip=True)
                elif 'liều dùng' in section_text:
                    info['dosage'] = section.get_text(strip=True)
            
            # Lấy toàn bộ nội dung
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
            if main_content:
                info['content'] = main_content.get_text(strip=True)
            
            return info
            
        except Exception as e:
            logger.error(f"Error parsing {url}: {e}")
            return None


def crawl_all_sources():
    """Chạy tất cả các crawler"""
    crawlers = [
        VinmecCrawler(),
        WikipediaCrawler(),
        HelloBacSiCrawler(),
        DrugBankVNCrawler(),
    ]
    
    for crawler in crawlers:
        logger.info(f"Starting crawler: {crawler.source_name}")
        try:
            crawler.run()
        except Exception as e:
            logger.error(f"Error in {crawler.source_name}: {e}")
        
        logger.info(f"Finished crawler: {crawler.source_name}")


if __name__ == "__main__":
    crawl_all_sources()
