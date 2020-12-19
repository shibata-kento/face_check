from icrawler.builtin import BingImageCrawler
import sys
import os

save_path = os.path.join(os.path.dirname(__file__)) + '/data/gakky'
print(save_path)
crawler = BingImageCrawler(storage = {"root_dir" : save_path})
crawler.crawl(keyword = '新垣結衣', max_num = 1000)