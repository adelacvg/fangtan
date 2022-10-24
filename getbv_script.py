from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import pickle
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
service = Service(executable_path=ChromeDriverManager().install())

op = webdriver.ChromeOptions()
op.add_argument('headless')
# op.add_argument('disable-gpu')
op.add_argument("--log-level=3")
driver = webdriver.Chrome(options=op,service=service)

def get_by_uid(uid):
    space = 'https://space.bilibili.com/{}/video'.format(uid)
    driver.get(space)
    time.sleep(1)
    page_nums = driver.find_element(by=By.XPATH,value='//*[@id="submit-video-list"]/ul[3]/span[1]')
    time.sleep(1)
    pagen = int(page_nums.text[2:-3])
    l=[]
    for page in tqdm(range(1,pagen+1)):
        driver.get(space+'?page='+str(page))
        time.sleep(1)
        xp = "//*[@id='submit-video-list']/ul[2]/*"
        vid_list = driver.find_elements(by=By.XPATH, value=xp)
        for vid in vid_list:
            bv = vid.get_attribute("data-aid")
            l.append(bv)
    return l
bv_list = []
for uid in ['496688267','1266160043']:
    bv_list=bv_list+get_by_uid(uid)
print(len(bv_list))
with open('bv_file', 'wb') as fp:
    pickle.dump(bv_list,fp)

with open ('bv_file', 'rb') as fp:
    itemlist = pickle.load(fp)
    print(itemlist)
