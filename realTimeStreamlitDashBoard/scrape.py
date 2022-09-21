#https://www.scraperapi.com/blog/how-to-scrape-stock-market-data-with-python/
import requests
from bs4 import BeautifulSoup
###
apiKey='ccfmlaqad3i2p1r03dh0'
###
urls = [
    'https://vn.investing.com/indices/vn',
    'https://www.investing.com/equities/nike',
    'https://www.investing.com/equities/coca-cola-co',
    'https://www.investing.com/equities/microsoft-corp',
]
for url in urls:
   page = requests.get(url)
   
   soup = BeautifulSoup(page.text, 'html.parser')
   company = soup.find('h1', {'class': 'text-2xl font-semibold instrument-header_title__GTWDv mobile:mb-2'}).text
   price = soup.find('div', {'class': 'instrument-price_instrument-price__3uw25 flex items-end flex-wrap font-bold'}).find_all('span')[0].text
   change = soup.find('div', {'class': 'instrument-price_instrument-price__3uw25 flex items-end flex-wrap font-bold'}).find_all('span')[2].text
  
   print('Loading: ', url)
   print(company, price, change)