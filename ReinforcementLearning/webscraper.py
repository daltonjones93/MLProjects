import requests
from bs4 import BeautifulSoup

def scrape_yahoo_finance_qqq():
    try:
        url = 'https://finance.yahoo.com/quote/QQQ'
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find elements containing mentions of QQQ (modify as needed)
            mentions = []
            for paragraph in soup.find_all('p'):
                if 'QQQ' in paragraph.get_text():
                    mentions.append(paragraph.get_text())

            return mentions

        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
qqq_mentions = scrape_yahoo_finance_qqq()

if qqq_mentions:
    print("QQQ Mentions on Yahoo Finance:")
    for mention in qqq_mentions:
        print(mention)
else:
    print("No mentions of QQQ found.")
