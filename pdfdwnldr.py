import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# Define the URL of the FBR page with the PDFs
url = "https://www.fbr.gov.pk/act-rules-ordinances/131226"

# Specify the folder where you want to save the downloaded PDFs
folder_location = 'pdfs'

# Specify the log file where downloaded file names will be stored
log_file = 'downloaded_files.txt'

if not os.path.exists(folder_location):
    os.mkdir(folder_location)

# Read the log file to get the list of already downloaded files
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        downloaded_files = set(line.strip() for line in f)
else:
    downloaded_files = set()

# Function to download files with retries and in chunks
def download_file(url, dest, retries=3, chunk_size=1024, subpage=False):
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            with open(log_file, 'a') as log:
                log.write(dest + '\n')
            downloaded_files.add(dest)
            if subpage:
                print(f"Downloaded from subpage: {dest}")
            else:
                print(f"Downloaded: {dest}")
            return
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 == retries:
                print(f"Failed to download {url} after {retries} attempts.")

# Get the webpage content
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Specify the class names you want to target
target_classes = ['col-xs-12 col-md-4 col-sm-6']  # Replace with your actual class names

# Find all divs with the specified class names
for class_name in target_classes:
    divs = soup.find_all('div', {'class': class_name})
    for div in divs:
        # Find links within the div
        links = div.find_all('a', href=True)
        for link in links:
            # Process each link
            link_url = urljoin(url, link['href'])
            if link_url.lower().endswith('.pdf'):
                # Download the PDF file
                filename = os.path.join(folder_location, link['href'].split('/')[-1])
                if filename not in downloaded_files:
                    download_file(link_url, filename)
            #else:
                # Explore subpages (if needed)
                # Commenting out the subpage exploration
                # subpage_response = requests.get(link_url)
                # subpage_soup = BeautifulSoup(subpage_response.text, "html.parser")
                # subpage_links = subpage_soup.find_all('a', href=True)
                # for subpage_link in subpage_links:
                #     subpage_link_url = urljoin(link_url, subpage_link['href'])
                #     if subpage_link_url.lower().endswith('.pdf'):
                #         # Download the PDF file from the subpage
                #         filename = os.path.join(folder_location, subpage_link['href'].split('/')[-1])
                #         if filename not in downloaded_files:
                #             download_file(subpage_link_url, filename, subpage=True)