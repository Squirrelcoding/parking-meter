from bs4 import BeautifulSoup
import re
import requests

with open("data/index.html", "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "html.parser")

pattern = re.compile(r"^https://www\.carsized\.com/en/cars/")

car_links = []
for a_tag in soup.find_all("a", href=True):
    href = a_tag['href']
    if pattern.match(href):
        car_links.append(href)

size_pattern = re.compile(
    r'<h2>.*?Size summary.*?\|\s*<b>(.*?)</b>.*?is\s*([\d.]+)\s*m\s*<b>long</b>\s*and\s*([\d.]+)\s*m\s*<b>high</b>\.?.*?(\d+\.?\d*)\s*cm ground clearance.*?(\d+\.?\d*)\s*l cargo',
    re.IGNORECASE
)

dimensions = []

for i, url in enumerate(car_links):
    print(f"Processing: {url} | {i}/{len(car_links)}")
    response = requests.get(url)
    lines = response.text.splitlines()

    width_values = []

    for i, line in enumerate(lines):
        if '>Width<' in line:
            # Look at 10 lines before and after
            nearby_lines = lines[max(0, i-15): i]
            joined_text = " ".join(nearby_lines)

            # Find all numbers before 'cm'
            matches = re.findall(r'([\d.]+)\s*cm', joined_text)
            width_values.extend(matches)
            break  # Stop after first match to avoid duplicates

    if width_values:
        print(f"Width values near '>Width<': {width_values}")
        dimensions.append((float(width_values[0]), float(width_values[1])))
    else:
        print("Could not find width values.")
        
print(dimensions)