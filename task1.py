from selenium import webdriver                              # control browser in python
from selenium.webdriver.common.by import By                 # how you want to find elements on the webpage
from selenium.webdriver.chrome.service import Service       # layer between selenium and chrome browser
from webdriver_manager.chrome import ChromeDriverManager    # Automatically downloads the correct ChromeDriver version for your browser
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd



driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

url = "https://www.cars24.com/buy-used-tata-cars-mumbai/"
driver.get(url)

time.sleep(3)

for _ in range(5):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 5);")
    time.sleep(2)

#is used to scroll to the bottom of the webpage,helpful in lazyLoading
# driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.335);")
# time.sleep(5)

# Get all car
cars = driver.find_elements(By.CLASS_NAME, 'styles_carCardWrapper__sXLIp')
print("Total cars found:", len(cars))


# driver.close()
print("Total cars found:", len(cars))
# print(cars)