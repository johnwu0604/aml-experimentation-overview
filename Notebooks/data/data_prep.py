import os
import requests
import argparse
import cv2
from time_util import time_limit

# +
output_dir = '../images'
num_images = 100
image_dim = 200

train_dir = os.path.join(output_dir, 'train')
valid_dir = os.path.join(output_dir, 'valid')
test_dir = os.path.join(output_dir, 'test')

# Make train, valid, test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)
# -

# Set search headers and URL
headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

# Define API endpoints
subscription_key = os.environ['COGNITIVE_SERVICES_API_KEY']
search_url = 'https://eastus.api.cognitive.microsoft.com/bing/v7.0/images/search'

# Define classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Make query for each class and download images
for name in classes:

    train_class_dir = os.path.join(train_dir, name)
    valid_class_dir = os.path.join(valid_dir, name)
    test_class_dir = os.path.join(test_dir, name)
    
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    
    if not os.path.exists(valid_class_dir):
        os.makedirs(valid_class_dir)
        
    if not os.path.exists(test_class_dir):
        os.makedirs(test_class_dir)
        
    counter = 0
    num_searches = int(num_images/150)+1

    for i in range(num_searches):
        
        response = requests.get(
            search_url, 
            headers = {
                'Ocp-Apim-Subscription-Key' : subscription_key
            }, 
            params = {
                'q': name, 
                'imageType': 'photo',
                'count': 150,
                'offset': i*150
            })
        response.raise_for_status()
        results = response.json()["value"]

        for image in results:
            if counter > num_images:
                break
            if image['encodingFormat'] == 'jpeg':
                
                print('Writing image {} for {}...'.format(counter, name))
                
                if counter < num_images * 0.7:
                    filename = '{}/{}.jpg'.format(train_class_dir, counter)
                elif counter < num_images * 0.9:
                    filename = '{}/{}.jpg'.format(valid_class_dir, counter)
                else:
                    filename = '{}/{}.jpg'.format(test_class_dir, counter)
                    
                try:
                    with time_limit(5):
                        with open(filename, 'wb') as file:
                            download = requests.get(image['contentUrl'], headers=headers)
                            file.write(download.content)
                            image = cv2.imread(filename)
                            image = cv2.resize(image, (image_dim, image_dim))
                            cv2.imwrite(filename, image)
                        counter += 1
                except:
                    print('Skipping {} due to download error:'.format(filename))
