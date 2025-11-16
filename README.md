## 2024 Intel AI Global Impact Festival - Global Winner (https://www.intel.com/content/www/us/en/corporate/artificial-intelligence/winners2024.html](https://www.intel.com/content/www/us/en/corporate/artificial-intelligence/winners2024.html#:~:text=and%20mentorship%20opportunities.-,Chong%20Yao%20Ong,-%2C%20Zi%20Xuan%20Mok)
## 2023 STEM Innovation Showcase - Best in Automation & AI

# Code & logic by Ong Chong Yao
# fully functional as of 24/6/2024
# :D good luck!

- Python 3.11
- A webcam
- A Google Firebase account

# Set Up Project & Firebase
In your Firebase project, go to Project Settings > Service Accounts.\
Click "Generate new private key" to download a JSON file.\
Rename this file to 'firebase-credentials.json' and place it in the same folder as the Python script.

# Install Dependencies
pip install ultralytics firebase_admin

Install the Nvidia CUDA Toolkit to maximise performance,\
Uninstall the existing PyTorch libraries,\
Then install PyTorch for CUDA.

Common bug:
"cap = cv2.VideoCapture(0)"

Need to change the camera index to the correct one.

