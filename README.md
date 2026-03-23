# playlist_recommender
int4203 group 9 project

# Installation
1. Clone the repository:
git clone https://github.com/juicemin/playlist_recommender.git

2. Install the required libraries:
pip install -r requirements.txt

Required packages include pandas, scikit-learn, ipywidgets, ipykernel and voila.

# Run Project
1. Option A: Web Application (Recommended)
Run this command in your terminal to launch the clean, interactive interface:
voila main_recommender.ipynb

2. Option B: Technical Review (Jupyter Notebook)
Open main_recommender.ipynb and select Run All. You may uncomment the print() statements within the code cells if you wish to verify that the environment setup and data preparation steps are executing correctly.

Note: Ensure the assets/ folder and clean_music_dataset.csv remain in the same directory as the notebook.

# Project Structure
-main_recommender.ipynb: The core Python code and user interface.
-clean_music_dataset.csv: Preprocessed data used for recommendations.
-assets/: MP3 preview files for the different study modes.
-requirements.txt: List of necessary Python dependencies.

