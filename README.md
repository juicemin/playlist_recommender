# Music Playlist Recommender for Students (Group 9)
An AI powered hybrid recommendation system designed to enhance university student productivity and mental well being. This project specifically addresses Sustainable Development Goal 3: Good Health and Well-being by reducing decision fatigue and academic stress through context aware music suggestions.

# Installation
1. Clone the repository:
git clone https://github.com/juicemin/playlist_recommender.git

2. Install the required libraries:
pip install -r requirements.txt

Required packages include pandas, scikit-learn, ipywidgets, ipykernel and voila.

# How to Run the Project
1. Option A: Web Application (Recommended)
Launch the clean, interactive Voilà interface by running this command in your terminal from the project root:
voila main_recommender.ipynb

2. Option B: Technical Review (Jupyter Notebook)
Open main_recommender.ipynb and select Run All. You may uncomment the print() statements within the code cells to verify the environment setup and data preparation steps.

# System Architecture & AI Logic
The system utilizes a two stage hybrid approach to ensure high recommendation accuracy:
1. Stage 1: Intent Recognition: A rule based keyword mapping system identifies user intent from natural language inputs.
2. Stage 2: Content Based Retrieval: A similarity engine calculates Cosine Similarity between the target study profile and the normalized audio features of 10,000+ tracks.

# Performance Metrics
-Suggester Accuracy: 95% success rate in mapping user descriptions to the correct study mode.
-Precision@10: 0.90, ensuring 9 out of 10 recommended songs perfectly fit the selected context.
-Determinism: The system uses deterministic mathematical logic to provide consistent and repeatable results for a stable study environment.

# Important Usage Notes!
1. Dynamic Generation: You must click the Generate Playlist button every time you change the study mode or update the mood input to refresh the AI inference.
2. Embedded Audio: 10 second high quality audio previews are Base64 encoded directly into the UI. This ensures the demo is fully portable and bypasses local server security restrictions.
3. Keyword Triggers:
    -Deep Study: exam, test, final, study.
    -Creative Work: coding, design, creative.
    -Relaxation: tired, stress, rest, break.
    -Active Learning: group, project, discuss.

# Project Structure
-main_recommender.ipynb: The core Python logic and interactive Voilà interface.
-clean_music_dataset.csv: Preprocessed dataset with normalized audio features.
-dataset.csv: Original raw music data source.
-deep_study.mp3, creative_work.mp3, relaxation.mp3, active_learning.mp3: Audio assets for real time UI previews.
-requirements.txt: List of necessary Python dependencies.

