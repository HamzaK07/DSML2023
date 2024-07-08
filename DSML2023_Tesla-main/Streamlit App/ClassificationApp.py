import streamlit as st
from transformers import FlaubertForSequenceClassification, FlaubertTokenizer
import torch
import joblib
import pandas as pd
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Titre de l'application
st.title('French sentence classification in CEFR levels')

# Initialisation de l'historique
if 'historique_predictions' not in st.session_state:
    st.session_state['historique_predictions'] = []

# Chargement des modèles
modele_lr = joblib.load("./LogisticRegressionModel/LRModel.pkl")
vectorisor_lr = joblib.load('./LogisticRegressionModel/LRVectorisor.pkl')

modele_rf = joblib.load("./RandomForestModel/RFModel.pkl")
vectorisor_rf = joblib.load("./RandomForestModel/RFvectorisor.pkl")

# Chargement du modèle et du tokenizer
# Remplacez 'chemin_vers_le_modele' par l'URL ou le chemin du dossier contenant votre modèle
model_path = "./FlaubertModel"

flaubert_model = FlaubertForSequenceClassification.from_pretrained(model_path)
flaubert_tokenizer = FlaubertTokenizer.from_pretrained(model_path)

# Dictionnaire de mappage indice -> label CEFR
cefr_labels = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

# Sélection du modèle
model_choice = st.radio("Select a classification model :", ('FlauBERT', 'Logistic Regression', 'Random Forest'))

# Fonction pour afficher l'historique des prédictions
def afficher_historique_predictions():
    if st.session_state['historique_predictions']:
        df = pd.DataFrame(st.session_state['historique_predictions'])
        st.write("Predictions history :")
        st.dataframe(df)
    else:
        st.write("No predictions.")

# Fonction de prédiction pour chaque modèle
def classify_text(text, model_choice):
    if model_choice == 'FlauBERT':
        inputs = flaubert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = flaubert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction_index = torch.argmax(probs).item()
        # Mettre à jour l'historique des prédictions
        st.session_state['historique_predictions'].append({
            'Sentence': user_input,
            'Model used': model_choice,
            'Prediction': cefr_labels[prediction_index]})
        return cefr_labels[prediction_index]
    elif model_choice == 'Logistic Regression':
        texte_transforme = vectorisor_lr.transform([text])

        # Faire une prédiction avec le modèle de régression logistique
        prediction_index = modele_lr.predict(texte_transforme)[0]
        # Mettre à jour l'historique des prédictions
        st.session_state['historique_predictions'].append({
            'Sentence': user_input,
            'Model used': model_choice,
            'Prediction': prediction_index})
        return prediction_index
    elif model_choice == 'Random Forest':
        texte_transforme = vectorisor_rf.transform([text])  # Assurez-vous d'utiliser le bon vectoriseur

        # Faire une prédiction avec le modèle de forêt aléatoire
        prediction_index = modele_rf.predict(texte_transforme)[0]
        # Mettre à jour l'historique des prédictions
        st.session_state['historique_predictions'].append({
            'Sentence': user_input,
            'Model used': model_choice,
            'Prediction': prediction_index})
        return prediction_index


# Interface utilisateur pour la saisie de texte
user_input = st.text_area("Enter a sentence in French :", "", key="unique_key_for_user_input")

if st.button('Classify sentence'):
    if user_input:
        classification_label = classify_text(user_input, model_choice)
        st.write(f"The sentence is classified in CEFR levels : {classification_label}")
        afficher_historique_predictions()  # Afficher l'historique après chaque prédiction
    else:
        st.write("Enter a sentence in French.")

# Informations sur le modèle
if st.button("Information about the model"):
    if model_choice == 'FlauBERT':
        st.write("FlauBERT is a pre-trained model that excels in capturing the nuances of the French language, making it exceptionally well-suited for our specific task. Compared to CamemBERT, another popular French language model, FlauBERT demonstrated superior performance in our applications, offering better accuracy. By retraining FlauBERT and fine-tuning its hyperparameters, we were able to optimize the model, ultimately achieving an accuracy of 0.62. This substantial improvement propelled us to the second position on the leaderboard, highlighting the effectiveness of FlauBERT in complex language processing tasks.")

    elif model_choice == 'Logistic Regression':
        st.write("Our initial attempt involved implementing logistic regression, which, while a straightforward and commonly used method for classification tasks, proved to be less effective for this particular challenge. It allowed us to achieve accuracies ranging between 0.4 and 0.43. Logistic regression, primarily due to its linear nature, often struggles with complex and nuanced data patterns that are typical in advanced language processing tasks. Additionally, our efforts to enhance the model's performance through feature augmentation were unsuccessful, indicating that a more sophisticated model might be necessary to effectively capture and interpret the intricacies of the data.")

    elif model_choice == 'Random Forest':
        st.write("In a subsequent approach, we utilized a Random Forest classifier. This method, known for its capability to handle complex and non-linear data relationships, initially yielded an accuracy of 0.37. Recognizing the potential for improvement, we embarked on feature augmentation, which proved to be fruitful in this case. By enriching our dataset with additional features, we were able to enhance the model's understanding of the data's complexities. This strategic enhancement led to a notable increase in performance, boosting our accuracy from 0.37 to 0.43. The success of feature augmentation in this instance underscored the adaptability and robustness of the Random Forest approach for our language processing task.")


# Function to fetch video subtitles
def fetch_video_subtitles(url):
    pattern = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    match = re.match(pattern, url)
    video_id = match.group(6)

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        french_transcript = transcripts.find_transcript(['fr', 'fr-FR'])
        subtitle_items = french_transcript.fetch()

        subtitle_words = []
        for item in subtitle_items:
            subtitle_words.extend(item['text'].split())
            if len(subtitle_words) >= 1000:
                break

        return ' '.join(subtitle_words[:1000])
    except Exception as e:
        return f"Subtitle error or not available: {e}"

# Function to analyze text with FlauBERT
def analyze_text_with_flaubert(text):
    tokenized_input = flaubert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        model_output = flaubert_model(**tokenized_input)

    probabilities = torch.nn.functional.softmax(model_output.logits, dim=-1)
    highest_prob_index = torch.argmax(probabilities).item()

    return cefr_labels[highest_prob_index]

# Streamlit UI
st.title("FlauBERT YouTube Subtitle CEFR Classifier")

video_url_input = st.text_input("YouTube Video URL:", key="video_url_input")

if st.button('Analyze Video'):
    if video_url_input:
        video_subtitles = fetch_video_subtitles(video_url_input)
        if video_subtitles and not video_subtitles.startswith("Subtitle error"):
            cefer_result = analyze_text_with_flaubert(video_subtitles)
            st.write("CEFR Level:", cefer_result)
            st.text_area("Subtitles:", video_subtitles, height=150)
        else:
            st.write(video_subtitles)
    else:
        st.write("Enter a valid YouTube video URL.")



# Exécutez cette application avec `streamlit run votre_fichier.py`