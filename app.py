import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, jsonify, render_template
import whisper
from pydub import AudioSegment
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the spaCy model
nlp = spacy.load("en_core_web_trf")

app = Flask(__name__)

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
modelgen= T5ForConditionalGeneration.from_pretrained("t5-small")

try:
    model = whisper.load_model("small")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")

def summarize_text(text):
    """
    Summarizes the given text using T5.
    """
    input_ids = tokenizer("summarize: " + text, return_tensors="pt", truncation=True).input_ids
    # Check if the input text is too short to summarize
    if len(text.split()) < 15:  # For example, consider texts with fewer than 10 words as too short to summarize
        return text
    else:
        # If the input text is long enough, summarize it
        outputs = modelgen.generate(input_ids, max_length=100, min_length=30, length_penalty=2.0, num_beams=4,
                                    early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_text(prompt):
    # Load GPT-2 model
    generator = pipeline('text-generation', model='gpt2')

    # Generate text
    output = generator(prompt, max_length=100,         # Increase the max length further
    num_return_sequences=1,
    temperature=0.1,        # Control randomness
    top_k=50,               # Control diversity of text generation
    top_p=0.95,
    no_repeat_ngram_size=2, # Avoid repetition
    pad_token_id=50256,     # Ensure padding token is defined
    eos_token_id=50256 ,     # Explicitly set end-of-sequence token
    truncation=True)


    return output[0]['generated_text']



def preprocess_text(text):
    """
    Preprocess the given text using NLTK.
    - Tokenization
    - Stopword Removal
    - Lemmatization
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Return the preprocessed text as a string
    return " ".join(lemmatized_tokens)

@app.route("/")
def serve_index():
    return render_template("test.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    print("Received audio file: ", audio_file.filename)

    # Save the uploaded audio file
    audio_path = "uploaded_audio"
    audio_file.save(audio_path)

    # Convert to WAV if not already in WAV format
    try:
        audio = AudioSegment.from_file(audio_path)
        audio.export("uploaded_audio.wav", format="wav")
    except Exception as e:
        print(f"Error converting audio: {e}")
        return jsonify({"error": "Error processing audio file"}), 500

    try:
        # Transcribe audio using Whisper model
        result = model.transcribe("uploaded_audio.wav")
        transcription = result['text']

        # Preprocess the transcription using NLTK
        preprocessed_text = preprocess_text(transcription)

        # Perform NER on the preprocessed text
        doc = nlp(transcription)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        print("Detected entities:", entities)

        # Sentiment Analysis
        sentiment = TextBlob(transcription).sentiment
        polarity = sentiment.polarity  # Retain the raw polarity value
        if polarity > 0:
            sentiment_label = "Positive"
            positive_score = round(polarity * 100, 2)
            negative_score = 100 - positive_score
        else:
            sentiment_label = "Negative"
            positive_score = 100- round(abs(polarity) * 100, 2)
            negative_score = round(abs(polarity) * 100, 2)

        sentiment_score = {
            "label": sentiment_label,
            "positive_percentage": positive_score,
            "negative_percentage": negative_score
        }

        # Perform Summarization and Text Generation
        summary = summarize_text(transcription)
        generated_text = generate_text(transcription)

        return jsonify({
            "transcription": transcription,
            "preprocessed_text": preprocessed_text,
            "named_entities": entities,
            "sentiment": sentiment_score,
            "summary": summary,
            "generated_text": generated_text
        })
    except Exception as e:
        print(f"Error during transcription: {e}")
        return jsonify({"error": "Error in transcription"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=3000)
