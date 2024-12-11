import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import whisper
from pydub import AudioSegment
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware if frontend is hosted on a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models
nlp = spacy.load("en_core_web_trf")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
modelgen = T5ForConditionalGeneration.from_pretrained("t5-small")
model = whisper.load_model("small")


def summarize_text(text):
    """
    Summarizes the given text using T5.
    """
    input_ids = tokenizer("summarize: " + text, return_tensors="pt", truncation=True).input_ids
    if len(text.split()) < 15:
        return text
    else:
        outputs = modelgen.generate(input_ids, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_text(prompt):
    """
    Generates text using GPT-2.
    """
    generator = pipeline('text-generation', model='gpt2')
    output = generator(prompt, max_length=100, num_return_sequences=1, temperature=0.1, top_k=50, top_p=0.95, no_repeat_ngram_size=2, pad_token_id=50256, eos_token_id=50256, truncation=True)
    return output[0]['generated_text']


def preprocess_text(text):
    """
    Preprocess the given text using NLTK.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return " ".join(lemmatized_tokens)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Transcribes the given audio file, processes the text, and performs various NLP tasks.
    """
    try:
        # Save uploaded file temporarily
        audio_path = "uploaded_audio"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # Convert audio to WAV
        audio = AudioSegment.from_file(audio_path)
        audio.export("uploaded_audio.wav", format="wav")

        # Transcribe audio using Whisper
        result = model.transcribe("uploaded_audio.wav")
        transcription = result['text']

        # Preprocess transcription
        preprocessed_text = preprocess_text(transcription)

        # Perform Named Entity Recognition (NER)
        doc = nlp(transcription)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

        # Sentiment Analysis
        sentiment = TextBlob(transcription).sentiment
        polarity = sentiment.polarity
        if polarity > 0:
            sentiment_label = "Positive"
            positive_score = round(polarity * 100, 2)
            negative_score = 100 - positive_score
        else:
            sentiment_label = "Negative"
            positive_score = 100 - round(abs(polarity) * 100, 2)
            negative_score = round(abs(polarity) * 100, 2)

        sentiment_score = {
            "label": sentiment_label,
            "positive_percentage": positive_score,
            "negative_percentage": negative_score
        }

        # Perform summarization and text generation
        summary = summarize_text(transcription)
        generated_text = generate_text(transcription)

        # Return JSON response
        return JSONResponse(content={
            "transcription": transcription,
            "preprocessed_text": preprocessed_text,
            "named_entities": entities,
            "sentiment": sentiment_score,
            "summary": summary,
            "generated_text": generated_text
        })

    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
