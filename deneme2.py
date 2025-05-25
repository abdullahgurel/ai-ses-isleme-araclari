import streamlit as st
import numpy as np
import torch
import torchaudio
import tempfile
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
from io import BytesIO

# Sayfa düzeni ayarları
st.set_page_config(page_title="Ses İşleme Araçları", layout="wide")

@st.cache_resource
def load_tts_model():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    # Örnek konuşmacı embeddingi için
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    return processor, model, vocoder, speaker_embeddings

@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    return processor, model

@st.cache_resource
def load_conformer_model():
    # Conformer yerine normal Wav2Vec2 modelini kullanacağız
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

def text_to_speech(text, language="tr"):
    processor, model, vocoder, speaker_embeddings = load_tts_model()
    
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Ses verisini bir dosyaya kaydet
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        sf.write(tmp_file.name, speech.numpy(), 16000)
        return tmp_file.name

def transcribe_audio(audio_file, language="tr"):
    processor, model = load_whisper_model()
    
    # Ses dosyasını yükle
    audio_array, sampling_rate = torchaudio.load(audio_file)
    
    # Gerekirse tek kanala dönüştür
    if audio_array.shape[0] > 1:
        audio_array = torch.mean(audio_array, dim=0, keepdim=True)
    
    # Gerekirse örnekleme hızını değiştir
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_array = resampler(audio_array)
        sampling_rate = 16000
    
    input_features = processor(audio_array.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_features
    
    # Dil belirtme (isteğe bağlı)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    
    # Çözümleme
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def translate_audio(audio_file, target_language="en"):
    processor, model = load_whisper_model()
    
    # Ses dosyasını yükle
    audio_array, sampling_rate = torchaudio.load(audio_file)
    
    # Gerekirse tek kanala dönüştür
    if audio_array.shape[0] > 1:
        audio_array = torch.mean(audio_array, dim=0, keepdim=True)
    
    # Gerekirse örnekleme hızını değiştir
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_array = resampler(audio_array)
        sampling_rate = 16000
    
    input_features = processor(audio_array.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_features
    
    # Çeviri görevi için decoder_ids ayarla
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_language, task="translate")
    
    # Çeviri
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return translation

def conformer_transcribe(audio_file):
    processor, model = load_conformer_model()
    
    # Ses dosyasını yükle
    audio_array, sampling_rate = torchaudio.load(audio_file)
    
    # Gerekirse tek kanala dönüştür
    if audio_array.shape[0] > 1:
        audio_array = torch.mean(audio_array, dim=0, keepdim=True)
    
    # Gerekirse örnekleme hızını değiştir
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio_array = resampler(audio_array)
        sampling_rate = 16000
    
    # Wav2Vec2 formatı için işlem
    input_values = processor(audio_array.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Ana uygulama
def main():
    st.title("Ses İşleme Araçları")
    
    tabs = st.tabs(["TTS ile Ses Sentezleme", "Whisper ile Transkripsiyon", "Whisper ile Tercüme", "Conformer ile Transkripsiyon"])
    
    # TTS ile Ses Sentezleme
    with tabs[0]:
        st.header("Metinden Sese Dönüştürme")
        
        text_input = st.text_area("Sese dönüştürülecek metni girin:", value="Merhaba, bu bir test metnidir.")
        language = st.selectbox("Dil seçin:", ["tr", "en", "fr", "de", "es"], index=0)
        
        if st.button("Sesi Oluştur", key="tts_button"):
            with st.spinner("Ses oluşturuluyor..."):
                try:
                    output_file = text_to_speech(text_input, language)
                    
                    # Ses dosyasını görüntüle
                    st.audio(output_file, format="audio/wav")
                    
                    # Temp dosyasını sil
                    if os.path.exists(output_file):
                        os.unlink(output_file)
                except Exception as e:
                    st.error(f"Ses oluşturma hatası: {str(e)}")
    
    # Whisper ile Transkripsiyon
    with tabs[1]:
        st.header("Ses Transkripsiyon")
        
        uploaded_file = st.file_uploader("Transkripsiyon için ses dosyası yükleyin", type=["wav", "mp3", "ogg"], key="transcribe_file")
        language = st.selectbox("Ses dilini seçin:", ["tr", "en", "fr", "de", "es"], index=0, key="transcribe_lang")
        
        if uploaded_file is not None:
            # Geçici dosyaya kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            
            if st.button("Transkripsiyonu Başlat", key="transcribe_button"):
                with st.spinner("Transkripsiyon yapılıyor..."):
                    try:
                        transcription = transcribe_audio(audio_path, language)
                        st.success("Transkripsiyon tamamlandı!")
                        st.text_area("Transkripsiyon Sonucu:", value=transcription, height=2000)
                    except Exception as e:
                        st.error(f"Transkripsiyon hatası: {str(e)}")
                    finally:
                        # Geçici dosyayı sil
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
    
    # Whisper ile Tercüme
    with tabs[2]:
        st.header("Sesli Tercüme")
        
        uploaded_file = st.file_uploader("Tercüme için ses dosyası yükleyin", type=["wav", "mp3", "ogg","ts"], key="translate_file")
        target_language = st.selectbox("Hedef dili seçin:", ["en", "tr", "fr", "de", "es"], index=0)
        
        if uploaded_file is not None:
            # Geçici dosyaya kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            
            if st.button("Tercümeyi Başlat", key="translate_button"):
                with st.spinner("Tercüme yapılıyor..."):
                    try:
                        translation = translate_audio(audio_path, target_language)
                        st.success("Tercüme tamamlandı!")
                        st.text_area("Tercüme Sonucu:", value=translation, height=2000)
                    except Exception as e:
                        st.error(f"Tercüme hatası: {str(e)}")
                    finally:
                        # Geçici dosyayı sil
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
    
    # Wav2Vec2 ile Transkripsiyon
    with tabs[3]:
        st.header("Wav2Vec2 Transkripsiyon")
        
        uploaded_file = st.file_uploader("Wav2Vec2 transkripsiyon için ses dosyası yükleyin", type=["wav", "mp3", "ogg"], key="conformer_file")
        
        if uploaded_file is not None:
            # Geçici dosyaya kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                audio_path = tmp_file.name
            
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
            
            if st.button("Wav2Vec2 Transkripsiyonu Başlat", key="conformer_button"):
                with st.spinner("Wav2Vec2 transkripsiyon yapılıyor..."):
                    try:
                        transcription = conformer_transcribe(audio_path)
                        st.success("Transkripsiyon tamamlandı!")
                        st.text_area("Transkripsiyon Sonucu:", value=transcription, height=2000
                                     )
                    except Exception as e:
                        st.error(f"Transkripsiyon hatası: {str(e)}")
                    finally:
                        # Geçici dosyayı sil
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)

if __name__ == "__main__":
    main()