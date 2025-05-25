
## Dosya : deneme2.py (Hugging Face Modelleri ile Ses İşleme)

Bu dosya, Hugging Face Transformers kütüphanesindeki modelleri kullanarak (ilk çalıştırmada modelleri indirerek) ses işleme görevlerini gerçekleştiren bir Streamlit uygulamasıdır.

### Proje Kapsam Dosyası: Ses İşleme Araçları (Hugging Face Tabanlı)

**1. Proje Adı:**  
Ses İşleme Araçları (Hugging Face Modelleri ile)

**2. Proje Amacı:**  
Bu proje, Hugging Face Transformers kütüphanesinde bulunan son teknoloji ürünü (SOTA) ses modellerini kullanarak metinden ses sentezleme (TTS), ses dosyalarından metin transkripsiyonu ve sesli tercüme yapabilen kullanıcı dostu bir web uygulaması sunmayı amaçlamaktadır. Tüm işlemler, modellerin yerel olarak çalıştırılması (veya ilk kullanımda indirilmesi) prensibine dayanır.

**3. Ana Özellikler:**

- **Metinden Sese Dönüştürme (TTS):**
  
  - Kullanılan Model: microsoft/speecht5_tts (ana model) ve microsoft/speecht5_hifigan (vocoder).
    
  - Matthijs/cmu-arctic-xvectors veri setinden örnek konuşmacı gömmeleri (speaker embeddings) kullanarak ses üretimi.
    
  - Kullanıcının girdiği metni seçilen dilde (TR, EN, FR, DE, ES) sese dönüştürme.
    
  - Oluşturulan sesi .wav formatında geçici bir dosyaya kaydedip arayüzde çalma ve sonrasında dosyayı silme.
    
- **Whisper ile Transkripsiyon:**
  
  - Kullanılan Model: openai/whisper-medium.
    
  - Kullanıcının yüklediği ses dosyasını (WAV, MP3, OGG) metne dönüştürme.
    
  - Sesin orijinal dilini seçebilme (TR, EN, FR, DE, ES).
    
  - Otomatik ses ön işleme:
    
    - Gerektiğinde tek kanala (mono) dönüştürme.
      
    - Gerektiğinde 16kHz örnekleme hızına yeniden örnekleme (resample).
      
  - Transkripsiyon sonucunu metin alanında gösterme.
    
  - İşlem için yüklenen sesi geçici dosyaya kaydetme ve işlem sonrası silme.
    
- **Whisper ile Sesli Tercüme:**
  
  - Kullanılan Model: openai/whisper-medium.
    
  - Kullanıcının yüklediği ses dosyasını (WAV, MP3, OGG, TS) doğrudan hedeflenen dile tercüme etme.
    
  - Hedef dil seçimi (EN, TR, FR, DE, ES).
    
  - Otomatik ses ön işleme (yukarıdaki gibi).
    
  - Tercüme sonucunu metin alanında gösterme.
    
  - İşlem için yüklenen sesi geçici dosyaya kaydetme ve işlem sonrası silme.
    
- **Wav2Vec2 ile Transkripsiyon (Arayüzde "Conformer ile Transkripsiyon" olarak adlandırılmış):**
  
  - Kullanılan Model: facebook/wav2vec2-large-960h.
    
  - Kullanıcının yüklediği ses dosyasını (WAV, MP3, OGG) metne dönüştürme.
    
  - Otomatik ses ön işleme (yukarıdaki gibi).
    
  - Transkripsiyon sonucunu metin alanında gösterme.
    
  - İşlem için yüklenen sesi geçici dosyaya kaydetme ve işlem sonrası silme.
    
- **Kullanıcı Arayüzü:**
  
  - Streamlit ile sekmeli (tabs) arayüz.
    
  - Dosya yükleme (st.file_uploader), metin girişi (st.text_area), dil seçimi (st.selectbox) bileşenleri.
    
  - İşlem sırasında yükleme göstergeleri (st.spinner).
    
  - Hata mesajlarını (st.error) kullanıcıya gösterme.
    
- **Verimlilik ve Kaynak Yönetimi:**
  
  - @st.cache_resource dekoratörü ile modellerin bir kez yüklenip bellekte tutulması, böylece uygulama içinde tekrar tekrar indirilmesinin/yüklenmesinin önlenmesi.
    
  - Yüklenen ve üretilen ses dosyaları için geçici dosyaların (tempfile.NamedTemporaryFile) kullanımı ve işlem sonrası (os.unlink) güvenli bir şekilde silinmesi.
    

**4. Kullanılan Teknolojiler:**

- **Programlama Dili:** Python
  
- **Web Framework:** Streamlit
  
- **Makine Öğrenmesi:** PyTorch, Hugging Face Transformers, Hugging Face Datasets
  
- **Ses İşleme:** Torchaudio, SoundFile
  
- **Veri İşleme:** NumPy
  
- **Standart Kütüphaneler:** os, tempfile, io
  

**5. Hedef Kitle:**

- Yerel kaynaklarını kullanarak (model indirme sonrası) gelişmiş ses işleme yeteneklerine erişmek isteyen kullanıcılar.
  
- İçerik üreticileri, dilbilimciler, öğrenciler.
  
- Ses teknolojileri ve Hugging Face ekosistemiyle çalışan geliştiriciler ve araştırmacılar.
  

**6. Kısıtlamalar ve Varsayımlar:**

- **Model İndirme:** Modeller ilk çalıştırmada Hugging Face Hub'dan indirilecektir. Bu, aktif bir internet bağlantısı ve yeterli disk alanı gerektirir (örneğin, whisper-medium yaklaşık 1.5GB, speecht5 modelleri toplamda ~500MB, wav2vec2-large-960h yaklaşık 1.2GB).
  
- **Performans:** İşlemler (özellikle büyük modellerle CPU üzerinde) zaman alabilir. GPU kullanımı (eğer sistemde uygun PyTorch kurulumu varsa) performansı önemli ölçüde artırır. Kod, PyTorch'un varsayılan cihaz atamasını kullanır.
  
- "Conformer ile Transkripsiyon" sekmesi aslında facebook/wav2vec2-large-960h modelini kullanmaktadır; bu, modelin adı/mimarisi olan Conformer'dan ziyade, bir model ailesi olan Wav2Vec2'ye aittir.
  
- Desteklenen ses dosyası formatları, st.file_uploader içindeki type parametreleriyle sınırlıdır.
  

### Nasıl Çalıştırılır? (deneme2.py)

1. **Dosyayı Kaydetme:**  
  Kodu deneme2.py olarak bir Python dosyasına kaydedin.
  
2. **Ön Koşullar:**
  
  - Python 3.8 veya üzeri.
    
  - pip paket yöneticisi.
    
3. **Gerekli Kütüphanelerin Kurulumu:**  
  Terminali veya komut istemcisini açın ve deneme2.py dosyasının bulunduğu dizine gidin. Ardından aşağıdaki komutları çalıştırarak gerekli Python kütüphanelerini yükleyin:
  
  ```
   `pip install streamlit numpy sentencepiece torch torchaudio transformers datasets soundfile`
  ```
  

Not: torch ve torchaudio kurulumu bazen işletim sisteminize ve eğer varsa CUDA (GPU için) sürümünüze göre özelleştirme gerektirebilir. Sorun yaşarsanız, PyTorch resmi web sitesinden (pytorch.org) sisteminize uygun kurulum komutunu almanız en sağlıklısıdır.

4. **Uygulamanın Çalıştırılması:**  
  Terminalde, deneme2.py dosyasının bulunduğu dizindeyken aşağıdaki komutu çalıştırın:
  
  ```
   `streamlit run deneme2.py`
  ```
  

Bu komut, Streamlit uygulamasını başlatacak ve varsayılan web tarayıcınızda genellikle http://localhost:8501 adresinde açacaktır.

5. **İlk Çalıştırma Notu:**  
  Uygulama ilk kez çalıştırıldığında veya herhangi bir sekmedeki bir model ilk kez kullanılacağında, Hugging Face Hub'dan ilgili modeller ve işlemciler indirilecektir. Bu işlem internet hızınıza ve modelin boyutuna bağlı olarak birkaç dakika sürebilir. İndirme tamamlandıktan sonra modeller @st.cache_resource sayesinde önbelleğe alınır ve sonraki çalıştırmalar veya aynı modelin tekrar kullanımı çok daha hızlı olur.

## Dosya 2: format_Donustur.py (TS'den MP3'e Dönüştürücü)

Bu dosya, .ts (Transport Stream) dosyalarını .mp3 formatına dönüştürmek için ffmpeg kullanan basit bir Streamlit uygulamasıdır.

### Proje Kapsam Dosyası: TS to MP3 Dönüştürücü (FFmpeg ile)

**1. Proje Adı:**  
TS to MP3 Dönüştürücü

**2. Proje Amacı:**  
Bu proje, kullanıcıların .ts (Transport Stream) formatındaki video/ses dosyalarını yükleyerek bu dosyaların ses bileşenini .mp3 formatına dönüştürmelerini sağlayan basit bir web uygulaması sunmayı amaçlamaktadır. Dönüştürme işlemi için sistemde kurulu FFmpeg aracı kullanılır.

**3. Ana Özellikler:**

- .ts dosyası yükleme (st.file_uploader).
  
- Yüklenen .ts dosyasının ses akışını libmp3lame kodeği kullanarak MP3'e dönüştürme.
  
- Video bileşenini göz ardı etme (-vn FFmpeg parametresi).
  
- Belirli bir ses kalitesiyle (-q:a 2 FFmpeg parametresi) dönüştürme.
  
- Dönüştürme işlemi sırasında kullanıcıya ilerleme bilgisi (st.spinner).
  
- Dönüştürülen MP3 dosyasını indirme butonu (st.download_button) ile kullanıcıya sunma.
  
- Uygulama başlangıcında FFmpeg'in sistemde kurulu olup olmadığını kontrol etme ve kullanıcıyı bilgilendirme (st.error).
  
- Yüklenen dosya ve dönüştürülen çıktı için geçici dosyaların (tempfile.NamedTemporaryFile) güvenli bir şekilde yönetimi ve işlem sonrası otomatik olarak (os.unlink) silinmesi.
  

**4. Kullanılan Teknolojiler:**

- **Programlama Dili:** Python
  
- **Web Framework:** Streamlit
  
- **Harici Araç:** FFmpeg (sistemde kurulu ve PATH'e ekli olmalı)
  
- **Sistem Komutları:** subprocess modülü ile FFmpeg çalıştırma.
  
- **Standart Kütüphaneler:** os, tempfile
  

**5. Hedef Kitle:**

- .ts dosyalarından hızlıca ses (audio track) çıkarmak ve MP3 olarak kaydetmek isteyen kullanıcılar.
  
- Video arşivleri veya yayın kayıtları ile çalışan ve bu kayıtlardan sesi ayıklamak isteyenler.
  

**6. Kısıtlamalar ve Varsayımlar:**

- **FFmpeg Bağımlılığı:** Uygulamanın çalışması için **FFmpeg'in sistemde kurulu ve sistemin PATH ortam değişkenine ekli olması zorunludur.** Aksi takdirde uygulama hata verecektir.
  
- Uygulama, yüklenen .ts dosyasının geçerli bir ses akışı içerdiğini varsayar.
  
- Dönüştürme süresi, orijinal .ts dosyasının boyutuna ve sistem kaynaklarına bağlı olarak değişebilir.
  

### Nasıl Çalıştırılır? (format_Donustur.py)

1. **Dosyayı Kaydetme:**  
  Kodu format_Donustur.py olarak bir Python dosyasına kaydedin.
  
2. **Ön Koşullar:**
  
  - Python 3.7 veya üzeri.
    
  - pip paket yöneticisi.
    
  - **FFmpeg:** Sisteminizde FFmpeg'in kurulu ve PATH ortam değişkenine eklenmiş olması gerekmektedir.
    
    - **Windows:** FFmpeg'i resmi sitesinden (ffmpeg.org) indirip, bin klasörünü sisteminizin PATH ortam değişkenlerine ekleyebilirsiniz.
      
    - **Linux (Debian/Ubuntu):** sudo apt update && sudo apt install ffmpeg
      
    - **Linux (Fedora):** sudo dnf install ffmpeg
      
    - **macOS (Homebrew ile):** brew install ffmpeg  
      FFmpeg'in doğru kurulup kurulmadığını terminalde ffmpeg -version komutuyla kontrol edebilirsiniz.
      
3. **Gerekli Kütüphanelerin Kurulumu:**  
  Bu uygulama için sadece Streamlit'in kurulu olması yeterlidir. Terminali veya komut istemcisini açın ve format_Donustur.py dosyasının bulunduğu dizine gidin. Ardından aşağıdaki komutu çalıştırın:
  
  ```
   `pip install streamlit`
  ```
  

IGNORE_WHEN_COPYING_START

content_copy download

Use code [with caution](https://support.google.com/legal/answer/13505487). Bash

IGNORE_WHEN_COPYING_END

4. **Uygulamanın Çalıştırılması:**  
  Terminalde, format_Donustur.py dosyasının bulunduğu dizindeyken aşağıdaki komutu çalıştırın:
  
  ```
   `streamlit run format_Donustur.py`
  ```
  

IGNORE_WHEN_COPYING_START

content_copy download

Use code [with caution](https://support.google.com/legal/answer/13505487). Bash

IGNORE_WHEN_COPYING_END

Bu komut, Streamlit uygulamasını başlatacak ve varsayılan web tarayıcınızda genellikle http://localhost:8501 adresinde açacaktır. Eğer FFmpeg kurulu değilse, uygulama açıldığında bir hata mesajı gösterecektir.

---
