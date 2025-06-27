# Mini-Asystent Wiedzy
## Autor: Krzysztof Wyrzykowski

### Instrukcja uruchomienia
Należy sklonować repozytorium.
```
git clone https://github.com/kwyrzyk/RAG.git
```
Wejść do folderu z projektem
```
cd RAG
```
Teraz należy pobrać wszystkie wymagane biblioteki.
```
pip install -r requirements.txt
```
Dokumenty o którę chcemy pytać system należy umieścic w folderze `documents`.
Aby korzysztać z wybranego modelu `Llama-3.2-3B-Instruct` należy umieści swój token HuggingFace w pliku `hf_token.txt`. Token musi mieć dostęp do tego modelu.

Następnie można uruchomic skrypt.
```
python3 main.py
```
Możemy teraz zadać pytanie do naszym dokumentów.

### Opis rozwiązania

Cały projekt podzieliłem na trzy moduły.

##### DocumentsLoader

Odpowiada ze wczytanie plików `pdf` i `txt` z folderu `documents`, podział wczytanych plików na chunki oraz zapis do bazy. Do tych operacji wykorzystałem funkcje z biblioteki `LangChain`. Chunki przechowuje w bazie 'ChromaDB'. Parametry chunkowania:
- chunk size - 300
- chunk overlap - 100
Chunki są konwertowane na wektory za pomocą modelu `all-MiniLM-L6-v2`.

##### Retriever

Przetwarza zapytanie na wektor i wyszukanie w bazie najbardziej pasujących wektorów do zapytania.

##### ResponseGenerator

Przygotowuje prompt do modelu językowego na podstawie 3 najbardziej pasujących chunków i odpytuje model językowy. Z odpowiedzi tworzy wiadomość do użytkownika.
Zdecydowałem się na model `Llama-3.2-3B-Instruct`, ponieważ dobrze radził sobie zarówno z językiem polskim jaki i angielskim.

### Największe wyzwania

Zdecydowanie największym problemem okazało się znalezienie modelu, który był w stanie działać na moim laptopie i jednocześnie udzielał odpowiedzi na zadowalającym poziomie. Problemy z tym związany wynikł głównie z mojej słabej znajomości środowiska GooglColab i problemów z wykorzystaniem GPU.

### Refleksje

Przede wszystkim powinniem przetestować więcej konfiguracji i rozmiarów chunków i overlapa. Jednakże nie wystarczyło czasu z powodu wymienionego w poprzednim punkcie.

### Przykład działanie

```
"Kto utworzył Irańską Radę Rewolucyjną"
Response: Chomejni utworzyli Irańską Radę Rewolucyjną.
Sources: documents/Historia_Iranu.pdf
```
```
"Ile państw jest w Azji?"
Response: The context does not contain enough information.
Sources: documents/Historia_Iranu.pdf
```

```
"What is the motto of European Union"
Response: The context does not contain enough information.
Sources: documents/Historia_Iranu.pdf
```
Czasami model nie udziela odpowiedzi po mimo informacji w pliku.
```
"Jaka jest stolic Iranu"
Response: The context does not contain enough information.
Sources: documents/Historia_Iranu.pdf
```










