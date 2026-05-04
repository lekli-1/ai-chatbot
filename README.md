# AI Chatbot
## Individual Project - Cloud Native Application Development
#### This project is an AI chatbot designed to be easily integrated into various websites to fulfill customer support tasks.
### Features:
- RAG (Retrieval-Augmented Generation) for enhanced response accuracy.
- Vector database for efficient data retrieval.
- FastAPI for building the backend API.
- Docker for containerization and easy deployment.
### Technologies Used:
- Python
- FastAPI
- Docker
- Vector Database (PostGreSQL with pgvector extension)
- Terraform (for infrastructure management)
- 
### Setup Instructions:
- docker compose up -d --build (run containers)
- docker compose exec api uv run python db_setup.py (initialize database)

## Hungarian specification:
#### Általános Leírás
Felhő natív alkalmazásfejlesztés témához az általam választott feladat egy weboldalra könnyen
beágyazható widget, amelyen keresztül egy AI chatbottal lehet folytatni csevegést, amely
ügyfélszolgálati feladatokat képes ellátni.
#### Működés
A kód Docker konténerekben fog futni, majd egy felhőszolgáltató segítségével lesz üzembe
helyezve.
Többféle LLM modellel is fog működni a kód, amelyekkel API hívásokkal fog kommunikálni.
A rendszer alkalmazni fogja a RAG(Retrieval-Augmented Generation) technológiát, aminek a
segítségével hiteles kontextusa lesz a botnak, elkerülve a hallucinálást és a hamis információk
megadását. Ennek a megvalósításához egy vektor adatbázist fogok használni.
#### A felhasználó szempontjából
A felhasználó, ha felmegy egy weboldalra, ami használja ezt az applikációt, a jobb alsó
sarokban fog látni egy ikont. Az ikonra kattintva megjelenik egy csevegőablak, amelyben
kérdéseket tehet fel a weboldalt üzemeltető cég szolgáltatásairól és egyéb információkat
kérdezhet. A kérdésre azonnal válaszolni fog az AI bot, a neki adott kontextus alapján.
Az applikáció Architektúrája
A felhő natív elveknek megfelelően az applikáció mikroszolgáltatásokból fog összeállni, képes
lesz skálázódni az igényeknek megfelelően.
#### Fő részek: 
- UI – Maga a csevegőfelület, amelyen keresztül a kérdéseket fel lehet tenni.
- LLM feldolgozás – Itt történnek a kérdések feldolgozása, illetve az arra érkező válasz
ki generálása az LLM segítségével
- Vektor adatbázis – Itt lesznek eltárolva az üzleti információk, amelyek majd az AI
kontextusát képezik.