# G-TRON 2.0: Comprehensive Project Architecture & Technical Analysis

> **Platform Overview**: G-TRON 2.0 is an end-to-end, AI-powered **E-Waste Management, Analytics, and Regulatory Compliance Platform** tailored specifically for the Indian electronics and recycling ecosystem. It bridges circular economy principles, real-time IoT monitoring, advanced machine learning, and Indian regulatory compliance (CPCB / EPR rules).

---

## 1. Executive Summary & Core Mission

India is the third-largest producer of electronic waste globally, generating millions of metric tonnes annually across hubs like Mumbai, Delhi, Bengaluru, and Chennai. The majority of this waste is processed through informal, hazardous channels.

**G-TRON 2.0** addresses this crisis through:
1. **Predictive Intelligence**: Machine learning models predicting state-wise waste generation trends (Meta Prophet) and economic recovery potential (XGBoost, CatBoost).
2. **Computer Vision & Multimodal AI**: Real-time identification of electronic devices and hazardous components using Qwen2-VL, Google Vision API, ResNet, and Gemini 1.5 Flash.
3. **Regulatory Automation**: Central Pollution Control Board (CPCB) compliance tracking, automated generation of statutory Form 1 / 2 / 3, Extended Producer Responsibility (EPR) tracking, and ESG scoring.
4. **IoT & Edge Monitoring**: Anomaly detection (Isolation Forest) on sensor feeds (gas, temperature, vibration, humidity, weight) to catch battery leaks, toxic off-gassing, and fraudulent dumping.
5. **Geospatial & Route Optimization**: Citizen-driven illegal dumping reporting with Firebase & Google Maps / Leaflet, plus fleet routing algorithms for collection trucks.
6. **Decentralized Marketplace & Client-Side AI**: Neural networks running directly inside browser memory via TensorFlow.js to estimate e-waste market valuation across Indian states.

---

## 2. High-Level System Architecture

G-TRON 2.0 is designed as a **hybrid multi-layer architecture**:
- **Presentation Layer**: React 18 Single-Page Application (SPA) bundled with Vite, styled with Tailwind CSS, utilizing Recharts, Leaflet, and Google Maps API.
- **Micro-App Streaming Layer**: An iframe-integrated mesh of individual Python Streamlit applications running on isolated localhost ports.
- **Microservices Layer**: Flask REST APIs providing prediction endpoints for carbon footprint calculation, route planning, and live computer vision inference.
- **Client-Side Edge ML**: Browser-native TensorFlow.js deep neural network running locally on the client's GPU/CPU.
- **External AI & Cloud Integrations**:
  - Google Gemini 1.5 Flash (for G-Tron conversational assistant and LangGraph compliance reasoning)
  - Hugging Face Gradio Client (`Qwen/Qwen2-VL-7B-Instruct` for vision detection)
  - Firebase (Cloud Firestore & Firebase Cloud Storage for dumping reports and media)
  - Neo4j / Py2Neo (Graph relationships for recycler-waste supply chains)
  - Local LLM inference (Ollama running `llama3.2:3b` for local offline reasoning)

### Architecture Diagram

```
+----------------------------------------------------------------------------------------------------+
|                                    REACT FRONTEND (Vite / Port 5173)                               |
|  - App.jsx / React Router DOM                                                                      |
|  - Dashboard Overview (Recharts metrics & trends)                                                  |
|  - Dumping Reporting UI (Google Maps Heatmap & Leaflet)                                            |
|  - In-Browser E-Waste Valuation Model (TensorFlow.js)                                              |
|  - Static Embeds (/gtron.html, /carbon.html, /live.html)                                           |
+--------------------+-------------------------------+-----------------------------+-----------------+
                     |                               |                             |
      (IFrame Streamlit Embeds)               (REST Calls)                  (Direct Cloud)
                     |                               |                             |
                     v                               v                             v
+--------------------+--------------+   +-----------+-------------+   +---------------+----------------+
| STREAMLIT MICRO-APP MESH          |   | FLASK REST SERVERS      |   | CLOUD & EXTERNAL SERVICES      |
|  :8501 - esp.py (ESG Tracking)    |   |  :5000 - app.py         |   |  - Google Gemini 1.5 Flash     |
|  :8502 - epr.py (EPR Compliance)  |   |    (Route Optimization) |   |  - HuggingFace Qwen2-VL        |
|  :8503 - lifecycle.py (Lifecycle) |   |  :5000 - live.py        |   |  - Firebase Firestore/Storage  |
|  :8504 - classification.py (Vision|   |    (Vision Detect API)  |   |  - Google Maps & Vision API    |
|  :8505 - toxmat.py (Toxic Detect) |   |  :5000 - carbon.py      |   |  - Neo4j Knowledge Graph       |
|  :8506 - iotsensor.py (IoT Anomaly|   |    (CO2e Predictor)     |   |  - Local Ollama (llama3.2)     |
|  :8507 - obd.py (Device Detect)   |   +-------------------------+   +--------------------------------+
|  :8508 - illegal.py (Dumping Maps)|
|  :8512 - compilance.py (CPCB/Gov) |
|  :8513 - wa.py (State Analytics)  |
|  :8514 - ma.py (Material Recovery)|
|  :8516 - buyrecycle.py (Market)   |
+-----------------------------------+
```

---

## 3. Technology Stack Breakdown

| Layer | Technologies & Libraries | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | React 18, Vite 5, Tailwind CSS 3, Lucide React, clsx, tailwind-merge | Modern, responsive dark-mode dashboard interface |
| **Data Visualization**| Recharts, Leaflet, react-leaflet, @react-google-maps/api, Plotly Express | Time-series trends, route mapping, GIS choropleths, heatmaps |
| **Client-Side ML** | `@tensorflow/tfjs` (TensorFlow.js) | In-browser dense neural network for e-waste scrap price calculation |
| **Backend Frameworks**| Python 3.10+, Streamlit, Flask, Flask-CORS | Rapid interactive micro-apps, RESTful inference endpoints |
| **Data Processing** | Pandas, NumPy, GeoPandas, PapaParse, XLSX, Scikit-learn (StandardScaler) | Tabular analytics, geospatial coordinate joins, format parsing |
| **Machine Learning** | Scikit-Learn (Isolation Forest, MLPRegressor), XGBoost, CatBoost, Meta Prophet | Anomaly detection, yield prediction, time-series e-waste forecasting |
| **Deep Learning & CV**| PyTorch, TensorFlow / Keras (`best_resnet_fold_5.h5`), Diffusers (Stable Diffusion v1.5), Qwen2-VL | Image classification, toxic waste generation, device detection |
| **LLMs & GenAI** | Google Generative AI (Gemini 1.5 Flash), LangChain, LangGraph, Ollama (`llama3.2:3b`), spaCy, TextBlob | Conversational chatbot, multi-agent compliance drafting, sentiment/ESG |
| **Databases & Cloud** | Firebase Firestore, Firebase Cloud Storage, Neo4j (`py2neo`) | Cloud persistence for reports, images, and supply chain graphs |

---

## 4. Deep Directory & File-by-File Walkthrough

### 4.1 Root Directory & Orchestration

- `main.py`: **Master Launcher Process**.
  - Iterates through a configured registry of Streamlit apps (`esp.py`, `epr.py`, `lifecycle.py`, `classification.py`, `toxmat.py`, `iotsensor.py`, `illegal.py`, `compilance.py`, `wa.py`, `ma.py`, `buyrecycle.py`) and launches each on its designated port (8501–8516) using `subprocess.Popen`.
  - Spawns background scripts such as `carbon.py` and `live.py`.
- `app.py`: **Flask Route Optimization & LLaMA Waste Predictor API**.
  - Exposes endpoints `/api/initial-data`, `/api/optimize-routes`, and `/api/predict-waste`.
  - Integrates truck fleet allocations, capacity bounds, and collection points across Indian cities (e.g., Ramdaspeth, Dharampeth, Civil Lines).
- `package.json` & `vite.config.js`:
  - Configures Vite with React plugin, path aliases (`@` resolving to `src/`), and client dependencies including Leaflet, Google Maps, TensorFlow.js, and Recharts.

---

### 4.2 Machine Learning & Streamlit Applications

#### A. Analytics & Forecasting
- `wa.py` (and duplicate in `dataset/ewassteanalytics/wa.py`): **E-Waste Geographic & Temporal Analytics**.
  - **Meta Prophet Integration**: Generates 5-year future projections of e-waste generation by state and device category.
  - **XGBoost & CatBoost Ensemble**: Evaluates hazard scores and economic recovery potentials.
  - **Choropleth Mapping**: Uses `india_state_geo.json` via Plotly Express to render state-wise e-waste densities across India.
  - **Neo4j Graph**: Connects states, device classes, and recyclers into a knowledge network.
- `train_prophet.py`: Script to train and persist `prophet_karnataka.pkl`.
- `train_xgboost.py`: Trains an `XGBRegressor` on `e_waste_data.csv` to predict waste volume from lead, plastic, and informal collection rates. Saves `xgboost_model.joblib` and `scaler.joblib`.
- `train_catboost.py`: Trains a `CatBoostRegressor` to predict `economic_recovery_potential`. Saves `catboost_model.cbm` and `catboost_scaler.joblib`.

#### B. Regulatory Compliance & Governance
- `compilance.py`: **Indian E-Waste Compliance Reporting System**.
  - Implements Central Pollution Control Board (CPCB) compliance rules under Ministry of Environment, Forest and Climate Change (MoEFCC).
  - Uses Google Gemini 1.5 Flash to automatically draft and validate legal compliance filings, hazardous substance threshold limits (RoHS), and annual returns.
  - Provides state-wise compliance tracking, penalty simulations, and interactive risk metrics.
- `epr.py`: **Extended Producer Responsibility (EPR) Calculator**.
  - Loads `esg_model.pkl` and `scaler.pkl`.
  - Takes inputs: ESG score, energy usage, carbon emissions, renewable adoption, waste recycling %, water conservation, and regulatory fines.
  - Computes compliance status (Compliant vs Non-compliant) and classification probability distributions.
- `esp.py`: **ESG (Environmental, Social, Governance) Intelligence Platform**.
  - Uses `spacy` (`en_core_web_sm`) and `TextBlob` for natural language processing on corporate environmental reports.
  - Performs keyword extraction and sentiment polarity/subjectivity analysis.
  - Predicts corporate ESG classifications via `esg_model.pkl`.

#### C. Computer Vision & Device Classification
- `classification.py`: **AI E-Waste Material Breakdown**.
  - Uploaded images are passed to `Qwen/Qwen2-VL-7B-Instruct` via Gradio client to identify visible components (PCBs, casings, batteries, wires).
  - Component descriptions are piped into Google Gemini 1.5 Flash to compute fractional material breakdown percentages (e.g., Plastic 40%, Copper 25%, PCB 15%).
  - Visualizes breakdown dynamically with custom dark-themed Matplotlib bar charts.
- `toxmat.py`: **Toxic Material Detector**.
  - Combines Google Cloud Vision Object Localization with local Ollama (`llama3.2:3b`) reasoning.
  - Detects hazardous components (lithium batteries, lead solder, flame-retardant plastics) and draws colored bounding polygons (Red = Toxic, Blue = Non-toxic).
- `obd.py`: **Electronic Device Object Detector**.
  - Streamlit interface utilizing `Qwen2-VL-7B-Instruct` to locate and catalogue discarded electronic models.
- `toxicdetector.py` & `test_tf.py`:
  - Flask endpoint serving a deep convolutional neural network (`best_resnet_fold_5.h5`) predicting binary "Toxic" vs "Non-Toxic" waste confidence.
- `toxic.py`:
  - Synthetic data generation script utilizing Hugging Face `diffusers` (`runwayml/stable-diffusion-v1-5`) to synthesize realistic images of hazardous landfill dumps vs sorted clean recycling plants.

#### D. IoT Sensor Monitoring & Anomaly Detection
- `iotsensor.py` & `anamoly.py`:
  - Loads `e_waste_anomaly_dataset_with_reasons.csv`.
  - Employs an `IsolationForest` model (5% contamination rate) monitoring 5 sensor channels:
    1. **Weight (grams)**
    2. **Gas Concentration (ppm)**
    3. **Temperature (°C)**
    4. **Humidity (%)**
    5. **Vibration (g-force)**
  - Rule-based reason attribution: Classifies anomalies into **Hazardous Material Leak**, **Recycling Efficiency Issue**, **Component Degradation**, or **Fraudulent Disposal**.

#### E. Material Recovery & Circular Economy Marketplace
- `ma.py`: **Material Recovery Prediction**.
  - Reads detailed elemental compositions (Gold, Silver, Copper, Aluminum, Plastic, Lithium) for smartphones, laptops, tablets, and LEDs.
  - Utilizes a multi-layer perceptron neural network (`e_waste_model.joblib`, trained via `train24.py`) to estimate net precious metal recovery yields.
- `buyrecycle.py`: **Recycler & Refurbished Marketplace Assistant**.
  - LangChain-powered conversational bot with Tavily Search integration.
  - Simulates an intelligent market connecting buyers, sellers, and certified recyclers with real-time price estimation and warranty checks.
- `carbon.py`: **Carbon Footprint Microservice**.
  - Flask endpoint `/predict` utilizing `carbon_footprint_model.pkl` and `scaler2.pkl`.
  - Takes weights of plastics, aluminum, copper, lithium, gold, glass, plus manufacturing energy and transport distance to calculate total kg CO₂ equivalent emissions saved.

---

### 4.3 Frontend (`src/`) Architecture

- `src/App.jsx`:
  - Defines the client-side router with `createBrowserRouter`.
  - Connects native React routes:
    - `/` -> `DashboardPage` (KPI statistics, processing overview area chart, recent transactions).
    - `/dumping` -> `DumpingPage` (Firebase Firestore real-time reporting & Google Maps Heatmap).
    - `/pricing` -> `EWastePricePredictorIndia` (TensorFlow.js in-browser AI model).
    - `/route` -> `RouteOptimization` (Leaflet polyline route mapping).
  - Mounts Streamlit micro-services via `StreamlitEmbedPage` component on iframes:
    - `/lifecycle` (`:8503`), `/classification` (`:8504`), `/dataset/ewassteanalytics/wa` (`:8513`), `/toxmat` (`:8505`), `/illegal` (`:8508`), `/iotsensor` (`:8506`), `/esg` (`:8520`), `/ma` (`:8519`), `/epr` (`:8502`), `/compliance` (`:8512`), `/buyrecycle` (`:8516`).
  - Redirects static standalone tools:
    - `/gtron` -> redirects to `/gtron.html`
    - `/carbon` -> redirects to `/carbon.html`
    - `/e-wastedetection` -> redirects to `/live.html`
- `src/model.js`:
  - **In-Browser Deep Learning with TensorFlow.js**.
  - Builds a sequential neural network with Dense, Batch Normalization, ReLU activation, and Dropout layers.
  - Encodes one-hot features: Device Category, Subcategory, Brand, Condition, Material, and Indian State/Location.
  - Compiles with Huber Loss and Adam optimizer; supports client-side model training, saving to browser IndexedDB/LocalStorage, and real-time scrap valuation.
- `src/components/EWastePricePredictorIndia.jsx`:
  - Interactive UI for the TensorFlow.js pricing model. Allows custom input configurations, feature weight visualizations, and batch CSV training data uploads.
- `src/routes/dumping/page.jsx`:
  - Citizen engagement interface for reporting illegal e-waste dumps.
  - Connects directly to Firebase Firestore for real-time document synchronization (`onSnapshot`) and Firebase Storage for photo uploads.
  - Displays reports on an interactive Google Map with heatmaps and individual location pins.
- `src/pages/RouteOptimization.jsx`:
  - OpenStreetMap / Leaflet viewer that decodes Google polyline strings to render fleet collection paths across city pick-up hubs.

---

### 4.4 Static Standalone Applications (`public/`)

- `public/gtron.html`:
  - Standalone dark-mode chat UI for the **G-Tron AI Assistant**.
  - Directly interfaces with Google Generative AI API (`gemini-1.5-flash`), conditioned with specialized system instructions to guide users on recycling procedures, hazardous item safety, and Indian compliance.
- `public/carbon.html`:
  - Interactive carbon footprint calculator making async fetch calls to `http://127.0.0.1:5000/predict`.
- `public/live.html`:
  - Live webcam / image upload detection interface communicating with `live.py` (`Qwen2-VL-7B-Instruct`).

---

## 5. Data Assets and Model Inventory

| File Name | Format | Role |
| :--- | :--- | :--- |
| `india_state_geo.json` | GeoJSON (22.9 MB) | Detailed administrative boundary geometries for all Indian States & UTs used in Choropleth maps. |
| `e_waste_data.csv` | Tabular CSV | Records e-waste quantity, lead content, plastic content, economic recovery value, and informal rate across states. |
| `enhanced_e_waste_data.csv`| Tabular CSV | Extended dataset with multi-device metrics and environmental conditions. |
| `e_waste_anomaly_dataset_with_reasons.csv` | Tabular CSV | Labeled IoT sensor readings with classified failure reasons. |
| `xgboost_model.joblib` | Scikit-learn/Joblib | Trained XGBoost Regressor for e-waste quantity predictions. |
| `catboost_model.cbm` | CatBoost Binary | Trained CatBoost model for economic recovery valuation. |
| `prophet_karnataka.pkl`| Pickle | Trained Meta Prophet model for Karnataka time-series trends. |
| `carbon_footprint_model.pkl` | Pickle | Linear/ensemble regression model for CO₂e emission calculation. |
| `esg_model.pkl` | Pickle | Multi-class classification model for corporate ESG ratings. |
| `enhanced_e_waste_model.pkl` | Pickle | Random Forest / Gradient Boosted model for device lifecycle stage. |
| `best_resnet_fold_5.h5` | Keras HDF5 | Deep CNN for toxic vs non-toxic image classification. |

---

## 6. How to Run and Operate the Project

### Prerequisites
- Node.js (v18+) and npm
- Python (v3.10 or v3.11 recommended)
- Virtual environment (`venv` or `conda`) with required packages installed:
  `pip install streamlit flask flask-cors pandas numpy scikit-learn xgboost catboost prophet joblib torch torchvision diffusers transformers google-generativeai langchain langgraph gradio_client py2neo geopandas folium streamlit-folium plotly`

### Step 1: Launch Backend Microservices
Run the orchestrator script to spin up the Streamlit apps:
```bash
python main.py
```
*(Or launch individual apps manually, e.g., `streamlit run wa.py --server.port 8513`)*

### Step 2: Launch Flask Background APIs
In separate terminals:
```bash
python carbon.py  # Runs on port 5000 (Carbon prediction)
# OR
python live.py    # Runs on port 5000 (Vision detection)
# OR
python app.py     # Runs on port 5000 (Route optimization)
```

### Step 3: Launch React Frontend
In the root directory:
```bash
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

---

## 7. Key Findings & Engineering Recommendations

1. **API Keys in Source Code**:
   - `classification.py` and `compilance.py` contain hardcoded Google Gemini API keys.
   - *Recommendation*: Migrate these keys immediately to a `.env` file and read them via `os.environ.get("GEMINI_API_KEY")`.
2. **Flask Port Collisions**:
   - `app.py`, `carbon.py`, `live.py`, and `toxicdetector.py` all default to port `5000`.
   - *Recommendation*: Assign distinct ports (e.g., 5001 for Carbon, 5002 for Live Detection, 5003 for Route Planning) and update the corresponding frontend fetch calls.
3. **Typo in File Names**:
   - `compilance.py` is misspelled (instead of `compliance.py`).
   - `anamoly.py` is misspelled (instead of `anomaly.py`).
   - *Recommendation*: Keep existing filenames for backward-compatibility with routes, but create clean aliases or standardize them.
4. **Unified Microservice Gateway**:
   - Instead of 12 distinct Streamlit ports, a long-term production pattern is to encapsulate the ML logic into a single FastAPI / Dockerized service exposing clean REST/WebSocket endpoints that the React dashboard consumes directly.
