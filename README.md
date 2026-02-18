# Sorbot Technical Specification

Sorbot is a distributed algorithmic trading system consisting of a Java/Spring Boot backend, a React frontend, and a Python-based machine learning inference engine.

(Figma|https://www.figma.com/make/H0x4DruIOw6sZ7Qwu1e2d3/5-Page-Crypto-Trading-Website?p=f&t=ezv0vYgc4ttAnvFq-0)

---

## 1. System Architecture

The system follows a decoupled microservices pattern. Communication between the Application layer and the Machine Learning layer is handled via RESTful API calls to ensure language-agnostic development.

### 1.1 Service Interaction (API Contract)
The Application layer polls the AI Engine via the following specification:

**Endpoint:** `GET /predict?symbol={string}`

**Response Schema:**
```json
{
  "symbol": "string",
  "direction": "BUY | SELL | HOLD",
  "confidence": "float [0.0, 1.0]",
  "current_price": "float",
  "timestamp": "ISO-8601",
  "indicators": {
    "rsi": "float",
    "macd": "string"
  }
}

```

---

## 2. Component Specifications

### 2.1 Backend Application (Java/Spring Boot)

* **Persistence Layer:** PostgreSQL/H2 Database.
* `Users`: id (PK), username, password_hash, role.
* `Portfolio`: user_id (FK), symbol, quantity, average_price.
* `TradeHistory`: id (PK), symbol, action, execution_price, quantity, timestamp.
* `BotSettings`: user_id (FK), risk_threshold, is_active.


* **Execution Engine:** Implementation of `@Scheduled(fixedRate = 60000)` to trigger periodic inference requests.
* **Service Layer:** Interface-driven design for `PredictionService` allowing for a `MockPredictionServiceImpl` during development and a `RestPredictionServiceImpl` for production.

### 2.2 Frontend Dashboard (React)

* **State Management:** React Query for asynchronous data fetching and automatic polling every 2000ms.
* **Visualizations:** Recharts library for time-series price data and technical indicator overlays.
* **Components:**
* `MarketDepth`: Tabular representation of current holdings.
* `OrderBook`: Log of historical transactions.
* `ConfigurationInterface`: CRUD operations for `BotSettings`.



### 2.3 AI Engine (Python/FastAPI)

* **Data Ingestion:** Integration with `yfinance` or `ccxt` for OHLCV data retrieval.
* **Feature Engineering:** Calculation of technical indicators (RSI, Bollinger Bands, ATR) using `pandas` or `TA-Lib`.
* **Inference Model:** XGBoost (Extreme Gradient Boosting) classifier.
* Model persistence via `model.json` or `pickle`.


* **API Layer:** FastAPI implementation to serve model predictions with sub-second latency.

---

## 3. Infrastructure and Deployment

The system is designed for containerized deployment using Docker Compose to manage environment variables and network links between services.

```yaml
version: '3.8'
services:
  ai-engine:
    build: ./ai-engine
    ports:
      - "8000:8000"
  backend-api:
    build: ./backend-api
    ports:
      - "8080:8080"
    environment:
      - AI_SERVICE_URL=http://ai-engine:8000
  frontend-client:
    build: ./frontend-client
    ports:
      - "3000:3000"

```

---

## 4. Development Roadmap

### Phase 1: Environment Setup

* Initialize Spring Boot project with JPA and Security dependencies.
* Establish Python virtual environment and FastAPI skeleton.

### Phase 2: Implementation

* **App Team:** Define JPA entities and implement the trade execution logic loop.
* **AI Team:** Implement data pipeline and baseline XGBoost training script.

### Phase 3: Integration

* Replace Mock services in Java with `RestTemplate` or `WebClient` calls to the Python API.
* Finalize React dashboard components for real-time telemetry.
