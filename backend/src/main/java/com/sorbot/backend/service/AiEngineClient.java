package com.sorbot.backend.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.Map;

/**
 * Client service to communicate with the Python AI Engine (FastAPI on port 8000).
 */
@Service
public class AiEngineClient {

    private static final Logger log = LoggerFactory.getLogger(AiEngineClient.class);
    private final WebClient webClient;
    private final ObjectMapper objectMapper;

    public AiEngineClient(WebClient aiEngineWebClient, ObjectMapper objectMapper) {
        this.webClient = aiEngineWebClient;
        this.objectMapper = objectMapper;
    }

    /**
     * GET /predict — Get latest AI prediction with enriched analysis.
     */
    public Map<String, Object> getPrediction() {
        try {
            String json = webClient.get()
                    .uri("/predict")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to get prediction from AI engine: {}", e.getMessage());
            throw new RuntimeException("AI Engine prediction failed: " + e.getMessage(), e);
        }
    }

    /**
     * POST /execute — Execute trade with specific prediction parameters.
     * Passes the pre-approved quantity so execution matches what the user saw.
     */
    public Map<String, Object> executeTrade(String signal, double entryPrice, double slPrice, double tpPrice, Double estQtyBtc) {
        try {
            Map<String, Object> body = new java.util.LinkedHashMap<>();
            body.put("signal", signal);
            body.put("entry_price", entryPrice);
            body.put("sl_price", slPrice);
            body.put("tp_price", tpPrice);
            if (estQtyBtc != null && estQtyBtc > 0) {
                body.put("qty_btc", estQtyBtc);
            }

            String json = webClient.post()
                    .uri("/execute")
                    .contentType(org.springframework.http.MediaType.APPLICATION_JSON)
                    .bodyValue(body)
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to execute trade via AI engine: {}", e.getMessage());
            throw new RuntimeException("Trade execution failed: " + e.getMessage(), e);
        }
    }

    /**
     * POST /trade — Execute trade on Binance via AI engine (auto mode, generates new prediction).
     */
    public Map<String, Object> executeTrade() {
        try {
            String json = webClient.post()
                    .uri("/trade")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to execute trade via AI engine: {}", e.getMessage());
            throw new RuntimeException("Trade execution failed: " + e.getMessage(), e);
        }
    }

    /**
     * POST /train — Retrain the AI model.
     */
    public Map<String, Object> trainModel() {
        try {
            String json = webClient.post()
                    .uri("/train")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to train model: {}", e.getMessage());
            throw new RuntimeException("Model training failed: " + e.getMessage(), e);
        }
    }

    /**
     * GET /status — Get account status from AI engine.
     */
    public Map<String, Object> getStatus() {
        try {
            String json = webClient.get()
                    .uri("/status")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to get status: {}", e.getMessage());
            throw new RuntimeException("Status fetch failed: " + e.getMessage(), e);
        }
    }

    /**
     * POST /close — Close open position.
     */
    public Map<String, Object> closePosition() {
        try {
            String json = webClient.post()
                    .uri("/close")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to close position: {}", e.getMessage());
            throw new RuntimeException("Close position failed: " + e.getMessage(), e);
        }
    }

    /**
     * GET /model-info — Get model metrics.
     */
    public Map<String, Object> getModelInfo() {
        try {
            String json = webClient.get()
                    .uri("/model-info")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("Failed to get model info: {}", e.getMessage());
            throw new RuntimeException("Model info fetch failed: " + e.getMessage(), e);
        }
    }

    /**
     * GET / — Health check.
     */
    public Map<String, Object> healthCheck() {
        try {
            String json = webClient.get()
                    .uri("/")
                    .retrieve()
                    .bodyToMono(String.class)
                    .block();

            return objectMapper.readValue(json, Map.class);
        } catch (Exception e) {
            log.error("AI Engine health check failed: {}", e.getMessage());
            return Map.of("status", "unreachable", "error", e.getMessage());
        }
    }
}
