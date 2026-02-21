package com.sorbot.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sorbot.backend.model.Prediction;
import com.sorbot.backend.model.Trade;
import com.sorbot.backend.model.TradingSettings;
import com.sorbot.backend.repository.PredictionRepository;
import com.sorbot.backend.repository.TradeRepository;
import com.sorbot.backend.repository.TradingSettingsRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;

/**
 * Core trading service: manages predictions, trade execution, and mode switching.
 */
@Service
public class TradingService {

    private static final Logger log = LoggerFactory.getLogger(TradingService.class);

    private final AiEngineClient aiEngineClient;
    private final PredictionRepository predictionRepo;
    private final TradeRepository tradeRepo;
    private final TradingSettingsRepository settingsRepo;
    private final SimpMessagingTemplate messagingTemplate;
    private final ObjectMapper objectMapper;

    public TradingService(
            AiEngineClient aiEngineClient,
            PredictionRepository predictionRepo,
            TradeRepository tradeRepo,
            TradingSettingsRepository settingsRepo,
            SimpMessagingTemplate messagingTemplate,
            ObjectMapper objectMapper
    ) {
        this.aiEngineClient = aiEngineClient;
        this.predictionRepo = predictionRepo;
        this.tradeRepo = tradeRepo;
        this.settingsRepo = settingsRepo;
        this.messagingTemplate = messagingTemplate;
        this.objectMapper = objectMapper;
    }

    // ── Settings ──────────────────────────────────

    public TradingSettings getSettings() {
        return settingsRepo.findAll().stream().findFirst()
                .orElseGet(() -> {
                    TradingSettings s = new TradingSettings();
                    return settingsRepo.save(s);
                });
    }

    public TradingSettings updateMode(String mode) {
        TradingSettings settings = getSettings();
        settings.setMode(mode.toUpperCase());
        settings = settingsRepo.save(settings);
        log.info("Trading mode changed to: {}", mode);

        // Broadcast mode change
        messagingTemplate.convertAndSend("/topic/settings", settings);
        return settings;
    }

    public boolean isAutoMode() {
        return "AUTO".equalsIgnoreCase(getSettings().getMode());
    }

    // ── Predictions ───────────────────────────────

    /**
     * Fetch a new prediction from the AI engine, store it, and broadcast via WebSocket.
     */
    public Prediction fetchNewPrediction() {
        Map<String, Object> raw = aiEngineClient.getPrediction();
        Prediction prediction = mapToPrediction(raw);

        TradingSettings settings = getSettings();
        prediction.setTradeMode(settings.getMode());

        // In auto mode with a tradeable signal -> auto-execute
        if ("AUTO".equalsIgnoreCase(settings.getMode())
                && !"NO_TRADE".equals(prediction.getSignal())) {
            prediction.setTradeStatus("AUTO_EXECUTING");
            prediction = predictionRepo.save(prediction);

            // Execute on Binance
            try {
                Map<String, Object> tradeResult = aiEngineClient.executeTrade();
                String action = (String) tradeResult.getOrDefault("action", "UNKNOWN");

                if ("TRADE_EXECUTED".equals(action)) {
                    prediction.setTradeStatus("AUTO_EXECUTED");
                    prediction.setTradeExecutedAt(LocalDateTime.now());

                    // Save trade record
                    Trade trade = createTradeFromPrediction(prediction, tradeResult);
                    tradeRepo.save(trade);

                    log.info("AUTO TRADE executed: {} at {}", prediction.getSignal(), prediction.getCurrentPrice());
                } else {
                    prediction.setTradeStatus("AUTO_SKIPPED");
                    prediction.setRejectReason(
                            (String) tradeResult.getOrDefault("reason",
                                    (String) tradeResult.getOrDefault("error", action)));
                }
            } catch (Exception e) {
                prediction.setTradeStatus("AUTO_FAILED");
                prediction.setRejectReason(e.getMessage());
                log.error("Auto-trade failed: {}", e.getMessage());
            }

            prediction = predictionRepo.save(prediction);
        } else if (!"NO_TRADE".equals(prediction.getSignal())) {
            prediction.setTradeStatus("PENDING");
            prediction = predictionRepo.save(prediction);
        } else {
            prediction.setTradeStatus("SKIPPED");
            prediction = predictionRepo.save(prediction);
        }

        // Broadcast to WebSocket subscribers
        messagingTemplate.convertAndSend("/topic/predictions", prediction);
        return prediction;
    }

    /**
     * Accept a pending prediction (manual mode) and execute the trade.
     */
    public Map<String, Object> acceptPrediction(Long predictionId) {
        Prediction prediction = predictionRepo.findById(predictionId)
                .orElseThrow(() -> new RuntimeException("Prediction not found: " + predictionId));

        if (!"PENDING".equals(prediction.getTradeStatus())) {
            return Map.of("error", "Prediction is not pending. Status: " + prediction.getTradeStatus());
        }

        try {
            // Use the EXACT parameters from the accepted prediction,
            // including the pre-approved quantity so execution matches what the user saw
            Map<String, Object> tradeResult = aiEngineClient.executeTrade(
                    prediction.getSignal(),
                    prediction.getCurrentPrice(),
                    prediction.getSlPrice(),
                    prediction.getTpPrice(),
                    prediction.getEstQtyBtc()
            );
            String action = (String) tradeResult.getOrDefault("action", "UNKNOWN");

            if ("TRADE_EXECUTED".equals(action)) {
                prediction.setTradeStatus("ACCEPTED");
                prediction.setTradeExecutedAt(LocalDateTime.now());

                Trade trade = createTradeFromPrediction(prediction, tradeResult);
                tradeRepo.save(trade);

                log.info("MANUAL TRADE accepted: {} at {}", prediction.getSignal(), prediction.getCurrentPrice());
            } else if ("NO_TRADE".equals(action) || "BLOCKED".equals(action) || "ERROR".equals(action)) {
                // Insufficient balance, signal blocked, or other pre-trade check failed
                String reason = (String) tradeResult.getOrDefault("reason",
                        (String) tradeResult.getOrDefault("error", action));
                prediction.setTradeStatus("INSUFFICIENT_FUNDS");
                prediction.setRejectReason(reason);
                log.warn("Trade blocked by AI engine: {}", reason);
            } else {
                prediction.setTradeStatus("EXECUTION_FAILED");
                String reason = (String) tradeResult.getOrDefault("reason",
                        (String) tradeResult.getOrDefault("error", action));
                prediction.setRejectReason(reason);
            }

            predictionRepo.save(prediction);
            messagingTemplate.convertAndSend("/topic/predictions", prediction);
            messagingTemplate.convertAndSend("/topic/trades", tradeRepo.findTop50ByOrderByExecutedAtDesc());

            return Map.of("status", "ok", "predictionId", prediction.getId(),
                    "tradeStatus", prediction.getTradeStatus(), "tradeResult", tradeResult);
        } catch (Exception e) {
            prediction.setTradeStatus("EXECUTION_FAILED");
            String errMsg = e.getMessage();
            prediction.setRejectReason(errMsg != null && errMsg.length() > 500 ? errMsg.substring(0, 500) : errMsg);
            predictionRepo.save(prediction);
            return Map.of("error", errMsg != null ? errMsg : "Trade execution failed");
        }
    }

    /**
     * Reject a pending prediction (manual mode).
     */
    public Prediction rejectPrediction(Long predictionId) {
        Prediction prediction = predictionRepo.findById(predictionId)
                .orElseThrow(() -> new RuntimeException("Prediction not found: " + predictionId));

        prediction.setTradeStatus("REJECTED");
        prediction = predictionRepo.save(prediction);

        messagingTemplate.convertAndSend("/topic/predictions", prediction);
        return prediction;
    }

    // ── Trade History ─────────────────────────────

    public List<Trade> getRecentTrades() {
        return tradeRepo.findTop50ByOrderByExecutedAtDesc();
    }

    public List<Trade> getOpenTrades() {
        return tradeRepo.findByStatusOrderByExecutedAtDesc("OPEN");
    }

    public Map<String, Object> getTradeStats() {
        long wins = tradeRepo.countWins();
        long losses = tradeRepo.countLosses();
        double totalPnl = tradeRepo.totalPnl();
        long totalTrades = wins + losses;
        double winRate = totalTrades > 0 ? (double) wins / totalTrades * 100 : 0;

        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalTrades", totalTrades);
        stats.put("wins", wins);
        stats.put("losses", losses);
        stats.put("winRate", Math.round(winRate * 10.0) / 10.0);
        stats.put("totalPnl", Math.round(totalPnl * 100.0) / 100.0);
        return stats;
    }

    // ── Predictions History ───────────────────────

    public List<Prediction> getRecentPredictions() {
        return predictionRepo.findTop50ByOrderByTimestampDesc();
    }

    public Prediction getLatestPrediction() {
        List<Prediction> predictions = predictionRepo.findTop50ByOrderByTimestampDesc();
        return predictions.isEmpty() ? null : predictions.get(0);
    }

    // ── Account Status ────────────────────────────

    public Map<String, Object> getAccountStatus() {
        try {
            return aiEngineClient.getStatus();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    // ── Close Position ────────────────────────────

    public Map<String, Object> closePosition() {
        try {
            return aiEngineClient.closePosition();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    // ── Model Info ────────────────────────────────

    public Map<String, Object> getModelInfo() {
        try {
            return aiEngineClient.getModelInfo();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    // ── Train ────────────────────────────────────

    public Map<String, Object> trainModel() {
        try {
            return aiEngineClient.trainModel();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    // ── AI Engine Health ──────────────────────────

    public Map<String, Object> getAiEngineHealth() {
        return aiEngineClient.healthCheck();
    }

    // ── Helpers ───────────────────────────────────

    @SuppressWarnings("unchecked")
    private Prediction mapToPrediction(Map<String, Object> raw) {
        Prediction p = new Prediction();
        p.setTimestamp(LocalDateTime.now());
        p.setSymbol(getString(raw, "symbol"));
        p.setSignal(getString(raw, "signal"));
        p.setProbabilityUp(getDouble(raw, "probability_up"));
        p.setProbabilityDown(getDouble(raw, "probability_down"));
        p.setConfidencePct(getDouble(raw, "confidence_pct"));
        p.setCurrentPrice(getDouble(raw, "current_price"));
        p.setAtr(getDouble(raw, "atr"));
        p.setAtrPct(getDouble(raw, "atr_pct"));
        p.setSlPrice(getDouble(raw, "sl_price"));
        p.setTpPrice(getDouble(raw, "tp_price"));
        p.setRiskReward(getDouble(raw, "risk_reward"));
        p.setRejectReason(getString(raw, "reject_reason"));
        p.setConclusion(getString(raw, "conclusion"));

        // Position sizing estimates
        p.setEstQtyBtc(getDouble(raw, "est_qty_btc"));
        p.setEstNotionalUsd(getDouble(raw, "est_notional_usd"));
        p.setEstRiskUsd(getDouble(raw, "est_risk_usd"));
        p.setEstCapitalUsedPct(getDouble(raw, "est_capital_used_pct"));
        p.setEstBalance(getDouble(raw, "est_balance"));

        // Market analysis
        Map<String, Object> market = (Map<String, Object>) raw.get("market_analysis");
        if (market != null) {
            p.setTrendDirection(getString(market, "trend_direction"));
            p.setMarketRegime(getString(market, "market_regime"));

            Map<String, Object> indicators = (Map<String, Object>) market.get("indicators");
            if (indicators != null) {
                p.setRsi(getDouble(indicators, "rsi"));
                p.setRsiZone(getString(indicators, "rsi_zone"));
                p.setAdx(getDouble(indicators, "adx"));
                p.setAdxInterpretation(getString(indicators, "adx_interpretation"));
                p.setMacdSignal(getString(indicators, "macd_signal"));
                p.setIsSqueeze((Boolean) indicators.get("is_squeeze"));
                p.setVolumeRatio(getDouble(indicators, "volume_ratio"));
            }
        }

        // HTF alignment
        Map<String, Object> htf = (Map<String, Object>) raw.get("htf_alignment");
        if (htf != null) {
            p.setHtfOverallAlignment(getString(htf, "overall"));
            Map<String, Object> h4 = (Map<String, Object>) htf.get("4h");
            if (h4 != null) p.setHtf4hBias(getString(h4, "bias"));
            Map<String, Object> d1 = (Map<String, Object>) htf.get("1d");
            if (d1 != null) p.setHtf1dBias(getString(d1, "bias"));
        }

        // Store raw JSON
        try {
            p.setRawResponse(objectMapper.writeValueAsString(raw));
        } catch (Exception e) {
            p.setRawResponse("{}");
        }

        return p;
    }

    private Trade createTradeFromPrediction(Prediction prediction, Map<String, Object> tradeResult) {
        Trade trade = new Trade();
        trade.setPrediction(prediction);
        trade.setExecutedAt(LocalDateTime.now());
        trade.setSymbol(prediction.getSymbol());
        trade.setSide(prediction.getSignal());
        trade.setEntryPrice(prediction.getCurrentPrice());
        trade.setSlPrice(prediction.getSlPrice());
        trade.setTpPrice(prediction.getTpPrice());
        trade.setRiskReward(prediction.getRiskReward());
        trade.setMode(prediction.getTradeMode());
        trade.setStatus("OPEN");

        // Extract quantity from trade result
        @SuppressWarnings("unchecked")
        Map<String, Object> sizing = (Map<String, Object>) tradeResult.get("sizing");
        if (sizing != null) {
            trade.setQuantity(getDouble(sizing, "qty_btc"));
        }

        try {
            trade.setOrderDetails(objectMapper.writeValueAsString(tradeResult));
        } catch (Exception e) {
            trade.setOrderDetails("{}");
        }

        return trade;
    }

    private String getString(Map<String, Object> map, String key) {
        Object val = map.get(key);
        return val != null ? val.toString() : null;
    }

    private Double getDouble(Map<String, Object> map, String key) {
        Object val = map.get(key);
        if (val instanceof Number) return ((Number) val).doubleValue();
        if (val instanceof String) {
            try { return Double.parseDouble((String) val); } catch (NumberFormatException e) { return null; }
        }
        return null;
    }
}
