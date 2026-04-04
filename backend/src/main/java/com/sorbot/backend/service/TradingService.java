package com.sorbot.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sorbot.backend.dto.HistoricalTradeImportRequest;
import com.sorbot.backend.dto.HistoricalTradeRowDto;
import com.sorbot.backend.model.Prediction;
import com.sorbot.backend.model.Trade;
import com.sorbot.backend.model.TradingSettings;
import com.sorbot.backend.model.User;
import com.sorbot.backend.repository.PredictionRepository;
import com.sorbot.backend.repository.TradeRepository;
import com.sorbot.backend.repository.TradingSettingsRepository;
import com.sorbot.backend.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.*;

@Service
public class TradingService {

    private static final Logger log = LoggerFactory.getLogger(TradingService.class);
    private static final String DEFAULT_SYMBOL = "BTCUSD";

    private final AiEngineClient aiEngineClient;
    private final PredictionRepository predictionRepo;
    private final TradeRepository tradeRepo;
    private final TradingSettingsRepository settingsRepo;
    private final UserRepository userRepo;
    private final SimpMessagingTemplate messagingTemplate;
    private final ObjectMapper objectMapper;

    public TradingService(
            AiEngineClient aiEngineClient,
            PredictionRepository predictionRepo,
            TradeRepository tradeRepo,
            TradingSettingsRepository settingsRepo,
            UserRepository userRepo,
            SimpMessagingTemplate messagingTemplate,
            ObjectMapper objectMapper
    ) {
        this.aiEngineClient = aiEngineClient;
        this.predictionRepo = predictionRepo;
        this.tradeRepo = tradeRepo;
        this.settingsRepo = settingsRepo;
        this.userRepo = userRepo;
        this.messagingTemplate = messagingTemplate;
        this.objectMapper = objectMapper;
    }

    public TradingSettings getSettings() {
        return settingsRepo.findAll().stream().findFirst()
                .orElseGet(() -> settingsRepo.save(new TradingSettings()));
    }

    public TradingSettings updateMode(String mode) {
        TradingSettings settings = getSettings();
        settings.setMode(mode.toUpperCase());
        settings = settingsRepo.save(settings);
        messagingTemplate.convertAndSend("/topic/settings", settings);
        return settings;
    }

    public boolean isAutoMode() {
        return "AUTO".equalsIgnoreCase(getSettings().getMode());
    }

    public Prediction fetchNewPrediction() {
        return fetchNewPrediction(DEFAULT_SYMBOL);
    }

    public Prediction fetchNewPrediction(String symbol) {
        User user = getCurrentUserOrNull();
        double virtualBalance = user != null && user.getVirtualBalance() != null ? user.getVirtualBalance() : 10000.0;
        String normalizedSymbol = normalizeSymbol(symbol);

        Map<String, Object> raw = aiEngineClient.getPrediction(normalizedSymbol, virtualBalance);
        Prediction prediction = mapToPrediction(raw);
        prediction.setUser(user);
        prediction.setTradeMode(getSettings().getMode());

        TradingSettings settings = getSettings();
        if ("AUTO".equalsIgnoreCase(settings.getMode())
                && user != null
                && !"NO_TRADE".equals(prediction.getSignal())) {
            prediction.setTradeStatus("AUTO_EXECUTING");
            prediction = predictionRepo.save(prediction);

            try {
                Map<String, Object> tradeResult = aiEngineClient.executeTrade(
                        prediction.getSignal(),
                        prediction.getCurrentPrice(),
                        prediction.getSlPrice(),
                        prediction.getTpPrice(),
                        prediction.getEstQtyBtc(),
                        normalizedSymbol,
                        virtualBalance
                );
                String action = (String) tradeResult.getOrDefault("action", "UNKNOWN");

                if ("TRADE_EXECUTED".equals(action)) {
                    prediction.setTradeStatus("AUTO_EXECUTED");
                    prediction.setTradeExecutedAt(LocalDateTime.now());
                    Trade trade = createTradeFromPrediction(prediction, tradeResult, user);
                    tradeRepo.save(trade);
                } else {
                    prediction.setTradeStatus("AUTO_SKIPPED");
                    prediction.setRejectReason((String) tradeResult.getOrDefault("reason",
                            tradeResult.getOrDefault("error", action)));
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

        messagingTemplate.convertAndSend("/topic/predictions", prediction);
        return prediction;
    }

    public Map<String, Object> acceptPrediction(Long predictionId) {
        User user = getCurrentUser();

        Prediction prediction = predictionRepo.findById(predictionId)
                .orElseThrow(() -> new RuntimeException("Prediction not found: " + predictionId));

        if (prediction.getUser() != null && !prediction.getUser().getId().equals(user.getId())) {
            return Map.of("error", "Prediction does not belong to the current user.");
        }

        if (!"PENDING".equals(prediction.getTradeStatus())) {
            return Map.of("error", "Prediction is not pending. Status: " + prediction.getTradeStatus());
        }

        double balance = user.getVirtualBalance() != null ? user.getVirtualBalance() : 10000.0;
        String symbol = normalizeSymbol(prediction.getSymbol());

        try {
            Map<String, Object> tradeResult = aiEngineClient.executeTrade(
                    prediction.getSignal(),
                    prediction.getCurrentPrice(),
                    prediction.getSlPrice(),
                    prediction.getTpPrice(),
                    prediction.getEstQtyBtc(),
                    symbol,
                    balance
            );

            String action = (String) tradeResult.getOrDefault("action", "UNKNOWN");
            if ("TRADE_EXECUTED".equals(action)) {
                prediction.setTradeStatus("ACCEPTED");
                prediction.setTradeExecutedAt(LocalDateTime.now());
                prediction.setUser(user);

                Trade trade = createTradeFromPrediction(prediction, tradeResult, user);
                tradeRepo.save(trade);
            } else if ("NO_TRADE".equals(action) || "BLOCKED".equals(action) || "ERROR".equals(action)) {
                String reason = (String) tradeResult.getOrDefault("reason",
                        tradeResult.getOrDefault("error", action));
                prediction.setTradeStatus("INSUFFICIENT_FUNDS");
                prediction.setRejectReason(reason);
            } else {
                prediction.setTradeStatus("EXECUTION_FAILED");
                String reason = (String) tradeResult.getOrDefault("reason",
                        tradeResult.getOrDefault("error", action));
                prediction.setRejectReason(reason);
            }

            predictionRepo.save(prediction);
            messagingTemplate.convertAndSend("/topic/predictions", prediction);
            messagingTemplate.convertAndSend("/topic/trades", getRecentTrades());

            return Map.of(
                    "status", "ok",
                    "predictionId", prediction.getId(),
                    "tradeStatus", prediction.getTradeStatus(),
                    "tradeResult", tradeResult
            );
        } catch (Exception e) {
            prediction.setTradeStatus("EXECUTION_FAILED");
            prediction.setRejectReason(trimError(e.getMessage()));
            predictionRepo.save(prediction);
            return Map.of("error", e.getMessage() != null ? e.getMessage() : "Trade execution failed");
        }
    }

    public Prediction rejectPrediction(Long predictionId) {
        User user = getCurrentUser();
        Prediction prediction = predictionRepo.findById(predictionId)
                .orElseThrow(() -> new RuntimeException("Prediction not found: " + predictionId));

        if (prediction.getUser() != null && !prediction.getUser().getId().equals(user.getId())) {
            throw new RuntimeException("Prediction does not belong to the current user.");
        }

        prediction.setTradeStatus("REJECTED");
        prediction = predictionRepo.save(prediction);
        messagingTemplate.convertAndSend("/topic/predictions", prediction);
        return prediction;
    }

    public List<Trade> getRecentTrades() {
        User user = getCurrentUserOrNull();
        if (user == null) return tradeRepo.findTop50ByOrderByExecutedAtDesc();
        return tradeRepo.findTop50ByUserIdOrderByExecutedAtDesc(user.getId());
    }

    public List<Trade> getOpenTrades() {
        User user = getCurrentUserOrNull();
        if (user == null) return tradeRepo.findByStatusOrderByExecutedAtDesc("OPEN");
        return tradeRepo.findByUserIdAndStatusOrderByExecutedAtDesc(user.getId(), "OPEN");
    }

    public Map<String, Object> getTradeStats() {
        User user = getCurrentUserOrNull();

        long wins;
        long losses;
        double totalPnl;

        if (user == null) {
            wins = tradeRepo.countWins();
            losses = tradeRepo.countLosses();
            totalPnl = tradeRepo.totalPnl();
        } else {
            wins = tradeRepo.countWinsByUserId(user.getId());
            losses = tradeRepo.countLossesByUserId(user.getId());
            totalPnl = tradeRepo.totalPnlByUserId(user.getId());
        }

        long totalTrades = wins + losses;
        double winRate = totalTrades > 0 ? (double) wins / totalTrades * 100 : 0;

        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("totalTrades", totalTrades);
        stats.put("wins", wins);
        stats.put("losses", losses);
        stats.put("winRate", Math.round(winRate * 10.0) / 10.0);
        stats.put("totalPnl", Math.round(totalPnl * 100.0) / 100.0);
        if (user != null) {
            stats.put("virtualBalance", round2(user.getVirtualBalance()));
        }
        return stats;
    }

    public List<Prediction> getRecentPredictions() {
        User user = getCurrentUserOrNull();
        if (user == null) return predictionRepo.findTop50ByOrderByTimestampDesc();
        return predictionRepo.findTop50ByUserIdOrderByTimestampDesc(user.getId());
    }

    public Prediction getLatestPrediction() {
        List<Prediction> predictions = getRecentPredictions();
        return predictions.isEmpty() ? null : predictions.get(0);
    }

    public Map<String, Object> getAccountStatus() {
        User user = getCurrentUser();

        String symbol = DEFAULT_SYMBOL;
        Trade open = tradeRepo.findFirstByUserIdAndStatusOrderByExecutedAtDesc(user.getId(), "OPEN");
        if (open != null && open.getSymbol() != null) {
            symbol = normalizeSymbol(open.getSymbol());
        }

        Map<String, Object> status = new LinkedHashMap<>();
        status.put("symbol", symbol);
        status.put("virtualAccount", true);
        status.put("virtualBalance", round2(user.getVirtualBalance()));
        status.put("openPositions", open != null ? 1 : 0);

        if (open != null) {
            try {
                double currentPrice = aiEngineClient.getPrice(symbol);
                double pnl = calculatePnl(open, currentPrice);
                status.put("currentPrice", currentPrice);
                status.put("openTrade", open);
                status.put("openTradeUnrealizedPnl", round2(pnl));
            } catch (Exception e) {
                status.put("priceError", e.getMessage());
                status.put("openTrade", open);
            }
        }

        return status;
    }

    public Map<String, Object> closePosition() {
        User user = getCurrentUser();
        Trade open = tradeRepo.findFirstByUserIdAndStatusOrderByExecutedAtDesc(user.getId(), "OPEN");
        if (open == null) {
            return Map.of("action", "NO_POSITION", "message", "No open virtual position");
        }

        return closePosition(open.getId());
    }

    public Map<String, Object> closePosition(Long tradeId) {
        User user = getCurrentUser();

        Trade open = tradeRepo.findByIdAndUserId(tradeId, user.getId())
                .orElseThrow(() -> new RuntimeException("Trade not found for current user: " + tradeId));

        if (!"OPEN".equalsIgnoreCase(open.getStatus())) {
            return Map.of(
                    "action", "NO_POSITION",
                    "message", "Trade is not open",
                    "tradeId", open.getId(),
                    "status", open.getStatus()
            );
        }

        String symbol = normalizeSymbol(open.getSymbol());
        double currentPrice = aiEngineClient.getPrice(symbol);
        double pnl = calculatePnl(open, currentPrice);

        open.setExitPrice(currentPrice);
        open.setPnl(round2(pnl));
        open.setPnlPct(open.getEntryPrice() != null && open.getEntryPrice() > 0
                ? round2((pnl / (open.getEntryPrice() * Math.max(open.getQuantity(), 0.00001))) * 100)
                : 0.0);
        open.setClosedAt(LocalDateTime.now());
        open.setCloseReason("MANUAL_CLOSE");
        open.setStatus("CLOSED");
        tradeRepo.save(open);

        double newBalance = (user.getVirtualBalance() != null ? user.getVirtualBalance() : 10000.0) + pnl;
        user.setVirtualBalance(round2(newBalance));
        userRepo.save(user);

        messagingTemplate.convertAndSend("/topic/trades", getRecentTrades());

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("action", "CLOSED");
        result.put("symbol", symbol);
        result.put("exitPrice", currentPrice);
        result.put("pnl", round2(pnl));
        result.put("newVirtualBalance", round2(newBalance));
        result.put("tradeId", open.getId());
        return result;
    }

    public Map<String, Object> getModelInfo() {
        try {
            return aiEngineClient.getModelInfo(DEFAULT_SYMBOL);
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    public Map<String, Object> importHistoricalTrades(HistoricalTradeImportRequest request) {
        User user = getCurrentUser();

        if (request == null || request.getTrades() == null || request.getTrades().isEmpty()) {
            return Map.of("error", "No trades provided for import.");
        }

        boolean clearExisting = Boolean.TRUE.equals(request.getClearExistingTrades());
        if (clearExisting) {
            tradeRepo.deleteByUserId(user.getId());
        }

        List<HistoricalTradeRowDto> rows = new ArrayList<>(request.getTrades());
        rows.sort(Comparator.comparing(r -> {
            LocalDateTime t = parseDateTimeSafe(r.getEntryTime());
            return t != null ? t : LocalDateTime.MIN;
        }));

        List<Trade> tradesToSave = new ArrayList<>();
        double totalPnl = 0.0;
        long wins = 0;
        long losses = 0;
        Double lastBalanceAfter = null;

        for (HistoricalTradeRowDto row : rows) {
            if (row == null) continue;

            Trade trade = new Trade();
            trade.setUser(user);
            trade.setPrediction(null);

            LocalDateTime executedAt = parseDateTimeSafe(row.getEntryTime());
            LocalDateTime closedAt = parseDateTimeSafe(row.getExitTime());
            if (executedAt == null) {
                executedAt = LocalDateTime.now();
            }
            if (closedAt == null || closedAt.isBefore(executedAt)) {
                closedAt = executedAt;
            }

            trade.setExecutedAt(executedAt);
            trade.setClosedAt(closedAt);
            trade.setSymbol(normalizeSymbol(row.getSymbol()));

            String side = row.getSide() != null ? row.getSide().toUpperCase(Locale.ROOT) : "LONG";
            trade.setSide("SHORT".equals(side) ? "SHORT" : "LONG");

            double entry = nz(row.getEntryPrice());
            double exit = nz(row.getExitPrice());
            double qty = nz(row.getQty());
            if (qty <= 0 && entry > 0 && row.getNotionalUsd() != null) {
                qty = Math.abs(row.getNotionalUsd() / entry);
            }
            qty = Math.max(qty, 0.0);

            trade.setEntryPrice(entry > 0 ? entry : null);
            trade.setSlPrice(row.getSlPrice());
            trade.setTpPrice(row.getTpPrice());
            trade.setQuantity(qty > 0 ? qty : null);
            trade.setExitPrice(exit > 0 ? exit : null);

            Double rr = computeRiskReward(entry, row.getSlPrice(), row.getTpPrice());
            trade.setRiskReward(rr);

            double pnl = row.getPnlUsd() != null ? row.getPnlUsd() : computePnlFromPrices(side, entry, exit, qty);
            trade.setPnl(round2(pnl));

            double denom = entry > 0 && qty > 0 ? (entry * qty) : 0.0;
            double pnlPct = denom > 0 ? (pnl / denom) * 100.0 : 0.0;
            trade.setPnlPct(round2(pnlPct));

            trade.setStatus("CLOSED");
            trade.setCloseReason(mapOutcomeToCloseReason(row.getOutcome()));
            trade.setMode("HISTORICAL_IMPORT");
            trade.setErrorMessage(null);

            Map<String, Object> details = new LinkedHashMap<>();
            details.put("source", "historical_simulation");
            details.put("day", row.getDay());
            details.put("slot_time", row.getSlotTime());
            details.put("confidence_pct", row.getConfidencePct());
            details.put("reject_reason", row.getRejectReason());
            details.put("notional_usd", row.getNotionalUsd());
            details.put("outcome", row.getOutcome());
            try {
                trade.setOrderDetails(objectMapper.writeValueAsString(details));
            } catch (Exception e) {
                trade.setOrderDetails("{}");
            }

            tradesToSave.add(trade);
            totalPnl += pnl;
            if (pnl > 0) wins++; else losses++;
            if (row.getBalanceAfter() != null) {
                lastBalanceAfter = row.getBalanceAfter();
            }
        }

        if (tradesToSave.isEmpty()) {
            return Map.of("error", "No valid trades to import.");
        }

        tradeRepo.saveAll(tradesToSave);

        double currentBalance = user.getVirtualBalance() != null ? user.getVirtualBalance() : 10000.0;
        double finalBalance;
        if (request.getFinalBalance() != null) {
            finalBalance = request.getFinalBalance();
        } else if (lastBalanceAfter != null) {
            finalBalance = lastBalanceAfter;
        } else {
            finalBalance = currentBalance + totalPnl;
        }

        user.setVirtualBalance(round2(finalBalance));
        userRepo.save(user);

        messagingTemplate.convertAndSend("/topic/trades", getRecentTrades());

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("status", "ok");
        result.put("importedTrades", tradesToSave.size());
        result.put("wins", wins);
        result.put("losses", losses);
        result.put("totalPnl", round2(totalPnl));
        result.put("clearExistingTrades", clearExisting);
        result.put("newVirtualBalance", round2(finalBalance));
        return result;
    }

    public Map<String, Object> getModelInfoAll() {
        try {
            return aiEngineClient.getModelInfoAll();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    public Map<String, Object> trainModel() {
        try {
            return aiEngineClient.trainModel();
        } catch (Exception e) {
            return Map.of("error", e.getMessage());
        }
    }

    public Map<String, Object> getAiEngineHealth() {
        return aiEngineClient.healthCheck(DEFAULT_SYMBOL);
    }

    @SuppressWarnings("unchecked")
    private Prediction mapToPrediction(Map<String, Object> raw) {
        Prediction p = new Prediction();
        p.setTimestamp(LocalDateTime.now());
        p.setSymbol(normalizeSymbol(getString(raw, "symbol")));
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

        p.setEstQtyBtc(getDouble(raw, "est_qty_btc"));
        p.setEstNotionalUsd(getDouble(raw, "est_notional_usd"));
        p.setEstRiskUsd(getDouble(raw, "est_risk_usd"));
        p.setEstCapitalUsedPct(getDouble(raw, "est_capital_used_pct"));
        p.setEstBalance(getDouble(raw, "est_balance"));

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

        Map<String, Object> htf = (Map<String, Object>) raw.get("htf_alignment");
        if (htf != null) {
            p.setHtfOverallAlignment(getString(htf, "overall"));
            Map<String, Object> h4 = (Map<String, Object>) htf.get("4h");
            if (h4 != null) p.setHtf4hBias(getString(h4, "bias"));
            Map<String, Object> d1 = (Map<String, Object>) htf.get("1d");
            if (d1 != null) p.setHtf1dBias(getString(d1, "bias"));
        }

        try {
            p.setRawResponse(objectMapper.writeValueAsString(raw));
        } catch (Exception e) {
            p.setRawResponse("{}");
        }

        return p;
    }

    private Trade createTradeFromPrediction(Prediction prediction, Map<String, Object> tradeResult, User user) {
        Trade trade = new Trade();
        trade.setUser(user);
        trade.setPrediction(prediction);
        trade.setExecutedAt(LocalDateTime.now());
        trade.setSymbol(normalizeSymbol(prediction.getSymbol()));
        trade.setSide(prediction.getSignal());
        trade.setEntryPrice(prediction.getCurrentPrice());
        trade.setSlPrice(prediction.getSlPrice());
        trade.setTpPrice(prediction.getTpPrice());
        trade.setRiskReward(prediction.getRiskReward());
        trade.setMode(prediction.getTradeMode());
        trade.setStatus("OPEN");

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
            try {
                return Double.parseDouble((String) val);
            } catch (NumberFormatException e) {
                return null;
            }
        }
        return null;
    }

    private String trimError(String errMsg) {
        if (errMsg == null) return null;
        return errMsg.length() > 500 ? errMsg.substring(0, 500) : errMsg;
    }

    private String normalizeSymbol(String symbol) {
        if (symbol == null || symbol.isBlank()) return DEFAULT_SYMBOL;
        String s = symbol.toUpperCase().replace("/", "").replace("-", "").replace(" ", "");
        if ("BTCUSD".equals(s) || "BTCUSDT".equals(s)) return "BTCUSD";
        if ("EURUSD".equals(s)) return "EURUSD";
        if ("XAUUSD".equals(s)) return "XAUUSD";
        return DEFAULT_SYMBOL;
    }

    private LocalDateTime parseDateTimeSafe(String raw) {
        if (raw == null || raw.isBlank()) return null;
        String value = raw.trim();

        try {
            return LocalDateTime.parse(value, DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        } catch (DateTimeParseException ignored) {
        }

        String normalized = value.replace(" ", "T");
        try {
            return LocalDateTime.parse(normalized, DateTimeFormatter.ISO_LOCAL_DATE_TIME);
        } catch (DateTimeParseException ignored) {
        }

        try {
            return LocalDateTime.parse(value, DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        } catch (DateTimeParseException ignored) {
        }

        try {
            return LocalDateTime.parse(value, DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm"));
        } catch (DateTimeParseException ignored) {
        }

        return null;
    }

    private String mapOutcomeToCloseReason(String outcome) {
        if (outcome == null) return "HISTORICAL_IMPORT";
        String o = outcome.toUpperCase(Locale.ROOT);
        if ("TP".equals(o)) return "TP_HIT";
        if ("SL".equals(o)) return "SL_HIT";
        if ("EXPIRED".equals(o)) return "TIME_EXIT";
        return "HISTORICAL_IMPORT";
    }

    private Double computeRiskReward(double entry, Double sl, Double tp) {
        if (entry <= 0 || sl == null || tp == null) return null;
        double risk = Math.abs(entry - sl);
        if (risk <= 0) return null;
        double reward = Math.abs(tp - entry);
        return round2(reward / risk);
    }

    private double computePnlFromPrices(String side, double entry, double exit, double qty) {
        if (entry <= 0 || exit <= 0 || qty <= 0) return 0.0;
        if ("SHORT".equalsIgnoreCase(side)) {
            return (entry - exit) * qty;
        }
        return (exit - entry) * qty;
    }

    private double nz(Double value) {
        return value != null ? value : 0.0;
    }

    private double calculatePnl(Trade trade, double currentPrice) {
        double qty = trade.getQuantity() != null ? trade.getQuantity() : 0.0;
        if (qty <= 0) return 0.0;
        double entry = trade.getEntryPrice() != null ? trade.getEntryPrice() : currentPrice;
        if ("SHORT".equalsIgnoreCase(trade.getSide())) {
            return (entry - currentPrice) * qty;
        }
        return (currentPrice - entry) * qty;
    }

    private double round2(Double n) {
        if (n == null) return 0.0;
        return Math.round(n * 100.0) / 100.0;
    }

    private User getCurrentUser() {
        User user = getCurrentUserOrNull();
        if (user == null) {
            throw new RuntimeException("User context is required for this operation.");
        }
        return user;
    }

    private User getCurrentUserOrNull() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || !auth.isAuthenticated() || auth instanceof AnonymousAuthenticationToken) {
            return null;
        }
        String username = auth.getName();
        return userRepo.findByUsername(username).orElse(null);
    }
}
