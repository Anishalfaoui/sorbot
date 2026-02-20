package com.sorbot.backend.scheduler;

import com.sorbot.backend.model.Prediction;
import com.sorbot.backend.service.TradingService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/**
 * Periodically polls the AI engine for new predictions.
 * In AUTO mode, trades are executed automatically.
 * In MANUAL mode, predictions are stored for user review.
 */
@Component
public class PredictionScheduler {

    private static final Logger log = LoggerFactory.getLogger(PredictionScheduler.class);
    private final TradingService tradingService;

    public PredictionScheduler(TradingService tradingService) {
        this.tradingService = tradingService;
    }

    /**
     * Poll every 60 seconds (configurable via properties).
     * Fetches prediction, stores it, and auto-trades if in AUTO mode.
     */
    @Scheduled(fixedDelayString = "${trading.poll-interval-ms:60000}")
    public void pollPrediction() {
        try {
            log.debug("Polling AI engine for prediction...");
            Prediction prediction = tradingService.fetchNewPrediction();
            log.info("Prediction: signal={}, confidence={}%, price={}, status={}",
                    prediction.getSignal(),
                    prediction.getConfidencePct(),
                    prediction.getCurrentPrice(),
                    prediction.getTradeStatus());
        } catch (Exception e) {
            log.error("Prediction poll failed: {}", e.getMessage());
        }
    }
}
