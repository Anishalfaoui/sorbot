package com.sorbot.backend.controller;

import com.sorbot.backend.service.TradingService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class DashboardController {

    private final TradingService tradingService;

    public DashboardController(TradingService tradingService) {
        this.tradingService = tradingService;
    }

    /**
     * GET /api/dashboard — Aggregate dashboard data.
     */
    @GetMapping("/dashboard")
    public Map<String, Object> getDashboard() {
        return Map.of(
                "settings", tradingService.getSettings(),
                "latestPrediction", tradingService.getLatestPrediction() != null
                        ? tradingService.getLatestPrediction() : Map.of(),
                "tradeStats", tradingService.getTradeStats(),
                "aiEngineHealth", tradingService.getAiEngineHealth()
        );
    }

    /**
     * GET /api/account — Get account/position status from Binance.
     */
    @GetMapping("/account")
    public Map<String, Object> getAccountStatus() {
        return tradingService.getAccountStatus();
    }

    /**
     * GET /api/model — Get AI model info.
     */
    @GetMapping("/model")
    public Map<String, Object> getModelInfo() {
        return tradingService.getModelInfo();
    }

    /**
     * POST /api/train — Trigger model retraining.
     */
    @PostMapping("/train")
    public Map<String, Object> trainModel() {
        return tradingService.trainModel();
    }

    /**
     * GET /api/health — Backend health check.
     */
    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of(
                "status", "running",
                "backend", "Sorbot Spring Boot v1.0",
                "aiEngine", tradingService.getAiEngineHealth()
        );
    }
}
