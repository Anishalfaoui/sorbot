package com.sorbot.backend.controller;

import com.sorbot.backend.model.Prediction;
import com.sorbot.backend.service.TradingService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/predictions")
public class PredictionController {

    private final TradingService tradingService;

    public PredictionController(TradingService tradingService) {
        this.tradingService = tradingService;
    }

    /**
     * GET /api/predictions — Get recent predictions (last 50).
     */
    @GetMapping
    public List<Prediction> getRecentPredictions() {
        return tradingService.getRecentPredictions();
    }

    /**
     * GET /api/predictions/latest — Get the latest prediction.
     */
    @GetMapping("/latest")
    public ResponseEntity<Prediction> getLatest() {
        Prediction latest = tradingService.getLatestPrediction();
        return latest != null ? ResponseEntity.ok(latest) : ResponseEntity.noContent().build();
    }

    /**
     * POST /api/predictions/fetch — Manually trigger a new prediction.
     */
    @PostMapping("/fetch")
    public Prediction fetchNewPrediction() {
        return tradingService.fetchNewPrediction();
    }

    /**
     * POST /api/predictions/{id}/accept — Accept a prediction (manual mode).
     */
    @PostMapping("/{id}/accept")
    public ResponseEntity<Map<String, Object>> acceptPrediction(@PathVariable Long id) {
        Map<String, Object> result = tradingService.acceptPrediction(id);
        if (result.containsKey("error")) {
            return ResponseEntity.badRequest().body(result);
        }
        return ResponseEntity.ok(result);
    }

    /**
     * POST /api/predictions/{id}/reject — Reject a prediction (manual mode).
     */
    @PostMapping("/{id}/reject")
    public Prediction rejectPrediction(@PathVariable Long id) {
        return tradingService.rejectPrediction(id);
    }
}
