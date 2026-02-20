package com.sorbot.backend.controller;

import com.sorbot.backend.model.TradingSettings;
import com.sorbot.backend.service.TradingService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/settings")
public class SettingsController {

    private final TradingService tradingService;

    public SettingsController(TradingService tradingService) {
        this.tradingService = tradingService;
    }

    /**
     * GET /api/settings — Get current trading settings.
     */
    @GetMapping
    public TradingSettings getSettings() {
        return tradingService.getSettings();
    }

    /**
     * PUT /api/settings/mode — Switch between AUTO and MANUAL mode.
     * Body: { "mode": "AUTO" } or { "mode": "MANUAL" }
     */
    @PutMapping("/mode")
    public TradingSettings updateMode(@RequestBody Map<String, String> body) {
        String mode = body.getOrDefault("mode", "MANUAL");
        return tradingService.updateMode(mode);
    }
}
