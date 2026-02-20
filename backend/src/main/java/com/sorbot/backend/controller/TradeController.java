package com.sorbot.backend.controller;

import com.sorbot.backend.model.Trade;
import com.sorbot.backend.service.TradingService;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/trades")
public class TradeController {

    private final TradingService tradingService;

    public TradeController(TradingService tradingService) {
        this.tradingService = tradingService;
    }

    /**
     * GET /api/trades — Get recent trades.
     */
    @GetMapping
    public List<Trade> getRecentTrades() {
        return tradingService.getRecentTrades();
    }

    /**
     * GET /api/trades/open — Get currently open trades.
     */
    @GetMapping("/open")
    public List<Trade> getOpenTrades() {
        return tradingService.getOpenTrades();
    }

    /**
     * GET /api/trades/stats — Get trade statistics.
     */
    @GetMapping("/stats")
    public Map<String, Object> getTradeStats() {
        return tradingService.getTradeStats();
    }

    /**
     * POST /api/trades/close — Close position via AI engine.
     */
    @PostMapping("/close")
    public Map<String, Object> closePosition() {
        return tradingService.closePosition();
    }
}
