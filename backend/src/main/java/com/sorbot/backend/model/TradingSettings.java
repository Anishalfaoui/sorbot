package com.sorbot.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "trading_settings")
public class TradingSettings {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String mode;            // AUTO, MANUAL
    private Boolean autoTradeEnabled;
    private Integer pollIntervalMs;
    private LocalDateTime updatedAt;

    public TradingSettings() {
        this.mode = "MANUAL";
        this.autoTradeEnabled = false;
        this.pollIntervalMs = 60000;
        this.updatedAt = LocalDateTime.now();
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getMode() { return mode; }
    public void setMode(String mode) {
        this.mode = mode;
        this.autoTradeEnabled = "AUTO".equalsIgnoreCase(mode);
        this.updatedAt = LocalDateTime.now();
    }

    public Boolean getAutoTradeEnabled() { return autoTradeEnabled; }
    public void setAutoTradeEnabled(Boolean autoTradeEnabled) {
        this.autoTradeEnabled = autoTradeEnabled;
    }

    public Integer getPollIntervalMs() { return pollIntervalMs; }
    public void setPollIntervalMs(Integer pollIntervalMs) {
        this.pollIntervalMs = pollIntervalMs;
    }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }
}
