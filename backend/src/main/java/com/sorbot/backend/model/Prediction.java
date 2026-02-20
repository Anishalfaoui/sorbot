package com.sorbot.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "predictions")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private LocalDateTime timestamp;
    private String symbol;
    private String signal;          // LONG, SHORT, NO_TRADE
    private Double probabilityUp;
    private Double probabilityDown;
    private Double confidencePct;
    private Double currentPrice;
    private Double atr;
    private Double atrPct;
    private Double slPrice;
    private Double tpPrice;
    private Double riskReward;
    private String rejectReason;

    // Market analysis fields
    private String trendDirection;
    private String marketRegime;
    private Double rsi;
    private String rsiZone;
    private Double adx;
    private String adxInterpretation;
    private String macdSignal;
    private Boolean isSqueeze;
    private Double volumeRatio;

    // HTF alignment
    private String htfOverallAlignment;
    private String htf4hBias;
    private String htf1dBias;

    // Conclusion
    @Column(columnDefinition = "TEXT")
    private String conclusion;

    // Full JSON response from AI engine
    @Column(columnDefinition = "TEXT")
    private String rawResponse;

    // Trade tracking
    private String tradeStatus;     // PENDING, ACCEPTED, REJECTED, AUTO_EXECUTED, SKIPPED
    private LocalDateTime tradeExecutedAt;
    private String tradeMode;       // AUTO, MANUAL

    // Constructors
    public Prediction() {}

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }

    public String getSignal() { return signal; }
    public void setSignal(String signal) { this.signal = signal; }

    public Double getProbabilityUp() { return probabilityUp; }
    public void setProbabilityUp(Double probabilityUp) { this.probabilityUp = probabilityUp; }

    public Double getProbabilityDown() { return probabilityDown; }
    public void setProbabilityDown(Double probabilityDown) { this.probabilityDown = probabilityDown; }

    public Double getConfidencePct() { return confidencePct; }
    public void setConfidencePct(Double confidencePct) { this.confidencePct = confidencePct; }

    public Double getCurrentPrice() { return currentPrice; }
    public void setCurrentPrice(Double currentPrice) { this.currentPrice = currentPrice; }

    public Double getAtr() { return atr; }
    public void setAtr(Double atr) { this.atr = atr; }

    public Double getAtrPct() { return atrPct; }
    public void setAtrPct(Double atrPct) { this.atrPct = atrPct; }

    public Double getSlPrice() { return slPrice; }
    public void setSlPrice(Double slPrice) { this.slPrice = slPrice; }

    public Double getTpPrice() { return tpPrice; }
    public void setTpPrice(Double tpPrice) { this.tpPrice = tpPrice; }

    public Double getRiskReward() { return riskReward; }
    public void setRiskReward(Double riskReward) { this.riskReward = riskReward; }

    public String getRejectReason() { return rejectReason; }
    public void setRejectReason(String rejectReason) { this.rejectReason = rejectReason; }

    public String getTrendDirection() { return trendDirection; }
    public void setTrendDirection(String trendDirection) { this.trendDirection = trendDirection; }

    public String getMarketRegime() { return marketRegime; }
    public void setMarketRegime(String marketRegime) { this.marketRegime = marketRegime; }

    public Double getRsi() { return rsi; }
    public void setRsi(Double rsi) { this.rsi = rsi; }

    public String getRsiZone() { return rsiZone; }
    public void setRsiZone(String rsiZone) { this.rsiZone = rsiZone; }

    public Double getAdx() { return adx; }
    public void setAdx(Double adx) { this.adx = adx; }

    public String getAdxInterpretation() { return adxInterpretation; }
    public void setAdxInterpretation(String adxInterpretation) { this.adxInterpretation = adxInterpretation; }

    public String getMacdSignal() { return macdSignal; }
    public void setMacdSignal(String macdSignal) { this.macdSignal = macdSignal; }

    public Boolean getIsSqueeze() { return isSqueeze; }
    public void setIsSqueeze(Boolean isSqueeze) { this.isSqueeze = isSqueeze; }

    public Double getVolumeRatio() { return volumeRatio; }
    public void setVolumeRatio(Double volumeRatio) { this.volumeRatio = volumeRatio; }

    public String getHtfOverallAlignment() { return htfOverallAlignment; }
    public void setHtfOverallAlignment(String htfOverallAlignment) { this.htfOverallAlignment = htfOverallAlignment; }

    public String getHtf4hBias() { return htf4hBias; }
    public void setHtf4hBias(String htf4hBias) { this.htf4hBias = htf4hBias; }

    public String getHtf1dBias() { return htf1dBias; }
    public void setHtf1dBias(String htf1dBias) { this.htf1dBias = htf1dBias; }

    public String getConclusion() { return conclusion; }
    public void setConclusion(String conclusion) { this.conclusion = conclusion; }

    public String getRawResponse() { return rawResponse; }
    public void setRawResponse(String rawResponse) { this.rawResponse = rawResponse; }

    public String getTradeStatus() { return tradeStatus; }
    public void setTradeStatus(String tradeStatus) { this.tradeStatus = tradeStatus; }

    public LocalDateTime getTradeExecutedAt() { return tradeExecutedAt; }
    public void setTradeExecutedAt(LocalDateTime tradeExecutedAt) { this.tradeExecutedAt = tradeExecutedAt; }

    public String getTradeMode() { return tradeMode; }
    public void setTradeMode(String tradeMode) { this.tradeMode = tradeMode; }
}
