package com.sorbot.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "trades")
public class Trade {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "prediction_id")
    private Prediction prediction;

    private LocalDateTime executedAt;
    private String symbol;
    private String side;            // LONG, SHORT
    private Double entryPrice;
    private Double slPrice;
    private Double tpPrice;
    private Double quantity;
    private Double riskReward;
    private String mode;            // AUTO, MANUAL

    private String status;          // OPEN, CLOSED, FAILED, CANCELLED
    private Double exitPrice;
    private Double pnl;
    private Double pnlPct;
    private LocalDateTime closedAt;
    private String closeReason;     // TP_HIT, SL_HIT, MANUAL_CLOSE, ERROR

    @Column(columnDefinition = "TEXT")
    private String orderDetails;    // raw JSON from Binance

    private String errorMessage;

    // Constructors
    public Trade() {}

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Prediction getPrediction() { return prediction; }
    public void setPrediction(Prediction prediction) { this.prediction = prediction; }

    public LocalDateTime getExecutedAt() { return executedAt; }
    public void setExecutedAt(LocalDateTime executedAt) { this.executedAt = executedAt; }

    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }

    public String getSide() { return side; }
    public void setSide(String side) { this.side = side; }

    public Double getEntryPrice() { return entryPrice; }
    public void setEntryPrice(Double entryPrice) { this.entryPrice = entryPrice; }

    public Double getSlPrice() { return slPrice; }
    public void setSlPrice(Double slPrice) { this.slPrice = slPrice; }

    public Double getTpPrice() { return tpPrice; }
    public void setTpPrice(Double tpPrice) { this.tpPrice = tpPrice; }

    public Double getQuantity() { return quantity; }
    public void setQuantity(Double quantity) { this.quantity = quantity; }

    public Double getRiskReward() { return riskReward; }
    public void setRiskReward(Double riskReward) { this.riskReward = riskReward; }

    public String getMode() { return mode; }
    public void setMode(String mode) { this.mode = mode; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public Double getExitPrice() { return exitPrice; }
    public void setExitPrice(Double exitPrice) { this.exitPrice = exitPrice; }

    public Double getPnl() { return pnl; }
    public void setPnl(Double pnl) { this.pnl = pnl; }

    public Double getPnlPct() { return pnlPct; }
    public void setPnlPct(Double pnlPct) { this.pnlPct = pnlPct; }

    public LocalDateTime getClosedAt() { return closedAt; }
    public void setClosedAt(LocalDateTime closedAt) { this.closedAt = closedAt; }

    public String getCloseReason() { return closeReason; }
    public void setCloseReason(String closeReason) { this.closeReason = closeReason; }

    public String getOrderDetails() { return orderDetails; }
    public void setOrderDetails(String orderDetails) { this.orderDetails = orderDetails; }

    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
}
