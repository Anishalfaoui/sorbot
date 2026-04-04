package com.sorbot.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class HistoricalTradeRowDto {
    private String day;

    @JsonProperty("slot_time")
    private String slotTime;

    private String symbol;
    private String side;

    @JsonProperty("confidence_pct")
    private Double confidencePct;

    @JsonProperty("reject_reason")
    private String rejectReason;

    @JsonProperty("entry_time")
    private String entryTime;

    @JsonProperty("entry_price")
    private Double entryPrice;

    @JsonProperty("sl_price")
    private Double slPrice;

    @JsonProperty("tp_price")
    private Double tpPrice;

    private Double qty;

    @JsonProperty("notional_usd")
    private Double notionalUsd;

    @JsonProperty("exit_time")
    private String exitTime;

    @JsonProperty("exit_price")
    private Double exitPrice;

    private String outcome;

    @JsonProperty("pnl_usd")
    private Double pnlUsd;

    @JsonProperty("balance_after")
    private Double balanceAfter;
}
