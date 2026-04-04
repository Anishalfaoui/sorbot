package com.sorbot.backend.dto;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class HistoricalTradeImportRequest {
    private Boolean clearExistingTrades = false;
    private Double finalBalance;
    private List<HistoricalTradeRowDto> trades = new ArrayList<>();
}
