package com.sorbot.backend.repository;

import com.sorbot.backend.model.TradingSettings;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface TradingSettingsRepository extends JpaRepository<TradingSettings, Long> {
}
