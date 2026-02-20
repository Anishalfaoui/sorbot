package com.sorbot.backend.repository;

import com.sorbot.backend.model.Trade;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TradeRepository extends JpaRepository<Trade, Long> {
    List<Trade> findTop50ByOrderByExecutedAtDesc();
    List<Trade> findByStatusOrderByExecutedAtDesc(String status);

    @Query("SELECT COUNT(t) FROM Trade t WHERE t.status = 'CLOSED' AND t.pnl > 0")
    long countWins();

    @Query("SELECT COUNT(t) FROM Trade t WHERE t.status = 'CLOSED' AND t.pnl <= 0")
    long countLosses();

    @Query("SELECT COALESCE(SUM(t.pnl), 0) FROM Trade t WHERE t.status = 'CLOSED'")
    double totalPnl();
}
