package com.sorbot.backend.repository;

import com.sorbot.backend.model.Trade;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface TradeRepository extends JpaRepository<Trade, Long> {
    List<Trade> findTop50ByOrderByExecutedAtDesc();
    List<Trade> findTop50ByUserIdOrderByExecutedAtDesc(Long userId);
    List<Trade> findByStatusOrderByExecutedAtDesc(String status);
    List<Trade> findByUserIdAndStatusOrderByExecutedAtDesc(Long userId, String status);

    Trade findFirstByUserIdAndStatusOrderByExecutedAtDesc(Long userId, String status);
    Optional<Trade> findByIdAndUserId(Long id, Long userId);
    void deleteByUserId(Long userId);

    @Query("SELECT COUNT(t) FROM Trade t WHERE t.status = 'CLOSED' AND t.pnl > 0")
    long countWins();

    @Query("SELECT COUNT(t) FROM Trade t WHERE t.user.id = :userId AND t.status = 'CLOSED' AND t.pnl > 0")
    long countWinsByUserId(@Param("userId") Long userId);

    @Query("SELECT COUNT(t) FROM Trade t WHERE t.status = 'CLOSED' AND t.pnl <= 0")
    long countLosses();

    @Query("SELECT COUNT(t) FROM Trade t WHERE t.user.id = :userId AND t.status = 'CLOSED' AND t.pnl <= 0")
    long countLossesByUserId(@Param("userId") Long userId);

    @Query("SELECT COALESCE(SUM(t.pnl), 0) FROM Trade t WHERE t.status = 'CLOSED'")
    double totalPnl();

    @Query("SELECT COALESCE(SUM(t.pnl), 0) FROM Trade t WHERE t.user.id = :userId AND t.status = 'CLOSED'")
    double totalPnlByUserId(@Param("userId") Long userId);
}
