package com.sorbot.backend.repository;

import com.sorbot.backend.model.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    List<Prediction> findTop50ByOrderByTimestampDesc();
    List<Prediction> findByTimestampAfterOrderByTimestampDesc(LocalDateTime after);
    List<Prediction> findByTradeStatusOrderByTimestampDesc(String tradeStatus);
    List<Prediction> findBySignalNotOrderByTimestampDesc(String signal);
}
