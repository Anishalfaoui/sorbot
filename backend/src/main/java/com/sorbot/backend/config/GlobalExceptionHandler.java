package com.sorbot.backend.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;

import java.time.LocalDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Global exception handler for consistent error responses across all controllers.
 */
@ControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /**
     * Handle RuntimeException — return 404 for "not found" errors, 500 for others.
     */
    @ExceptionHandler(RuntimeException.class)
    public ResponseEntity<Map<String, Object>> handleRuntimeException(RuntimeException ex) {
        String message = ex.getMessage() != null ? ex.getMessage() : "Internal server error";

        HttpStatus status;
        if (message.toLowerCase().contains("not found")) {
            status = HttpStatus.NOT_FOUND;
            log.warn("Resource not found: {}", message);
        } else {
            status = HttpStatus.INTERNAL_SERVER_ERROR;
            log.error("Unhandled runtime exception: {}", message, ex);
        }

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("timestamp", LocalDateTime.now().toString());
        body.put("status", status.value());
        body.put("error", status.getReasonPhrase());
        body.put("message", message);

        return ResponseEntity.status(status).body(body);
    }

    /**
     * Catch-all for any other unhandled exceptions.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> handleGenericException(Exception ex) {
        log.error("Unhandled exception: {}", ex.getMessage(), ex);

        Map<String, Object> body = new LinkedHashMap<>();
        body.put("timestamp", LocalDateTime.now().toString());
        body.put("status", 500);
        body.put("error", "Internal Server Error");
        body.put("message", ex.getMessage() != null ? ex.getMessage() : "An unexpected error occurred");

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(body);
    }
}
