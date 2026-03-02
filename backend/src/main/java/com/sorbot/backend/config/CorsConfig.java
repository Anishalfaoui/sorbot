package com.sorbot.backend.config;

// CORS is now handled by SecurityConfig — this class is intentionally empty.
// Keeping it to avoid breaking any component-scan references.

import org.springframework.context.annotation.Configuration;

@Configuration
public class CorsConfig {
    // See SecurityConfig.corsConfigurationSource()
}
