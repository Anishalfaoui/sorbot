# ── Sorbot Spring Boot Backend ────────────────
FROM maven:3.9-eclipse-temurin-17 AS build

WORKDIR /app
COPY pom.xml .
# Download dependencies first (layer cache)
RUN mvn dependency:go-offline -B

COPY src ./src
RUN mvn package -DskipTests -B

# ── Runtime ──────────────────────────────────
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

# Create data dir for H2 database
RUN mkdir -p data

COPY --from=build /app/target/sorbot-backend-1.0.0.jar app.jar

EXPOSE 8081

ENTRYPOINT ["java", "-jar", "app.jar"]
