# AI Engine Sorbot


## Table des matieres
1. Vision globale du moteur IA
2. Architecture interne du dossier ai_engine
3. Bases ML pour ce projet
4. Pourquoi XGBoost
5. Variables de configuration (config.py)
6. Collecte des donnees (data_loader.py)
7. Nettoyage et normalisation des donnees
8. Feature engineering (feature_eng.py)
9. Construction de la cible (target)
10. Build dataset final
11. Entrainement (trainer.py)
12. Validation walk-forward
13. Metriques de qualite
14. Fichiers de modele sauvegardes
15. Generation de prediction (predictor.py)
16. Analyse enrichie du marche
17. Regles de decision LONG/SHORT/NO_TRADE
18. Gestion du risque (risk_manager.py)
19. Execution Binance Spot (exchange.py)
20. API FastAPI (main.py)
21. Scripts utilitaires (run_train.py et backtest.py)
22. Flux complet de bout en bout
23. Limites actuelles et points d'amelioration
24. Plan d'apprentissage recommande
25. Glossaire rapide

---

## 1) Vision globale du moteur IA
Le moteur IA de Sorbot prend des donnees de marche BTC/USD, fabrique un grand nombre de signaux techniques (features), entraine un modele XGBoost, puis produit une prediction exploitable par le backend Java.

En simplifiant:
1. Recuperer OHLCV (Open, High, Low, Close, Volume)
2. Construire des features techniques
3. Construire une cible binaire (UP ou DOWN)
4. Entrainement du modele XGBoost
5. Prediction en temps reel
6. Filtrage par confiance et risque
7. Envoi des ordres Spot Binance (si conditions valides)

---

## 2) Architecture interne du dossier ai_engine
Le coeur du systeme est organise comme suit:

- config.py: toutes les constantes du moteur
- main.py: API FastAPI et orchestration runtime
- run_train.py: script rapide d'entrainement
- backtest.py: simulation historique
- ml_core/data_loader.py: telechargement des donnees
- ml_core/feature_eng.py: creation des features + target
- ml_core/trainer.py: entrainement walk-forward XGBoost
- ml_core/predictor.py: inference et enrichissement de la prediction
- ml_core/risk_manager.py: sizing et contraintes de risque
- ml_core/exchange.py: execution Spot sur Binance

---

## 3) Bases ML pour ce projet

### 3.1 Apprentissage supervise
Le modele apprend a partir d'exemples historiques:
- Entree X: vecteur de features techniques
- Sortie y: direction future (UP ou DOWN)

Le modele apprend une fonction approximative:
$$f(X) = P(UP \mid X)$$

### 3.2 Pourquoi binaire
La target est binaire:
- 1 = le prix monte suffisamment dans un horizon court
- 0 = le prix baisse suffisamment
- les zones plates sont ignorees (NaN)

C'est un choix tres courant en trading quantitatif pour eviter de sur-apprendre du bruit.

---

## 4) Pourquoi XGBoost
XGBoost est un algorithme de gradient boosting sur arbres de decision.

Avantages dans ce contexte:
- performant sur donnees tabulaires (features techniques)
- robuste aux relations non-lineaires
- bonne gestion des interactions entre variables
- entrainement relativement rapide
- feature importance exploitable

Principe simplifie:
- on ajoute des arbres successifs,
- chaque nouvel arbre corrige les erreurs des precedents,
- le modele final est la somme de nombreux petits arbres specialises.

---

## 5) Variables de configuration (config.py)
Points majeurs:

### 5.1 Donnees et symboles
- SYMBOL = BTCUSDT
- YFINANCE_TICKER = BTC-USD
- Timeframes: 1h (principal), 4h, 1d

### 5.2 Hyperparametres de prediction
- CONFIDENCE_LONG = 0.65
- CONFIDENCE_SHORT = 0.35

Interpretation:
- si P(UP) >= 0.65 -> biais LONG
- si P(UP) <= 0.35 -> biais SHORT
- sinon NO_TRADE

### 5.3 Hyperparametres de training
Exemples:
- n_estimators = 800
- max_depth = 5
- learning_rate = 0.01
- early stopping = 50

### 5.4 Gestion du risque
- compte de reference: 500 USD
- risque par trade: 1.5%
- max positions ouvertes: 1
- SL/TP ATR-based

---

## 6) Collecte des donnees (data_loader.py)

### 6.1 Source
Le moteur telecharge les donnees via yfinance.

### 6.2 Probleme yfinance 1h
La granularite 1h est limitee sur de petites fenetres (environ 60 jours par requete).

### 6.3 Solution implementee
Le code utilise un telechargement par morceaux:
- chunks de 59 jours,
- remontant jusqu'a 730 jours,
- concatenation + dedup + tri chronologique.

Resultat: dataset 1h beaucoup plus riche (environ 2 ans).

### 6.4 Multi-timeframe
- 1h: principal
- 4h: derive/resample de 1h
- 1d: contexte macro

---

## 7) Nettoyage et normalisation des donnees
Le pipeline applique plusieurs gardes-fous:
- conservation des colonnes OHLCV uniquement,
- suppression des valeurs manquantes critiques,
- aplatissement de colonnes si MultiIndex,
- remplacement des infinis par NaN,
- suppression finale des lignes inexploitables.

Important:
- les features HTF manquantes en debut de serie sont remplies a 0,
- la cible NaN (zones neutres) est retiree du dataset d'entrainement.

---

## 8) Feature engineering (feature_eng.py)
Le moteur construit un ensemble large de features (100+).

### 8.1 Families de features
1. Trend
2. Momentum
3. Volatility
4. Volume
5. Structure (S/R, pivots)
6. Candle anatomy/patterns
7. Regime
8. Divergence
9. Calendar/session
10. HTF confluence

### 8.2 Exemples concrets
- ema9_dist, ema21_dist, ema200_dist
- rsi, macd_hist, stoch_k, adx
- atr_pct, bb_bandwidth, squeeze
- vol_ratio, obv_norm
- pivot_dist, r1_dist, s1_dist
- doji, engulfing, hammer
- session_us, hour_sin, dow_cos
- htf_4h_rsi, htf_1d_macd_hist, etc.

### 8.3 Anti data leakage
Le code evite la fuite d'information future:
- tout est calcule avec historique courant/passe,
- la cible regarde le futur, mais uniquement pour etiqueter les exemples,
- les features restent strictement basees sur l'instant t.

---

## 9) Construction de la cible (target)
La cible est creee avec:
- LOOKAHEAD_CANDLES = 3
- UP_THRESHOLD = +0.3%
- DOWN_THRESHOLD = -0.3%

Pour chaque barre t:
- on calcule le rendement futur entre t et t+3,
- si rendement >= +0.3% -> target = 1
- si rendement <= -0.3% -> target = 0
- sinon -> NaN (ignore pour training)

C'est un design intelligent pour eviter les micro-mouvements non significatifs.

---

## 10) Build dataset final
La fonction build_dataset:
1. cree les features de base (1h)
2. ajoute les features HTF si disponibles
3. ajoute target si include_target = true
4. nettoie (inf/NaN)
5. drop les lignes invalides

Sortie:
- DataFrame pret pour l'entrainement
- shape typique: [N lignes, M features + target]

---

## 11) Entrainement (trainer.py)

### 11.1 Objectif
Entrainer un modele XGBoost binaire robuste au temps.

### 11.2 Strategie
- Walk-forward CV (series temporelles)
- Ajustement dynamique scale_pos_weight (balance classes)
- Early stopping pour limiter l'overfitting

### 11.3 Etapes
1. Separation X/y
2. Generation des splits temporels
3. Entrainement fold par fold
4. Calcul metriques fold
5. Entrainement final sur presque tout l'historique
6. Evaluation finale sur derniere fenetre
7. Sauvegarde modele + metadata

---

## 12) Validation walk-forward
C'est le point cle pour un modele de trading.

Contrairement a un split aleatoire classique, on respecte l'ordre du temps.
Chaque fold fait:
- train sur passe
- test sur futur immediat

Cela simule mieux la vraie vie de production.

---

## 13) Metriques de qualite
Le moteur suit:
- Accuracy
- Precision
- Recall
- F1
- AUC-ROC

Interpretation rapide:
- Accuracy: taux global correct
- Precision: qualite des signaux positifs
- Recall: couverture des vrais positifs
- F1: compromis precision/recall
- AUC: separation probabiliste globale

---

## 14) Fichiers de modele sauvegardes
Apres training:

1. models/btc_model.json
- le booster XGBoost natif

2. models/btc_meta.json
- date d'entrainement
- nombre d'echantillons
- nombre de features
- metriques CV/finales
- top features
- meilleurs hyperparametres utiles

---

## 15) Generation de prediction (predictor.py)

### 15.1 Chargement du modele
Predictor.load() charge:
- booster depuis btc_model.json
- meta depuis btc_meta.json

### 15.2 Alignement des features
Au runtime, les colonnes peuvent legerement differer.
Le code:
- conserve l'ordre des features d'entrainement,
- ajoute les features manquantes a 0,
- garantit la compatibilite DMatrix.

### 15.3 Probabilite
Le modele renvoie P(UP).
On deduit P(DOWN) = 1 - P(UP).

---

## 16) Analyse enrichie du marche
Le moteur ne renvoie pas juste LONG/SHORT.
Il calcule aussi:
- regime de marche,
- etat indicateurs (RSI, MACD, ADX, Stoch, etc.),
- support/resistance,
- divergences,
- alignement multi-timeframe,
- texte explicatif humain (conclusion).

C'est tres utile pour l'UI et la transparence decisionnelle.

---

## 17) Regles de decision LONG/SHORT/NO_TRADE
Regles principales:

1. Seuil de confiance
- LONG si P(UP) >= 0.65
- SHORT si P(UP) <= 0.35
- sinon NO_TRADE

2. Validation risk/reward
Si ratio R:R < minimum configure (ex: 1.8), le signal est degrade en NO_TRADE.

3. Stop Loss / Take Profit
SL et TP sont derives de l'ATR:
- SL = 1.5 x ATR
- TP = 3.0 x ATR (selon config)

---

## 18) Gestion du risque (risk_manager.py)
Le module impose des contraintes de securite:
- max 1 position ouverte
- verification balance minimale
- calcul taille de position par risque fixe

Formule simplifiee:
$$qty = \frac{risk\_usd}{distance\_SL}$$

Puis cap par balance (pas de levier en Spot).

Le module suit aussi:
- ouverture/fermeture de position,
- evolution du solde apres PnL.

---

## 19) Execution Binance Spot (exchange.py)

### 19.1 Philosophie
Le code est Spot-only:
- LONG = achat BTC/USDT
- SHORT = pas de short leverage, mais vente du BTC detenu

### 19.2 Entree LONG
- ordre market BUY
- tentative OCO SELL (TP + SL)
- fallback stop loss si OCO echoue

### 19.3 Sortie
close_position() vend le BTC disponible au marche, apres annulation des ordres ouverts.

### 19.4 Credentials
Le module peut utiliser:
- creds globaux depuis .env
- ou creds injectes dynamiquement via payload API

---

## 20) API FastAPI (main.py)
Endpoints principaux:

- GET /
  Health check moteur

- POST /train
  Relance l'entrainement puis recharge le modele

- GET /predict
  Renvoie prediction enrichie

- POST /trade
  Prediction + execution conditionnelle

- GET/POST /status
  Statut compte, position, prix, etc.

- POST /close
  Fermeture de position

- GET /model-info
  Metriques et infos du modele

---

## 21) Scripts utilitaires

### 21.1 run_train.py
Script simple pour:
- fetch data,
- build dataset,
- lancer training,
- afficher metriques cle.

### 21.2 backtest.py
Backtest walk-forward:
- re-entrainement periodique,
- signaux avec seuils de confiance,
- simulation SL/TP,
- equity curve,
- win rate, drawdown, profit factor.

C'est un excellent outil de validation hors production.

---

## 22) Flux complet de bout en bout

### 22.1 Training flow
1. POST /train
2. fetch 1h/4h/1d
3. build features + target
4. train walk-forward XGBoost
5. save model + meta
6. reload predictor

### 22.2 Inference flow
1. GET /predict
2. fetch donnees recentes
3. build features (sans target)
4. model -> probabilite
5. calcul SL/TP ATR + analyse marche
6. retour JSON enrichi

### 22.3 Trade flow
1. POST /trade
2. recuperer signal
3. valider risque
4. calcul sizing
5. execution Spot Binance
6. retour resultat ordre(s)

---

## 23) Limites actuelles et points d'amelioration

### 23.1 Limites
- modele unique (single model)
- regime adaptation encore simple
- dependance forte aux indicateurs techniques
- pas encore de calibration de proba explicite
- pas de modele sequence deep learning (ce n'est pas un defaut, juste un choix)

### 23.2 Ameliorations possibles
- calibration probabiliste (Platt/Isotonic)
- optimisation hyperparametres automatique
- modele par regime (trend/range)
- monitoring drift des features
- tests de robustesse (stress windows)
- journaling plus fin des decisions NO_TRADE

---

## 24) Plan d'apprentissage recommande (software engineer -> AI trading)

Semaine 1:
- comprendre pipeline data -> features -> target
- lancer run_train.py
- lire btc_meta.json

Semaine 2:
- analyser features importantes
- jouer avec thresholds UP/DOWN et confidence
- observer impact sur precision/recall

Semaine 3:
- executer backtest sur plusieurs periodes
- comparer drawdown/profit factor
- ajuster risk manager

Semaine 4:
- instrumenter monitoring en production
- ecrire tests de non-regression sur pipeline features
- documenter strategie de retrain periodic

---

## 25) Glossaire rapide

- OHLCV: Open, High, Low, Close, Volume
- Feature: variable explicative
- Target: variable a predire
- Inference: prediction en production
- Walk-forward: validation temporelle realiste
- Overfitting: apprentissage trop adapte au passe
- ATR: volatilite moyenne vraie
- R:R: ratio reward/risk
- OCO: ordre lie TP + SL

---

## Conclusion
Ton AI Engine est deja un socle solide, bien structure pour de la production:
- pipeline clair,
- modele tabulaire robuste,
- enrichissement explicatif,
- securite risque/execution,
- endpoints API bien separes.

En tant que software engineer, tu peux le faire evoluer incrementally sans tout casser:
- observabilite,
- calibration des seuils,
- optimisation hyperparametres,
- experimentation de nouveaux signaux.

Le plus important: garder la discipline de validation temporelle et de gestion du risque.
C'est ce qui fait la difference entre un prototype IA et un systeme de trading exploitable.
