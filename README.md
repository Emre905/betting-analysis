# Betting Site Scraping Project
This project focuses on scraping Lithuanian betting sites to gather data for identifying arbitrage betting opportunities. 
Currently, the targeted websites are **Topsport**, **Betsafe**, and **7Bet**.

## Workflow

### 1. Web Scraping
- Extract betting odds, matches, dates from all sites and make corresponding Pandas dataframe.

### 2. Data Integration
- Combine the scraped data into a single dataframe.

### 3. Arbitrage Analysis
- Check the merged data for arbitrage bets.
- Record the recommended stakes and the corresponding bookmakers for each bet.

### 4. Optional: Database Integration
- Upload the analyzed data to a database for:
  - Historical data storage.
  - Advanced analytics and trend identification.
