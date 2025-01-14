# Betting Site Scraping Project
This project focuses on scraping Lithuanian betting sites to gather data for identifying arbitrage betting opportunities. 
Currently, the targeted websites are **Topsport**, **Betsafe**, and **7Bet**.

## Workflow

### 1. Web Scraping
- Use asynchronous scraping to extract betting odds, matches, dates from all sites and make corresponding Pandas dataframe.

### 2. Data Integration
- Combine the scraped data into a single dataframe.

### 3. Arbitrage Analysis
- Check the merged data for arbitrage bets.
- Record the recommended stakes (as integers) and the corresponding bookmakers for each bet.

### 4. Optional: Database Integration
- Upload the analyzed data to a database for:
  - Historical data storage.
  - Advanced analytics and trend identification.

### 5. Optional: Sending Mails
- Send results of new arbitrage matches as mail.

## Guide
- If you just want to check for arbitraged matches, you can run the jupiter file (except for last 2 cells). And send results as mail from last cell (you will need to mail `arbitrage_list`). You can also run `scrape.py` file (without `save_to_database` function), which runs quite faster.
- If you want to use database features, you will need to make a SQL database and use `add tables.sql`
to make 2 necessary tables.
