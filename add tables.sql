USE ArbitrageAnalysis;
CREATE TABLE matches (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    team1 VARCHAR(255) NOT NULL,
    team2 VARCHAR(255) NOT NULL,
    league VARCHAR(255),
    match_date DATETIME NOT NULL,
    draw_possible TINYINT(1) DEFAULT 0,
    best_arbitrage FLOAT
);

CREATE TABLE scraped_data (
    match_id INT NOT NULL, -- Foreign key to matches table
    team1 VARCHAR(255) NOT NULL,
    team2 VARCHAR(255) NOT NULL,
	league VARCHAR(255),
    scrape_time DATETIME DEFAULT CURRENT_TIMESTAMP, -- Timestamp for when the scrape occurred
    best_arbitrage FLOAT,
    odd1_topsport FLOAT,
    odd_draw_topsport FLOAT,                     -- NULL if no draw option
    odd2_topsport FLOAT,
    odd1_betsafe FLOAT,
    odd_draw_betsafe FLOAT,                      -- NULL if no draw option
    odd2_betsafe FLOAT,
    odd1_7bet FLOAT,
    odd_draw_7bet FLOAT,                       -- NULL if no draw option
    odd2_7bet FLOAT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE -- Maintain one-to-many relationship
);
