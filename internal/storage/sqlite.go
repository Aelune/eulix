package storage

import (
    "database/sql"
    "fmt"
    "time"

    _ "github.com/mattn/go-sqlite3"
)

const dbPath = ".eulix/cache.db"

// SQLite manages persistent storage
type SQLite struct {
    db *sql.DB
}

// QueryRecord represents a stored query
type QueryRecord struct {
    ID        int64
    SessionID string
    Query     string
    Answer    string
    QueryType string
    Source    string
    Duration  float64
    Timestamp time.Time
}

// NewSQLite creates a new SQLite storage
func NewSQLite() (*SQLite, error) {
    db, err := sql.Open("sqlite3", dbPath)
    if err != nil {
        return nil, fmt.Errorf("failed to open database: %w", err)
    }

    store := &SQLite{db: db}

    // Initialize schema
    if err := store.initSchema(); err != nil {
        return nil, fmt.Errorf("failed to initialize schema: %w", err)
    }

    return store, nil
}

// initSchema creates the database schema
func (s *SQLite) initSchema() error {
    schema := `
    CREATE TABLE IF NOT EXISTS queries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        query_type TEXT,
        source TEXT,
        duration REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_session ON queries(session_id);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON queries(timestamp DESC);

    CREATE TABLE IF NOT EXISTS config (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS checksums (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        checksum TEXT NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    `

    _, err := s.db.Exec(schema)
    return err
}

// SaveQuery saves a query record
func (s *SQLite) SaveQuery(record QueryRecord) error {
    query := `
    INSERT INTO queries (session_id, query, answer, query_type, source, duration, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    `

    _, err := s.db.Exec(query,
        record.SessionID,
        record.Query,
        record.Answer,
        record.QueryType,
        record.Source,
        record.Duration,
        record.Timestamp,
    )

    return err
}

// GetHistory retrieves query history
func (s *SQLite) GetHistory(limit int) ([]QueryRecord, error) {
    query := `
    SELECT id, session_id, query, answer, query_type, source, duration, timestamp
    FROM queries
    ORDER BY timestamp DESC
    LIMIT ?
    `

    rows, err := s.db.Query(query, limit)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var records []QueryRecord
    for rows.Next() {
        var r QueryRecord
        err := rows.Scan(
            &r.ID,
            &r.SessionID,
            &r.Query,
            &r.Answer,
            &r.QueryType,
            &r.Source,
            &r.Duration,
            &r.Timestamp,
        )
        if err != nil {
            return nil, err
        }
        records = append(records, r)
    }

    return records, nil
}

// GetSessionHistory retrieves history for a specific session
func (s *SQLite) GetSessionHistory(sessionID string) ([]QueryRecord, error) {
    query := `
    SELECT id, session_id, query, answer, query_type, source, duration, timestamp
    FROM queries
    WHERE session_id = ?
    ORDER BY timestamp ASC
    `

    rows, err := s.db.Query(query, sessionID)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var records []QueryRecord
    for rows.Next() {
        var r QueryRecord
        err := rows.Scan(
            &r.ID,
            &r.SessionID,
            &r.Query,
            &r.Answer,
            &r.QueryType,
            &r.Source,
            &r.Duration,
            &r.Timestamp,
        )
        if err != nil {
            return nil, err
        }
        records = append(records, r)
    }

    return records, nil
}

// ClearHistory clears all query history
func (s *SQLite) ClearHistory() error {
    _, err := s.db.Exec("DELETE FROM queries")
    return err
}

// ClearOldHistory clears queries older than specified duration
func (s *SQLite) ClearOldHistory(olderThan time.Duration) error {
    cutoff := time.Now().Add(-olderThan)
    _, err := s.db.Exec("DELETE FROM queries WHERE timestamp < ?", cutoff)
    return err
}

// GetStats returns query statistics
func (s *SQLite) GetStats() (map[string]interface{}, error) {
    stats := make(map[string]interface{})

    // Total queries
    var total int
    err := s.db.QueryRow("SELECT COUNT(*) FROM queries").Scan(&total)
    if err != nil {
        return nil, err
    }
    stats["total_queries"] = total

    // Queries by type
    rows, err := s.db.Query(`
        SELECT query_type, COUNT(*) as count
        FROM queries
        GROUP BY query_type
    `)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    typeStats := make(map[string]int)
    for rows.Next() {
        var qtype string
        var count int
        if err := rows.Scan(&qtype, &count); err != nil {
            return nil, err
        }
        typeStats[qtype] = count
    }
    stats["by_type"] = typeStats

    // Average duration
    var avgDuration float64
    err = s.db.QueryRow("SELECT AVG(duration) FROM queries").Scan(&avgDuration)
    if err != nil && err != sql.ErrNoRows {
        return nil, err
    }
    stats["avg_duration"] = avgDuration

    return stats, nil
}

// SaveConfig saves a configuration value
func (s *SQLite) SaveConfig(key, value string) error {
    query := `
    INSERT INTO config (key, value)
    VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
    `
    _, err := s.db.Exec(query, key, value)
    return err
}

// GetConfig retrieves a configuration value
func (s *SQLite) GetConfig(key string) (string, error) {
    var value string
    err := s.db.QueryRow("SELECT value FROM config WHERE key = ?", key).Scan(&value)
    if err == sql.ErrNoRows {
        return "", nil
    }
    return value, err
}

// Close closes the database connection
func (s *SQLite) Close() error {
    return s.db.Close()
}
