package kb

import (
    "encoding/json"
    "fmt"
    "os"
    "sync"
)

const (
    kbPath      = ".eulix/knowledge_base.json"
    indexPath   = ".eulix/index.json"
    summaryPath = ".eulix/summary.json"
)

// Loader manages loading and caching of KB components
type Loader struct {
    kb      *KnowledgeBase
    index   *Index
    summary *Summary
    mu      sync.RWMutex
}

// NewLoader creates a new KB loader
func NewLoader() (*Loader, error) {
    loader := &Loader{}

    // Pre-load index and summary (small files)
    if err := loader.LoadIndex(); err != nil {
        return nil, fmt.Errorf("failed to load index: %w", err)
    }

    if err := loader.LoadSummary(); err != nil {
        return nil, fmt.Errorf("failed to load summary: %w", err)
    }

    return loader, nil
}

// LoadKB loads the full knowledge base (lazy loaded)
func (l *Loader) LoadKB() (*KnowledgeBase, error) {
    l.mu.Lock()
    defer l.mu.Unlock()

    if l.kb != nil {
        return l.kb, nil
    }

    data, err := os.ReadFile(kbPath)
    if err != nil {
        return nil, fmt.Errorf("failed to read KB: %w", err)
    }

    var kb KnowledgeBase
    if err := json.Unmarshal(data, &kb); err != nil {
        return nil, fmt.Errorf("failed to parse KB: %w", err)
    }

    l.kb = &kb
    return l.kb, nil
}

// LoadIndex loads the index
func (l *Loader) LoadIndex() error {
    l.mu.Lock()
    defer l.mu.Unlock()

    data, err := os.ReadFile(indexPath)
    if err != nil {
        return fmt.Errorf("failed to read index: %w", err)
    }

    var index Index
    if err := json.Unmarshal(data, &index); err != nil {
        return fmt.Errorf("failed to parse index: %w", err)
    }

    l.index = &index
    return nil
}

// LoadSummary loads the summary
func (l *Loader) LoadSummary() error {
    l.mu.Lock()
    defer l.mu.Unlock()

    data, err := os.ReadFile(summaryPath)
    if err != nil {
        return fmt.Errorf("failed to read summary: %w", err)
    }

    var summary Summary
    if err := json.Unmarshal(data, &summary); err != nil {
        return fmt.Errorf("failed to parse summary: %w", err)
    }

    l.summary = &summary
    return nil
}

// GetIndex returns the cached index
func (l *Loader) GetIndex() *Index {
    l.mu.RLock()
    defer l.mu.RUnlock()
    return l.index
}

// GetSummary returns the cached summary
func (l *Loader) GetSummary() *Summary {
    l.mu.RLock()
    defer l.mu.RUnlock()
    return l.summary
}

// GetKB returns the KB (loads if not cached)
func (l *Loader) GetKB() (*KnowledgeBase, error) {
    l.mu.RLock()
    if l.kb != nil {
        kb := l.kb
        l.mu.RUnlock()
        return kb, nil
    }
    l.mu.RUnlock()

    return l.LoadKB()
}
