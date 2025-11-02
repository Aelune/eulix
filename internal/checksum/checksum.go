package checksum

import (
    "crypto/sha256"
    "encoding/hex"
    "fmt"
    "io"
    "os"
    "path/filepath"
    "sort"
    "strings"

    // "eulix/internal/config"
)

const checksumFile = ".eulix/checksum.txt"

// NeedsReparse checks if codebase has changed since last parse
func NeedsReparse() (bool, error) {
    // Load previous checksum
    previousChecksum, err := loadPreviousChecksum()
    if err != nil {
        // No previous checksum = needs parse
        return true, nil
    }

    // Calculate current checksum
    currentChecksum, err := calculateChecksum()
    if err != nil {
        return false, fmt.Errorf("failed to calculate checksum: %w", err)
    }

    return previousChecksum != currentChecksum, nil
}

// SaveChecksum saves current checksum to disk
func SaveChecksum() error {
    checksum, err := calculateChecksum()
    if err != nil {
        return fmt.Errorf("failed to calculate checksum: %w", err)
    }

    return os.WriteFile(checksumFile, []byte(checksum), 0644)
}

// calculateChecksum computes SHA256 of all tracked files
func calculateChecksum() (string, error) {
    // Load ignore patterns
    ignoreFilter, err := loadIgnoreFilter()
    if err != nil {
        return "", err
    }

    // Collect all relevant files
    var files []string
    err = filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
        if err != nil {
            return err
        }

        // Skip directories
        if info.IsDir() {
            // Skip ignored directories
            if ignoreFilter.ShouldIgnore(path) {
                return filepath.SkipDir
            }
            return nil
        }

        // Skip ignored files
        if ignoreFilter.ShouldIgnore(path) {
            return nil
        }

        // Skip non-code files (basic heuristic)
        if !isCodeFile(path) {
            return nil
        }

        files = append(files, path)
        return nil
    })

    if err != nil {
        return "", err
    }

    // Sort for deterministic checksum
    sort.Strings(files)

    // Compute combined hash
    hasher := sha256.New()
    for _, file := range files {
        // Hash filename
        hasher.Write([]byte(file))

        // Hash file content
        f, err := os.Open(file)
        if err != nil {
            continue // Skip files we can't read
        }
        if _, err := io.Copy(hasher, f); err != nil {
            f.Close()
            continue
        }
        f.Close()
    }

    return hex.EncodeToString(hasher.Sum(nil)), nil
}

// loadPreviousChecksum loads checksum from .eulix/checksum.txt
func loadPreviousChecksum() (string, error) {
    data, err := os.ReadFile(checksumFile)
    if err != nil {
        return "", err
    }
    return strings.TrimSpace(string(data)), nil
}

// isCodeFile checks if file is a code file (simple heuristic)
func isCodeFile(path string) bool {
    codeExtensions := []string{
        ".py", ".js", ".jsx", ".ts", ".tsx",
        ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
        ".java", ".rb", ".php", ".swift", ".kt",
    }

    ext := strings.ToLower(filepath.Ext(path))
    for _, codeExt := range codeExtensions {
        if ext == codeExt {
            return true
        }
    }
    return false
}

// IgnoreFilter wraps ignore pattern checking
type IgnoreFilter struct {
    patterns []string
}

func loadIgnoreFilter() (*IgnoreFilter, error) {
    patterns := []string{
        ".git",
        ".eulix",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "target",
        "dist",
        "build",
    }

    // Load .euignore if exists
    data, err := os.ReadFile(".euignore")
    if err == nil {
        lines := strings.Split(string(data), "\n")
        for _, line := range lines {
            line = strings.TrimSpace(line)
            if line != "" && !strings.HasPrefix(line, "#") {
                patterns = append(patterns, line)
            }
        }
    }

    return &IgnoreFilter{patterns: patterns}, nil
}

func (f *IgnoreFilter) ShouldIgnore(path string) bool {
    // Normalize path
    path = filepath.Clean(path)

    for _, pattern := range f.patterns {
        // Simple matching (can be improved with glob)
        if strings.Contains(path, pattern) {
            return true
        }

        // Check if any path component matches
        parts := strings.Split(path, string(filepath.Separator))
        for _, part := range parts {
            if part == pattern || strings.HasPrefix(part, pattern) {
                return true
            }
        }
    }

    return false
}
