package config

import (
    "os"
    "github.com/pelletier/go-toml/v2"
)

const (
    eulixDir   = ".eulix"
    configFile = ".eulix/config.toml"
)

type Config struct {
    LLM struct {
        Provider    string `toml:"provider"`    // "ollama" or "openai"
        Model       string `toml:"model"`       // "llama3.2" or "gpt-4o-mini"
        Temperature float64 `toml:"temperature"`
    } `toml:"llm"`

    OpenAI struct {
        APIKey string `toml:"api_key"`
    } `toml:"openai"`

    Parser struct {
        BinaryPath string `toml:"binary_path"`
        Threads    int    `toml:"threads"`
    } `toml:"parser"`

    Cache struct {
        Enabled bool `toml:"enabled"`
    } `toml:"cache"`
}

// Load loads configuration from .eulix/config.toml
func Load() *Config {
    cfg := defaultConfig()

    data, err := os.ReadFile(configFile)
    if err != nil {
        return cfg // Return defaults if config doesn't exist
    }

    if err := toml.Unmarshal(data, cfg); err != nil {
        return cfg
    }

    return cfg
}

// Save saves configuration to .eulix/config.toml
func Save(cfg *Config) error {
    data, err := toml.Marshal(cfg)
    if err != nil {
        return err
    }

    return os.WriteFile(configFile, data, 0644)
}

// Set sets a configuration value
func Set(key, value string) error {
    cfg := Load()

    switch key {
    case "llm.provider":
        cfg.LLM.Provider = value
    case "llm.model":
        cfg.LLM.Model = value
    case "openai.api_key":
        cfg.OpenAI.APIKey = value
    case "parser.binary_path":
        cfg.Parser.BinaryPath = value
    default:
        return nil // Ignore unknown keys
    }

    return Save(cfg)
}

// IsInitialized checks if Eulix is initialized in current directory
func IsInitialized() bool {
    _, err := os.Stat(eulixDir)
    return err == nil
}

// defaultConfig returns default configuration
func defaultConfig() *Config {
    cfg := &Config{}
    cfg.LLM.Provider = "ollama"
    cfg.LLM.Model = "llama3.2"
    cfg.LLM.Temperature = 0.7
    cfg.Parser.Threads = 4
    cfg.Cache.Enabled = true
    return cfg
}
