package llm

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"
)

type OpenAIRequest struct {
    Model  string `json:"model"`
    Prompt string `json:"prompt"`
    Stream bool   `json:"stream"`
}

type OpenAIResponse struct {
    Response string `json:"response"`
    Done     bool   `json:"done"`
}

func QueryOpenAI(url, model, prompt string) (string, error) {
    // DEBUG
    fmt.Printf("[DEBUG QueryOpenAI] URL: %s, Model: %s\n", url, model)
    fmt.Printf("[DEBUG QueryOpenAI] Prompt length: %d\n", len(prompt))

    // Build request
    reqBody := OllamaRequest{
        Model:  model,
        Prompt: prompt,
        Stream: false,
    }

    jsonData, err := json.Marshal(reqBody)
    if err != nil {
        return "", fmt.Errorf("failed to marshal request: %w", err)
    }

    // Create request with timeout context
    apiURL := fmt.Sprintf("%s/api/generate", url)
    fmt.Printf("[DEBUG QueryOpenAI] Sending request to: %s\n", apiURL)

    // Set a longer timeout for LLM responses (3 minutes)
    ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(jsonData))
    if err != nil {
        return "", fmt.Errorf("failed to create request: %w", err)
    }
    req.Header.Set("Content-Type", "application/json")

    // Create HTTP client with timeout
    client := &http.Client{
        Timeout: 180 * time.Second,
    }

    fmt.Printf("[DEBUG QueryOpenAI] Waiting for response...\n")
    resp, err := client.Do(req)
    if err != nil {
        return "", fmt.Errorf("failed to send request: %w", err)
    }
    defer resp.Body.Close()

    fmt.Printf("[DEBUG QueryOpenAI] Response status: %d\n", resp.StatusCode)

    if resp.StatusCode != http.StatusOK {
        body, _ := io.ReadAll(resp.Body)
        return "", fmt.Errorf("ollama returned status %d: %s", resp.StatusCode, string(body))
    }

    // Read response
    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return "", fmt.Errorf("failed to read response: %w", err)
    }

    fmt.Printf("[DEBUG QueryOpenAI] Response body length: %d\n", len(body))

    // Check if body is empty
    if len(body) == 0 {
        return "", fmt.Errorf("ollama returned empty body")
    }

    var ollamaResp OllamaResponse
    if err := json.Unmarshal(body, &ollamaResp); err != nil {
        // Print the actual body for debugging
        fmt.Printf("[DEBUG QueryOpenAI] Failed to parse JSON. Body: %s\n", string(body[:min(200, len(body))]))
        return "", fmt.Errorf("failed to parse response: %w", err)
    }

    fmt.Printf("[DEBUG QueryOpenAI] Response text length: %d\n", len(ollamaResp.Response))

    if ollamaResp.Response == "" {
        return "", fmt.Errorf("ollama returned empty response")
    }

    return ollamaResp.Response, nil
}
