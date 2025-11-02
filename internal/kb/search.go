package kb

import (
	"strings"
)

type SearchResult struct {
	FilePath     string
	FunctionName string
	Code         string
	Docstring    string
	Score        float64
}

// Search performs keyword-based search on the knowledge base
func Search(kb *KnowledgeBase, query string, topK int) []SearchResult {
	queryLower := strings.ToLower(query)
	queryTokens := tokenize(queryLower)

	var results []SearchResult

	// Search through all files
	for filePath, fileData := range kb.Structure {
		// Search functions
		for _, fn := range fileData.Functions {
			score := scoreFunction(&fn, queryTokens)
			if score > 0 {
				results = append(results, SearchResult{
					FilePath:     filePath,
					FunctionName: fn.Name,
					Code:         buildFunctionSnippet(&fn),
					Docstring:    fn.Docstring,
					Score:        score,
				})
			}
		}

		// Search classes
		for _, cls := range fileData.Classes {
			score := scoreClass(&cls, queryTokens)
			if score > 0 {
				results = append(results, SearchResult{
					FilePath:     filePath,
					FunctionName: cls.Name + " (class)",
					Code:         buildClassSnippet(&cls),
					Docstring:    cls.Docstring,
					Score:        score,
				})
			}
		}
	}

	// Sort by score (descending)
	sortByScore(results)

	// Return top K
	if len(results) > topK {
		return results[:topK]
	}
	return results
}

func scoreFunction(fn *FunctionInfo, queryTokens []string) float64 {
	score := 0.0

	// Name match (high weight)
	nameLower := strings.ToLower(fn.Name)
	for _, token := range queryTokens {
		if strings.Contains(nameLower, token) {
			score += 10.0
		}
	}

	// Docstring match (medium weight)
	docLower := strings.ToLower(fn.Docstring)
	for _, token := range queryTokens {
		if strings.Contains(docLower, token) {
			score += 5.0
		}
	}

	// Signature match (low weight)
	sigLower := strings.ToLower(fn.Signature)
	for _, token := range queryTokens {
		if strings.Contains(sigLower, token) {
			score += 2.0
		}
	}

	return score
}

func scoreClass(cls *ClassInfo, queryTokens []string) float64 {
	score := 0.0

	nameLower := strings.ToLower(cls.Name)
	for _, token := range queryTokens {
		if strings.Contains(nameLower, token) {
			score += 10.0
		}
	}

	docLower := strings.ToLower(cls.Docstring)
	for _, token := range queryTokens {
		if strings.Contains(docLower, token) {
			score += 5.0
		}
	}

	return score
}

func tokenize(text string) []string {
	// Simple tokenization: split by spaces and punctuation
	text = strings.ToLower(text)
	tokens := strings.FieldsFunc(text, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	return tokens
}

func buildFunctionSnippet(fn *FunctionInfo) string {
	return fn.Signature
}

func buildClassSnippet(cls *ClassInfo) string {
	snippet := "class " + cls.Name
	if len(cls.BaseClasses) > 0 {
		snippet += "(" + strings.Join(cls.BaseClasses, ", ") + ")"
	}
	return snippet + ":"
}

func sortByScore(results []SearchResult) {
	// Simple bubble sort (fine for small N)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}
