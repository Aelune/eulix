package kb

import "time"

// Import represents a module import
type Import struct {
	Module string   `json:"module"`
	Items  []string `json:"items"`
}

// Param represents a function parameter
type Param struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// FunctionInfo represents a function in the codebase
type FunctionInfo struct {
	ID         string   `json:"id"`
	Name       string   `json:"name"`
	Signature  string   `json:"signature"`
	Params     []Param  `json:"params"`
	ReturnType string   `json:"return_type"`
	Docstring  string   `json:"docstring"`
	LineStart  int      `json:"line_start"`
	LineEnd    int      `json:"line_end"`
	Complexity int      `json:"complexity"`
	Calls      []string `json:"calls"`
	CalledBy   []string `json:"called_by"`
	IsAsync    bool     `json:"is_async"`
	Decorators []string `json:"decorators"`
}

// ClassInfo represents a class in the codebase
type ClassInfo struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Docstring   string   `json:"docstring"`
	LineStart   int      `json:"line_start"`
	LineEnd     int      `json:"line_end"`
	Methods     []string `json:"methods"`
	BaseClasses []string `json:"base_classes"`
}

// GlobalVar represents a global variable
type GlobalVar struct {
	Name      string `json:"name"`
	Type      string `json:"type"`
	Value     string `json:"value"`
	LineStart int    `json:"line_start"`
}

// FileInfo represents a parsed file
type FileInfo struct {
	Language      string         `json:"language"`
	LOC           int            `json:"loc"`
	Imports       []Import       `json:"imports"`
	Functions     []FunctionInfo `json:"functions"`
	Classes       []ClassInfo    `json:"classes"`
	GlobalVars    []GlobalVar    `json:"global_vars"`  // Fixed: array of GlobalVar objects
	Todos         []string       `json:"todos"`
	SecurityNotes []string       `json:"security_notes"`
}

// Metadata represents KnowledgeBase metadata
type Metadata struct {
	ProjectName    string    `json:"project_name"`
	Version        string    `json:"version"`
	ParsedAt       time.Time `json:"parsed_at"`
	Languages      []string  `json:"languages"`
	TotalFiles     int       `json:"total_files"`
	TotalLOC       int       `json:"total_loc"`
	TotalFunctions int       `json:"total_functions"`
	TotalClasses   int       `json:"total_classes"`
	TotalMethods   int       `json:"total_methods"`
}

// Node represents a dependency graph node
type Node struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

// Edge represents a dependency graph edge
type Edge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Type string `json:"type"`
}

// DependencyGraph represents the dependency graph
type DependencyGraph struct {
	Nodes []Node `json:"nodes"`
	Edges []Edge `json:"edges"`
}

// EntryPoint represents a project entry point
type EntryPoint struct {
	File     string `json:"file"`
	Function string `json:"function"`
	Type     string `json:"type"`
}

// ExternalDependency represents an external dependency
type ExternalDependency struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Source  string `json:"source"`
}

// KnowledgeBase represents the entire knowledge base
type KnowledgeBase struct {
	Metadata             Metadata               `json:"metadata"`
	Structure            map[string]*FileInfo   `json:"structure"`
	DependencyGraph      DependencyGraph        `json:"dependency_graph"`
	EntryPoints          []EntryPoint           `json:"entry_points"`
	ExternalDependencies []ExternalDependency   `json:"external_dependencies"`
}

// Location represents a code location
type Location struct {
	File string `json:"file"`
	Line int    `json:"line"`
	Type string `json:"type"`
}

// ClassLocation represents a class location with methods
type ClassLocation struct {
	File    string   `json:"file"`
	Line    int      `json:"line"`
	Methods []string `json:"methods"`
}

// FileStats represents file statistics
type FileStats struct {
	LOC       int    `json:"loc"`
	Language  string `json:"language"`
	Functions int    `json:"functions"`
	Classes   int    `json:"classes"`
}

// Index represents the quick lookup index
type Index struct {
	Functions map[string]Location      `json:"functions"`
	Classes   map[string]ClassLocation `json:"classes"`
	Files     map[string]FileStats     `json:"files"`
}

// DependencyInfo represents dependency information
type DependencyInfo struct {
	Internal []string `json:"internal"`
	External []string `json:"external"`
	Count    int      `json:"count"`
}

// PatternInfo represents pattern information
type PatternInfo struct {
	DesignPatterns []string          `json:"design_patterns"`
	Architectures  []string          `json:"architectures"`
	Common         map[string]int    `json:"common"`
}

// Summary represents the project summary
type Summary struct {
	ProjectName  string              `json:"project_name"`
	TotalFiles   int                 `json:"total_files"`
	TotalLOC     int                 `json:"total_loc"`
	Languages    []string            `json:"languages"`
	Categories   map[string][]string `json:"categories"`
	KeyFeatures  []string            `json:"key_features"`
	EntryPoints  []string            `json:"entry_points"`
	Dependencies DependencyInfo      `json:"dependencies"`
	Patterns     PatternInfo         `json:"patterns"`
}
