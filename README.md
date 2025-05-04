# tf-idf-matcher

A lightweight Rust library for approximate string matching using [TF–IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
vectorization and cosine similarity. The matcher
converts strings into n‑gram [TF–IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectors and computes similarity
scores to rank the closest matches in a corpus.

This project is heavily inspired by the Python package [tfidf-matcher](https://pypi.org/project/tfidf-matcher/).
The Rust implementation is more performant and more ergonomic.

---

## Features

- Generate n‑gram tokenization (configurable n‑gram length)
- Fit and transform a corpus of documents to sparse [TF–IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) matrices
- Compute cosine similarity between query strings and the corpus
- Retrieve top‑k matches with confidence scores ∈ `[0,1]`
- Expose feature indices for deeper analysis

## Installation

Add this crate to your `Cargo.toml`:

```toml
[dependencies]
tf-idf-matcher = "0.1"
```

## Quick Start

```rust
use tf_idf_matcher::{TFIDFMatcher, Needle};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prepare a list of documents (corpus)
    let corpus = vec![
        "The quick brown fox".to_string(),
        "Rust programming language".to_string(),
        "Natural language processing".to_string(),
    ];

    // Build the matcher with 3‑gram tokenization
    let matcher = TFIDFMatcher::new(corpus, 3)?;

    // Find the best match for a single query
    let result = matcher.find_one("quick fox", 3)?;
    println!("Query: {}", result.needle);
    for entry in result.matches {
        println!(
            "  Match: {} (index {}) – {:.2}% confidence",
            entry.haystack, entry.haystack_idx, entry.confidence
        );
    }

    // Or find matches for multiple queries
    let queries = vec!["rust lang", "natural NLP"];
    let results = matcher.find_many(queries, 2)?;
    for Needle { needle, matches } in results {
        println!("Query: {}", needle);
        for m in matches {
            println!("  {}: {:.2}%", m.haystack, m.confidence);
        }
    }

    Ok(())
}
```

## API Overview

- `TFIDFMatcher::new(haystack: Vec<String>, ngram_length: usize)`  
  Creates a new matcher from a list of strings, using n‑gram TF-IDF vectorization.

- `find_one(&self, needle: &str, top_k: usize)`  
  Returns a `Needle` containing the top‑`k` matches for a single query.

- `find_many(&self, needles: Vec<&str>, top_k: usize)`  
  Returns a vector of `Needle` structs, one per query string.

- `features(&self, needle: &str)`  
  Returns the indices of active TF-IDF Veatures for a given query.

## Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for details.
