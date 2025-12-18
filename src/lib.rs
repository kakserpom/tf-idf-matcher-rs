#![warn(clippy::pedantic)]
use linfa_preprocessing::tf_idf_vectorization::{FittedTfIdfVectorizer, TfIdfVectorizer};
use linfa_preprocessing::{PreprocessingError, Tokenizer};
use ndarray::Array1;
use sprs::{CsMat, CsVecView};
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::collections::BinaryHeap;

#[cfg(test)]
mod tests;
/// A single match result from the corpus.
#[derive(Debug, Clone)]
#[must_use]
pub struct MatchEntry<'a> {
    /// The matched string from the corpus.
    pub haystack: &'a str,
    /// Similarity score between 0.0 and 1.0.
    pub confidence: f64,
    /// Index of this match in the original corpus.
    pub haystack_idx: usize,
}

/// Container for query results, holding the original query and its matches.
#[derive(Debug, Clone)]
#[must_use]
pub struct Needle<'a> {
    /// The original query string.
    pub needle: &'a str,
    /// The top-k matches ranked by confidence.
    pub matches: Vec<MatchEntry<'a>>,
}

impl Needle<'_> {
    pub fn debug_print(&self) {
        for MatchEntry {
            haystack,
            confidence,
            haystack_idx,
        } in &self.matches
        {
            println!(
                "Needle: {:<15} | Haystack: {:<35} | Confidence: {:>5.2} | Haystack Index: {}",
                self.needle, haystack, confidence, haystack_idx
            );
        }
    }
}

trait Normalize {
    fn normalize(&self) -> Vec<f64>;
}
impl Normalize for CsMat<f64> {
    #[inline]
    fn normalize(&self) -> Vec<f64> {
        self.outer_iterator()
            .map(|row| row.data().iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect()
    }
}

/// A TF-IDF based string matcher for finding approximate matches in a corpus.
#[derive(Debug, Clone)]
pub struct TFIDFMatcher {
    haystack: Vec<String>,
    fitted: FittedTfIdfVectorizer,
    haystack_tfidf: CsMat<f64>,
    haystack_norm: Vec<f64>,
    ngram_length: usize,
}

/// Rounds a similarity score to 2 decimal places.
#[inline]
fn round_confidence(sim: f64) -> f64 {
    (sim * 100.0).round() / 100.0
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Scored {
    sim: f64,
    idx: usize,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed comparison for min-heap behavior
        other.sim.partial_cmp(&self.sim).unwrap_or(Equal)
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TFIDFMatcher {
    fn text_into_ngrams(text: &str, n: usize) -> String {
        // Pre-calculate capacity: text length + underscores + 2 boundary chars
        let word_count = text.split_whitespace().count();
        let estimated_len = text.len() + word_count.saturating_sub(1) + 2;
        let mut chars = Vec::with_capacity(estimated_len);

        // Build character sequence: _word1_word2_
        chars.push('_');
        let mut first = true;
        for word in text.split_whitespace() {
            if !first {
                chars.push('_');
            }
            first = false;
            chars.extend(word.chars().flat_map(char::to_lowercase));
        }
        chars.push('_');

        if chars.len() < n {
            return String::new();
        }

        // Estimate result capacity: ~(n+1) chars per n-gram including space
        let ngram_count = chars.len() - n + 1;
        let mut result = String::with_capacity(ngram_count * (n + 1));

        for i in 0..=chars.len() - n {
            // Skip n-grams that cross word boundaries (have _ in middle)
            if n > 2 && chars[i + 1..i + n - 1].contains(&'_') {
                continue;
            }
            if !result.is_empty() {
                result.push(' ');
            }
            result.extend(&chars[i..i + n]);
        }
        result
    }
    /// Creates a new TF-IDF matcher from a corpus of strings.
    ///
    /// # Arguments
    /// * `haystack` - The corpus of strings to match against.
    /// * `ngram_length` - The length of n-grams to use (e.g., 3 for trigrams).
    ///
    /// # Errors
    /// Returns an error if TF-IDF vectorization fails.
    pub fn new<T>(
        haystack: impl IntoIterator<Item = T>,
        ngram_length: usize,
    ) -> Result<Self, PreprocessingError>
    where
        T: Into<String>,
    {
        fn split_by_whitespace(text: &str) -> Vec<&str> {
            text.split_whitespace().collect()
        }

        let haystack: Vec<String> = haystack.into_iter().map(Into::into).collect();
        let processed_haystack: Vec<String> = haystack
            .iter()
            .map(|s| Self::text_into_ngrams(s, ngram_length))
            .collect();

        let processed_array = Array1::from_vec(processed_haystack);
        let fitted = TfIdfVectorizer::default()
            .convert_to_lowercase(true)
            .tokenizer(Tokenizer::Function(split_by_whitespace))
            .fit::<String, _>(&processed_array)?;

        let haystack_tfidf = fitted.transform(&processed_array)?;
        Ok(Self {
            haystack,
            fitted,
            haystack_norm: haystack_tfidf.normalize(),
            haystack_tfidf,
            ngram_length,
        })
    }

    /// Finds the top-k matches for a single needle string.
    ///
    /// Returns a [`Needle`] containing the query and its ranked matches.
    ///
    /// # Errors
    /// Returns an error if TF-IDF transformation fails.
    ///
    /// # Panics
    /// Panics if the TF-IDF transformation returns an empty result (should not happen).
    pub fn find<'a>(
        &'a self,
        needle: &'a str,
        top_k: usize,
    ) -> Result<Needle<'a>, PreprocessingError> {
        let needles_tfidf = self
            .fitted
            .transform(&Array1::from_iter([Self::text_into_ngrams(
                needle,
                self.ngram_length,
            )]))?;
        let needle_v = needles_tfidf.outer_iterator().next().unwrap();
        let q_norm = needles_tfidf.normalize()[0];
        let mut similarities: Vec<(usize, f64)> = self
            .haystack_tfidf
            .outer_iterator()
            .enumerate()
            .map(|(col_idx, row)| {
                let dot_val = row.dot(needle_v);
                let denom = q_norm * self.haystack_norm[col_idx];
                let sim = if denom == 0.0 { 0.0 } else { dot_val / denom };
                (col_idx, sim)
            })
            .collect();
        let k = top_k.min(similarities.len());
        let matches = if k > 0 {
            // Use partial sort: O(n) selection + O(k log k) sort of top k
            similarities.select_nth_unstable_by(k - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(Equal)
            });
            let top_k = &mut similarities[..k];
            top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Equal));
            top_k
                .iter()
                .map(|(idx, sim)| MatchEntry {
                    haystack: &self.haystack[*idx],
                    haystack_idx: *idx,
                    confidence: round_confidence(*sim),
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(Needle { needle, matches })
    }

    /// Returns the indices of active TF-IDF features for a needle.
    ///
    /// Useful for debugging and understanding which n-grams are matched.
    ///
    /// # Panics
    /// Panics if TF-IDF transformation fails (should not happen with valid input).
    #[must_use]
    pub fn features(&self, needle: &str) -> Vec<usize> {
        self.fitted
            .transform(&Array1::from(vec![Self::text_into_ngrams(
                needle,
                self.ngram_length,
            )]))
            .expect("Transform failed")
            .outer_view(0)
            .expect("Outer view failed")
            .indices()
            .to_vec()
    }

    /// Finds the top-k matches for multiple needle strings.
    ///
    /// More efficient than calling [`find`](Self::find) repeatedly due to
    /// batched TF-IDF transformation and heap-based top-k selection.
    ///
    /// # Errors
    /// Returns an error if TF-IDF transformation fails.
    ///
    /// # Panics
    /// Panics if the TF-IDF transformation returns fewer rows than expected.
    pub fn find_many<'a>(
        &'a self,
        needles: impl Into<Vec<&'a str>>,
        top_k: usize,
    ) -> Result<Vec<Needle<'a>>, PreprocessingError> {
        let needles: Vec<&str> = needles.into();
        let needles_tfidf = self.fitted.transform(&Array1::from_iter(
            needles
                .iter()
                .map(|needle| Self::text_into_ngrams(needle, self.ngram_length)),
        ))?;
        let needles_norm = needles_tfidf.normalize();

        let mut results = Vec::with_capacity(needles.len());
        for (i, &needle) in needles.iter().enumerate() {
            let needle_vec: CsVecView<f64> = needles_tfidf.outer_view(i).unwrap();
            let q_norm = needles_norm[i];
            let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(top_k + 1);

            for (j, hay_vec) in self.haystack_tfidf.outer_iterator().enumerate() {
                let dot = needle_vec.dot(&hay_vec);
                let denom = q_norm * self.haystack_norm[j];
                let sim = if denom == 0.0 { 0.0 } else { dot / denom };
                let entry = Scored { sim, idx: j };

                if heap.len() < top_k {
                    heap.push(entry);
                } else if heap.peek().is_some_and(|min_entry| entry.sim > min_entry.sim) {
                    heap.pop();
                    heap.push(entry);
                }
            }

            let matches = heap
                .into_sorted_vec()
                .into_iter()
                .map(|scored| MatchEntry {
                    haystack: &self.haystack[scored.idx],
                    haystack_idx: scored.idx,
                    confidence: round_confidence(scored.sim),
                })
                .collect();

            results.push(Needle { needle, matches });
        }

        Ok(results)
    }
}
