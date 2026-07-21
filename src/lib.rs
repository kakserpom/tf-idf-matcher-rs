#![warn(clippy::pedantic)]
use sprs::{CsMat, CsVecView};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::cmp::Ordering::Equal;
use std::collections::BinaryHeap;

mod vectorizer;
use vectorizer::Vectorizer;

#[cfg(test)]
mod tests;

/// Error type returned by [`TFIDFMatcher`] operations.
///
/// The current implementation is infallible, so this type is uninhabited: an `Err` value can never
/// be constructed. It is kept so the public API stays fallible (`Result`-returning), preserving
/// source compatibility with call sites that use `?` or `.expect()` and leaving room to introduce
/// real error conditions later without another breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatcherError {}

impl std::fmt::Display for MatcherError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {}
    }
}

impl std::error::Error for MatcherError {}
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
///
/// Retrieval uses an inverted index (`postings`: feature -> `[(doc, weight)]`) rather than a dense
/// scan of every document. A query touches only the documents that share at least one n-gram
/// feature with it, so cost scales with the query's posting lists — not the corpus size. This is
/// what keeps batch throughput from being bottlenecked on streaming the whole TF-IDF matrix per
/// query.
#[derive(Debug, Clone)]
pub struct TFIDFMatcher {
    haystack: Vec<String>,
    fitted: Vectorizer,
    /// Inverted index: `postings[feature]` is the list of `(document index, tf-idf weight)` for
    /// every document in which that feature (n-gram) occurs.
    postings: Vec<Vec<(u32, f64)>>,
    haystack_norm: Vec<f64>,
    n_docs: usize,
    ngram_length: usize,
}

/// Per-thread scratch for the sparse score accumulator, reused across queries so scoring allocates
/// nothing steady-state. `scores` is indexed by document; `touched` lists the documents given a
/// nonzero score this query, so only those are read back and reset (never the whole corpus).
#[derive(Default)]
struct ScoreScratch {
    scores: Vec<f64>,
    touched: Vec<u32>,
}

thread_local! {
    static SCRATCH: RefCell<ScoreScratch> = RefCell::new(ScoreScratch::default());
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
    /// Currently infallible; the `Result` is retained for API stability (see [`MatcherError`]).
    ///
    /// # Panics
    /// Panics if the corpus contains more than `u32::MAX` documents (the inverted index stores
    /// document indices as `u32`).
    pub fn new<T>(
        haystack: impl IntoIterator<Item = T>,
        ngram_length: usize,
    ) -> Result<Self, MatcherError>
    where
        T: Into<String>,
    {
        let haystack: Vec<String> = haystack.into_iter().map(Into::into).collect();
        let processed_haystack: Vec<String> = haystack
            .iter()
            .map(|s| Self::text_into_ngrams(s, ngram_length))
            .collect();

        let fitted = Vectorizer::fit(&processed_haystack);
        let haystack_tfidf = fitted.transform(&processed_haystack);
        let haystack_norm = haystack_tfidf.normalize();

        // Build the inverted index once from the doc-major TF-IDF matrix, then drop the matrix —
        // the postings hold the same nonzeros, transposed, so memory is unchanged.
        let n_docs = haystack_tfidf.rows();
        let mut postings: Vec<Vec<(u32, f64)>> = vec![Vec::new(); haystack_tfidf.cols()];
        for (doc, row) in haystack_tfidf.outer_iterator().enumerate() {
            for (feature, &weight) in row.iter() {
                postings[feature].push((u32::try_from(doc).expect("corpus exceeds u32"), weight));
            }
        }

        Ok(Self {
            haystack,
            fitted,
            postings,
            haystack_norm,
            n_docs,
            ngram_length,
        })
    }

    /// Score a query's sparse TF-IDF vector against the corpus via the inverted index and return the
    /// top-`top_k` `(document, cosine similarity)` matches, highest first. Only documents sharing a
    /// feature with the query are visited; the per-thread accumulator is reset in place afterwards.
    fn top_k_matches<'a>(
        &'a self,
        needle_v: CsVecView<f64>,
        q_norm: f64,
        top_k: usize,
    ) -> Vec<MatchEntry<'a>> {
        if top_k == 0 || q_norm == 0.0 {
            return Vec::new();
        }
        SCRATCH.with(|cell| {
            let ScoreScratch { scores, touched } = &mut *cell.borrow_mut();
            if scores.len() < self.n_docs {
                scores.resize(self.n_docs, 0.0);
            }
            // Accumulate dot products: for each query feature, add q_weight * d_weight to every
            // document carrying that feature, recording first-touch so the reset stays sparse.
            for (feature, &q_weight) in needle_v.iter() {
                let Some(list) = self.postings.get(feature) else {
                    continue;
                };
                for &(doc, d_weight) in list {
                    let score = &mut scores[doc as usize];
                    if *score == 0.0 {
                        touched.push(doc);
                    }
                    *score += q_weight * d_weight;
                }
            }

            let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(top_k + 1);
            for &doc in touched.iter() {
                let d = doc as usize;
                let denom = q_norm * self.haystack_norm[d];
                let sim = if denom == 0.0 { 0.0 } else { scores[d] / denom };
                scores[d] = 0.0; // reset in place; `touched` is cleared below
                let entry = Scored { sim, idx: d };
                if heap.len() < top_k {
                    heap.push(entry);
                } else if heap
                    .peek()
                    .is_some_and(|min_entry| entry.sim > min_entry.sim)
                {
                    heap.pop();
                    heap.push(entry);
                }
            }
            touched.clear();

            heap.into_sorted_vec()
                .into_iter()
                .map(|scored| MatchEntry {
                    haystack: &self.haystack[scored.idx],
                    haystack_idx: scored.idx,
                    confidence: round_confidence(scored.sim),
                })
                .collect()
        })
    }

    /// Finds the top-k matches for a single needle string.
    ///
    /// Returns a [`Needle`] containing the query and its ranked matches.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is retained for API stability (see [`MatcherError`]).
    ///
    /// # Panics
    /// Panics if the TF-IDF transformation returns an empty result (should not happen).
    pub fn find<'a>(&'a self, needle: &'a str, top_k: usize) -> Result<Needle<'a>, MatcherError> {
        let needle_ngrams = Self::text_into_ngrams(needle, self.ngram_length);
        let needles_tfidf = self.fitted.transform([needle_ngrams.as_str()]);
        let needle_v = needles_tfidf.outer_view(0).unwrap();
        let q_norm = needles_tfidf.normalize()[0];
        let matches = self.top_k_matches(needle_v, q_norm, top_k);
        Ok(Needle { needle, matches })
    }

    /// Returns the indices of active TF-IDF features for a needle.
    ///
    /// Useful for debugging and understanding which n-grams are matched.
    ///
    /// # Panics
    /// Panics if the TF-IDF transformation returns an empty result (should not happen).
    #[must_use]
    pub fn features(&self, needle: &str) -> Vec<usize> {
        let needle_ngrams = Self::text_into_ngrams(needle, self.ngram_length);
        self.fitted
            .transform([needle_ngrams.as_str()])
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
    /// Currently infallible; the `Result` is retained for API stability (see [`MatcherError`]).
    ///
    /// # Panics
    /// Panics if the TF-IDF transformation returns fewer rows than expected.
    pub fn find_many<'a>(
        &'a self,
        needles: impl Into<Vec<&'a str>>,
        top_k: usize,
    ) -> Result<Vec<Needle<'a>>, MatcherError> {
        let needles: Vec<&str> = needles.into();
        let needle_ngrams: Vec<String> = needles
            .iter()
            .map(|needle| Self::text_into_ngrams(needle, self.ngram_length))
            .collect();
        let needles_tfidf = self.fitted.transform(&needle_ngrams);
        let needles_norm = needles_tfidf.normalize();

        let mut results = Vec::with_capacity(needles.len());
        for (i, &needle) in needles.iter().enumerate() {
            let needle_vec: CsVecView<f64> = needles_tfidf.outer_view(i).unwrap();
            let matches = self.top_k_matches(needle_vec, needles_norm[i], top_k);
            results.push(Needle { needle, matches });
        }

        Ok(results)
    }
}
