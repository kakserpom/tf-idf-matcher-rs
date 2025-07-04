#![warn(clippy::pedantic)]
use itertools::Itertools;
use linfa_preprocessing::tf_idf_vectorization::{FittedTfIdfVectorizer, TfIdfVectorizer};
use linfa_preprocessing::{PreprocessingError, Tokenizer};
use ndarray::Array1;
use sprs::{CsMat, CsVecView};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

#[cfg(test)]
mod tests;

fn preprocess(text: &str) -> String {
    format!(" {} ", text.split_whitespace().join(" "))
}

fn tokenize(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut iter = text.char_indices().peekable();
    while let Some((start, _)) = iter.next() {
        let end = iter.peek().map_or(text.len(), |&(i, _)| i);
        tokens.push(&text[start..end]);
    }
    tokens
}

#[derive(Debug)]
pub struct MatchEntry<'a> {
    pub haystack: &'a str,
    pub confidence: f64,
    pub haystack_idx: usize,
}

#[derive(Debug)]
pub struct Needle<'a> {
    pub needle: &'a str,
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

#[derive(Debug, Clone)]
pub struct TFIDFMatcher {
    haystack: Vec<String>,
    fitted: FittedTfIdfVectorizer,
    haystack_tfidf: CsMat<f64>,
    haystack_norm: Vec<f64>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Scored {
    sim: f64,
    idx: usize,
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.sim.partial_cmp(&other.sim)
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl TFIDFMatcher {
    pub fn new<T>(
        haystack: impl IntoIterator<Item = T>,
        ngram_length: usize,
    ) -> Result<Self, PreprocessingError>
    where
        T: Into<String>,
    {
        let haystack: Vec<String> = haystack.into_iter().map(Into::into).collect();
        let processed_haystack: Vec<String> = haystack.iter().map(|s| preprocess(&s)).collect();

        let fitted = TfIdfVectorizer::default()
            .convert_to_lowercase(true)
            .tokenizer(Tokenizer::Function(tokenize))
            .n_gram_range(ngram_length, ngram_length)
            .fit::<String, _>(&Array1::from_vec(processed_haystack.clone()))
            .expect("TF-IDF training failed");

        let haystack_tfidf = fitted.transform(&Array1::from_vec(processed_haystack))?;
        Ok(Self {
            haystack,
            fitted,
            haystack_norm: haystack_tfidf.normalize(),
            haystack_tfidf,
        })
    }

    /// Find one needle
    pub fn find_one<'a>(
        &'a self,
        needle: &'a str,
        top_k: usize,
    ) -> Result<Needle<'a>, PreprocessingError> {
        Ok(self
            .find_many(vec![needle], top_k)?
            .into_iter()
            .next()
            .unwrap())
    }

    /// Get active TF-IDF features for a needle
    pub fn features(&self, needle: &str) -> Vec<usize> {
        self.fitted
            .transform(&Array1::from(vec![preprocess(needle)]))
            .expect("Transform failed")
            .outer_view(0)
            .expect("Outer view failed")
            .indices()
            .to_vec()
    }

    /// Find many needles using manual dot-products (no full matmul) and a min-heap for top_k largest
    pub fn find_many<'a>(
        &'a self,
        needles: impl Into<Vec<&'a str>>,
        top_k: usize,
    ) -> Result<Vec<Needle<'a>>, PreprocessingError> {
        let needles: Vec<&str> = needles.into();
        let needles_tfidf = self
            .fitted
            .transform(&Array1::from_iter(needles.iter().map(|s| preprocess(s))))?;
        let needles_norm = needles_tfidf.normalize();

        let mut results = Vec::with_capacity(needles.len());
        for (i, &needle) in needles.iter().enumerate() {
            let needle_vec: CsVecView<f64> = needles_tfidf.outer_view(i).unwrap();
            let q_norm = needles_norm[i];
            let mut heap: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(top_k + 1);

            for (j, hay_vec) in self.haystack_tfidf.outer_iterator().enumerate() {
                let dot = needle_vec.dot(&hay_vec);
                let denom = q_norm * self.haystack_norm[j];
                let sim = if denom == 0.0 { 0.0 } else { dot / denom };
                let entry = Scored { sim, idx: j };

                if heap.len() < top_k {
                    heap.push(Reverse(entry));
                } else if let Some(Reverse(min_entry)) = heap.peek() {
                    if entry.sim > min_entry.sim {
                        heap.pop();
                        heap.push(Reverse(entry));
                    }
                }
            }

            let matches = heap
                .into_sorted_vec()
                .into_iter()
                .map(|Reverse(scored)| MatchEntry {
                    haystack: &self.haystack[scored.idx],
                    haystack_idx: scored.idx,
                    confidence: (scored.sim * 100.0).round() / 100.0,
                })
                .collect();

            results.push(Needle { needle, matches });
        }

        Ok(results)
    }
}
