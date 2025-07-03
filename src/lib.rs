#![warn(clippy::pedantic)]
use itertools::Itertools;
use linfa_preprocessing::PreprocessingError;
use linfa_preprocessing::tf_idf_vectorization::{FittedTfIdfVectorizer, TfIdfVectorizer};
use ndarray::Array1;
use sprs::CsMat;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(test)]
mod tests;

fn preprocess(text: &str, n: usize) -> String {
    let mut chars: Vec<char> = text
        .to_lowercase()
        .split_whitespace()
        .join("_")
        .chars()
        .collect();
    chars.insert(0, '_');
    chars.push('_');
    if chars.len() < n {
        return String::new();
    }
    (0..=chars.len() - n)
        .filter_map(|i| {
            if chars[i + 1..i + n - 1].contains(&'_') {
                None
            } else {
                Some(String::from_iter(&chars[i..i + n]))
            }
        })
        .join(" ")
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
    ngram_length: usize,
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
        // на случай NaN
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
        let processed_haystack: Vec<String> = haystack
            .iter()
            .map(|s| preprocess(&s, ngram_length))
            .collect();

        let fitted = TfIdfVectorizer::default()
            .convert_to_lowercase(false)
            .fit::<String, _>(&Array1::from_vec(processed_haystack.clone()))
            .expect("TF-IDF training failed");

        let haystack_tfidf = fitted.transform(&Array1::from_vec(processed_haystack))?;
        Ok(Self {
            haystack,
            ngram_length,
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
            .transform(&Array1::from(vec![preprocess(needle, self.ngram_length)]))
            .expect("Transform failed")
            .outer_view(0)
            .expect("Outer view failed")
            .indices()
            .to_vec()
    }

    /// Find many needles
    pub fn find_many<'a>(
        &'a self,
        needles: impl Into<Vec<&'a str>>,
        top_k: usize,
    ) -> Result<Vec<Needle<'a>>, PreprocessingError> {
        let needles: Vec<&str> = needles.into();
        let needles_tfidf = self.fitted.transform(&Array1::from_iter(
            needles.iter().map(|s| preprocess(s, self.ngram_length)),
        ))?;
        let needles_norm = needles_tfidf.normalize();
        let sim_matrix = &needles_tfidf * &self.haystack_tfidf.transpose_view();

        Ok(needles
            .into_iter()
            .zip(sim_matrix.outer_iterator().enumerate())
            .map(|(needle, (i, row))| {
                let q_norm = needles_norm[i];
                let mut heap: BinaryHeap<Scored> = BinaryHeap::new();
                for (col_idx, &dot_val) in row.iter() {
                    let denom = q_norm * self.haystack_norm[col_idx];
                    let sim = if denom == 0.0 { 0.0 } else { dot_val / denom };
                    heap.push(Scored { sim, idx: col_idx });
                }
                
                let mut matches = Vec::with_capacity(top_k);
                for _ in 0..top_k {
                    if let Some(scored) = heap.pop() {
                        matches.push(MatchEntry {
                            haystack: &self.haystack[scored.idx],
                            haystack_idx: scored.idx,
                            confidence: (scored.sim * 100.0).round() / 100.0,
                        });
                    } else {
                        break;
                    }
                }

                Needle { needle, matches }
            })
            .collect())
    }
}
