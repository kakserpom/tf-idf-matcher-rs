#![warn(clippy::pedantic)]
use itertools::Itertools;
use linfa_preprocessing::PreprocessingError;
use linfa_preprocessing::tf_idf_vectorization::{FittedTfIdfVectorizer, TfIdfVectorizer};
use ndarray::Array1;
use sprs::CsMat;
use std::cmp::Ordering::Equal;

#[cfg(test)]
mod tests;

fn preprocess(text: &str, n: usize) -> String {
    let chars: Vec<char> = text
        .to_lowercase()
        .split_whitespace()
        .join("_")
        .chars()
        .collect();
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
#[derive(Debug, Clone)]
pub struct TFIDFMatcher {
    haystack: Vec<String>,
    ngram_length: usize,
    fitted: FittedTfIdfVectorizer,
    haystack_tfidf: CsMat<f64>,
    haystack_norm: Vec<f64>,
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

        let haystack_norm: Vec<f64> = haystack_tfidf
            .outer_iterator()
            .map(|row| row.data().iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();

        Ok(Self {
            haystack: haystack.into_iter().collect(),
            ngram_length,
            fitted,
            haystack_tfidf,
            haystack_norm,
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
        needles: Vec<&'a str>,
        top_k: usize,
    ) -> Result<Vec<Needle<'a>>, PreprocessingError> {
        let needles_tfidf = self.fitted.transform(&Array1::from_iter(
            needles.iter().map(|s| preprocess(s, self.ngram_length)),
        ))?;

        let needles_norm: Vec<f64> = needles_tfidf
            .outer_iterator()
            .map(|row| row.data().iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect();

        let sim_matrix = &needles_tfidf * &self.haystack_tfidf.transpose_view();

        Ok(needles
            .into_iter()
            .zip(sim_matrix.outer_iterator().enumerate())
            .map(|(needle, (i, row))| {
                let q_norm = needles_norm[i];

                let mut similarities: Vec<(usize, f64)> = row
                    .iter()
                    .map(|dot| {
                        let (col_idx, dot_val) = dot;
                        let denom = q_norm * self.haystack_norm[col_idx];
                        let sim = if denom == 0.0 { 0.0 } else { dot_val / denom };
                        (col_idx, sim)
                    })
                    .collect();
                let k = top_k.min(similarities.len());
                let matches = if k > 0 {
                    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Equal));
                    let top_k = &similarities[..k];
                    top_k
                        .iter()
                        .map(|(idx, sim)| MatchEntry {
                            haystack: &self.haystack[*idx],
                            haystack_idx: *idx,
                            confidence: (*sim * 100.0).round() / 100.0,
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                Needle { needle, matches }
            })
            .collect())
    }
}
