//! A small, self-contained TF-IDF vectorizer.
//!
//! This replaces the previous dependency on `linfa-preprocessing`. The matcher only ever used a
//! whitespace tokenizer with a `(1, 1)` n-gram range (the n-gram construction lives in
//! [`crate::TFIDFMatcher::text_into_ngrams`]), so the vectorizer boils down to: build a
//! token -> feature-index vocabulary, then map documents to a sparse TF-IDF matrix.
//!
//! The numerics reproduce linfa's defaults exactly so scores are unchanged:
//! * document strings are NFKD-normalized then lowercased before tokenizing (`normalize` +
//!   `convert_to_lowercase`);
//! * the IDF uses the `Smooth` method, `ln((1 + n) / (1 + df)) + 1`;
//! * crucially, IDF is recomputed per [`transform`](Vectorizer::transform) call from the batch
//!   being transformed (both `n` and the document frequencies come from that batch), never stored
//!   at fit time. Only the vocabulary is retained across calls — matching linfa's
//!   `FittedTfIdfVectorizer`.
//!
//! Because everything is owned plain data (`HashMap`, `Vec`), the vectorizer — and therefore
//! [`crate::TFIDFMatcher`] — is `Send + Sync`, which is what lets the matcher be shared across
//! threads.

use sprs::{CompressedStorage, CsMat, CsVec};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// Smooth inverse document frequency, matching linfa's `TfIdfMethod::Smooth`.
// Document counts and frequencies are small integers; f64 represents them exactly here.
#[allow(clippy::cast_precision_loss)]
#[inline]
fn smooth_idf(n: usize, df: usize) -> f64 {
    ((1.0 + n as f64) / (1.0 + df as f64)).ln() + 1.0
}

/// NFKD-normalize then lowercase, matching linfa's `transform_string` with both `normalize` and
/// `convert_to_lowercase` enabled (the defaults the matcher relied on).
fn normalize_document(s: &str) -> String {
    s.nfkd().collect::<String>().to_lowercase()
}

/// A fitted TF-IDF vectorizer holding only the learned vocabulary.
#[derive(Debug, Clone)]
pub(crate) struct Vectorizer {
    /// Maps each token to its feature index. Insertion order defines the index; the exact ordering
    /// is irrelevant to cosine similarity as long as `fit` and `transform` agree.
    vocabulary: HashMap<String, usize>,
}

impl Vectorizer {
    /// Learns a vocabulary from `docs`. Each document is normalized and split on whitespace; every
    /// distinct token becomes a feature.
    pub(crate) fn fit<I, S>(docs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut vocabulary: HashMap<String, usize> = HashMap::new();
        for doc in docs {
            let normalized = normalize_document(doc.as_ref());
            for token in normalized.split_whitespace() {
                let next = vocabulary.len();
                vocabulary.entry(token.to_owned()).or_insert(next);
            }
        }
        Self { vocabulary }
    }

    /// Number of learned features (vocabulary size).
    pub(crate) fn n_features(&self) -> usize {
        self.vocabulary.len()
    }

    /// Transforms `docs` into a `(n_docs, n_features)` sparse TF-IDF matrix (CSR).
    ///
    /// The IDF is derived from this batch: `n` is the number of documents passed in and each
    /// feature's document frequency is counted over those same documents.
    pub(crate) fn transform<I, S>(&self, docs: I) -> CsMat<f64>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let n_features = self.n_features();

        // Dense scratch for per-document term counts, reset sparsely via `touched`.
        let mut counts = vec![0usize; n_features];
        let mut doc_freqs = vec![0usize; n_features];
        // Raw term-frequency rows (sorted feature index -> count), scaled by IDF once the whole
        // batch has been seen.
        let mut rows: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();

        for doc in docs {
            let normalized = normalize_document(doc.as_ref());
            let mut touched: Vec<usize> = Vec::new();
            for token in normalized.split_whitespace() {
                if let Some(&idx) = self.vocabulary.get(token) {
                    if counts[idx] == 0 {
                        touched.push(idx);
                    }
                    counts[idx] += 1;
                }
            }
            touched.sort_unstable();

            let mut indices = Vec::with_capacity(touched.len());
            let mut values = Vec::with_capacity(touched.len());
            for &idx in &touched {
                indices.push(idx);
                // Term frequency is a small count; the f64 cast is exact in practice.
                #[allow(clippy::cast_precision_loss)]
                values.push(counts[idx] as f64);
                doc_freqs[idx] += 1;
                counts[idx] = 0; // reset in place; `touched` is dropped next iteration
            }
            rows.push((indices, values));
        }

        let n_docs = rows.len();
        let mut matrix = CsMat::empty(CompressedStorage::CSR, n_features);
        matrix.reserve_outer_dim_exact(n_docs);
        for (indices, mut values) in rows {
            for (value, &col) in values.iter_mut().zip(indices.iter()) {
                *value *= smooth_idf(n_docs, doc_freqs[col]);
            }
            // `indices` is sorted ascending, satisfying CsVec's ordering invariant.
            let row = CsVec::new(n_features, indices, values);
            matrix = matrix.append_outer_csvec(row.view());
        }
        matrix
    }
}
