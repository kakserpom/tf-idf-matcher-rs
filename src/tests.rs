use super::*;
/*
#[test]
fn test_preprocess() {
    let result = preprocess("abcde", 2);
    assert_eq!(result, "_a ab bc cd de e_");

    let result = preprocess("abc de", 2);
    assert_eq!(result, "_a ab bc c_ _d de e_");

    let result = preprocess("lets get rusty", 3);
    assert_eq!(result, "_le let ets ts_ _ge get et_ _ru rus ust sty ty_");
}

#[test]
fn test_ngrams_shorter_than_n() {
    assert_eq!(preprocess("a", 2), "_a a_");
}

#[test]
fn test_preprocess_lowercase_and_join() {
    let result = preprocess("AbCd", 2);
    assert_eq!(result, "_a ab bc cd d_");
}*/

#[test]
fn test_tfidf_matcher_find_one() {
    let matcher =
        TFIDFMatcher::new(["testddd", "testing", "example"], 3).expect("Failed to create matcher");
    let result = matcher.find_one("testddd", 2).expect("find_one failed");

    println!("{:?}", result);
    // The top match for "test" should be itself with 100.0 confidence
    assert!(!result.matches.is_empty());
    assert_eq!(result.matches[0].haystack, "testddd");
    assert_eq!(result.matches[0].haystack_idx, 0);
    assert!(result.matches[0].confidence >= 0.99);

    // The second match should be "testing" with confidence greater than 0 and less than 100
    let second = &result.matches[1];
    assert_eq!(second.haystack, "testing");
    assert!(second.confidence > 0.);
    assert!(second.confidence < 1.);
}

#[test]
fn test_tfidf_matcher_find_many() {
    let matcher =
        TFIDFMatcher::new(["test", "testing", "example"], 3).expect("Failed to create matcher");
    let results = matcher
        .find_many(["test", "example"], 1)
        .expect("find_many failed");

    // There should be one match per needle
    assert_eq!(results.len(), 2);

    // First needle "test" matches "test"
    assert_eq!(results[0].needle, "test");
    let first_match = &results[0].matches[0];
    assert_eq!(first_match.haystack, "test");
    assert!(first_match.confidence >= 0.99);

    // Second needle "example" matches "example"
    assert_eq!(results[1].needle, "example");
    let second_match = &results[1].matches[0];
    assert_eq!(second_match.haystack, "example");
    assert!((second_match.confidence - 1.).abs() < 1e-8);
}

#[test]
fn test_features_count() {
    let matcher = TFIDFMatcher::new(["test", "testing"], 2).expect("Failed to create matcher");
    let feature_indices = matcher.features("test");
    // "test" has 5 bigrams: "_t", "te", "es", "st", "t_"
    println!("{:?}", feature_indices);
    assert_eq!(feature_indices.len(), 5);
}

#[test]
fn test_bench() {
    let matcher = TFIDFMatcher::new(
        [
            "Joe Biden",
            "Donald Trump",
            "Barack Obama",
            "Angela Merkel",
            "Vladimir Putin",
            "Volodymyr Zelensky",
            "Emmanuel Macron",
            "Xi Jinping",
            "Narendra Modi",
        ],
        3,
    )
    .unwrap();
    let result = matcher
        .find_one("putinvladimir", 2)
        .expect("find_one failed");
    println!("{:?}", result);
    assert_eq!(result.matches[0].haystack, "Vladimir Putin");
}
