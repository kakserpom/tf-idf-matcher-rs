use criterion::{Criterion, criterion_group, criterion_main};
use tf_idf_matcher::{Needle, TFIDFMatcher};

fn make_sample_data(n: usize) -> Vec<String> {
    let names = vec![
        "Joe Biden",
        "Donald Trump",
        "Barack Obama",
        "Angela Merkel",
        "Vladimir Putin",
        "Volodymyr Zelensky",
        "Emmanuel Macron",
        "Xi Jinping",
        "Narendra Modi",
    ];
    (0..n)
        .map(|i| format!("{} #{}", names[i % names.len()], i))
        .collect()
}

fn bench_tfidf(c: &mut Criterion) {
    let haystack = make_sample_data(20_000);
    let matcher = TFIDFMatcher::new(haystack, 3).unwrap();
    let needle = "putinvladimir";

    let mut group = c.benchmark_group("tfidf");

    group.sample_size(10_000);

    group.bench_function("TFIDFMatcher::find_one", |b| {
        b.iter(|| {
            let _res: Needle = matcher.find(needle, 5).unwrap();
        })
    });
}

criterion_group!(benches, bench_tfidf);
criterion_main!(benches);
