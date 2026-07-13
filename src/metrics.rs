//! Shared eval metrics: ROUGE-1/ROUGE-L and the LLM judge.
//! Used by both the `eval` (single-conv) and `stress` (merged-store) binaries
//! so their numbers are directly comparable.

use crate::ollama::{ChatMessage, Ollama};

pub fn metric_tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            cur.push(ch.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n {
        return Vec::new();
    }
    (0..=tokens.len() - n).map(|i| tokens[i..i + n].to_vec()).collect()
}

fn f1(overlap: usize, pred_len: usize, gold_len: usize) -> f64 {
    if overlap == 0 {
        return 0.0;
    }
    let p = overlap as f64 / pred_len as f64;
    let r = overlap as f64 / gold_len as f64;
    2.0 * p * r / (p + r)
}

/// ROUGE-N F1 (multiset n-gram overlap).
pub fn rouge_n(pred: &str, gold: &str, n: usize) -> f64 {
    let pt = ngrams(&metric_tokenize(pred), n);
    let gt = ngrams(&metric_tokenize(gold), n);
    if pt.is_empty() || gt.is_empty() {
        return 0.0;
    }
    let mut gcount: std::collections::HashMap<&Vec<String>, usize> = Default::default();
    for g in &gt {
        *gcount.entry(g).or_insert(0) += 1;
    }
    let mut overlap = 0usize;
    for p in &pt {
        if let Some(c) = gcount.get_mut(p) {
            if *c > 0 {
                *c -= 1;
                overlap += 1;
            }
        }
    }
    f1(overlap, pt.len(), gt.len())
}

fn lcs_len(a: &[String], b: &[String]) -> usize {
    let mut dp = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        let mut prev = 0;
        for j in 1..=b.len() {
            let tmp = dp[j];
            if a[i - 1] == b[j - 1] {
                dp[j] = prev + 1;
            } else {
                dp[j] = dp[j].max(dp[j - 1]);
            }
            prev = tmp;
        }
    }
    dp[b.len()]
}

/// ROUGE-L F1 (longest common subsequence).
pub fn rouge_l(pred: &str, gold: &str) -> f64 {
    let p = metric_tokenize(pred);
    let g = metric_tokenize(gold);
    if p.is_empty() || g.is_empty() {
        return 0.0;
    }
    f1(lcs_len(&p, &g), p.len(), g.len())
}

/// LLM-as-judge: does the prediction convey the gold answer? YES/NO, temp 0.
pub fn judge(ollama: &Ollama, question: &str, gold: &str, pred: &str) -> Option<bool> {
    let prompt = format!(
        "Question: {question}\nGold answer: {gold}\nModel answer: {pred}\n\n\
         Does the model answer convey the same key information as the gold answer? \
         Minor wording/format differences are fine. Reply with exactly one word: YES or NO."
    );
    let msgs = [
        ChatMessage::new("system", "You are a strict but fair grader. Reply only YES or NO."),
        ChatMessage::new("user", prompt),
    ];
    let resp = ollama.chat(&msgs, 2048, 5).ok()?;
    Some(resp.to_uppercase().contains("YES"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rouge_basics() {
        assert!((rouge_l("7 May 2023", "7 May 2023") - 1.0).abs() < 1e-9);
        assert_eq!(rouge_l("", "x"), 0.0);
        assert!(rouge_n("the cat sat", "the cat", 1) > 0.7);
    }
}
