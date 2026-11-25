//! Guesses at the next few tokens, for the model to check in one sweep.
//!
//! Generation one token at a time reads every weight once per token. If the next few
//! tokens can be guessed, they can be fed as one batch behind the token just chosen: each
//! position's scores then say what the model would have chosen there, so the guesses are
//! checked one after another, and the first wrong one and everything after it is undone
//! (`Inkling::undo`). The text produced is exactly what generation without guesses
//! produces; only the number of sweeps changes. A right guess is a token for a fraction of
//! a sweep; a wrong guess costs the part of the sweep that was spent on it.

use crate::error::ModelError;

/// Something that guesses how a text goes on. `V` is the type of the model's values (see
/// `Ops`), for a drafter that looks at what the model computed; one that only looks at the
/// text works with any.
pub trait Drafter<V> {
    /// Up to `most` tokens that might follow `tokens` (the prompt and everything generated
    /// so far), most likely continuation first to last. No guess is an empty list.
    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32>;

    /// Which of the model's intermediate values the drafter wants to see: the running sum
    /// after this many layer applications, for each number listed, in ascending order.
    /// None by default.
    fn wants(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Called after every batch the model is fed, with the wanted values of that batch (one
    /// per number in `wants`, each holding every position of the batch). `first` is the
    /// position of the batch's first token (0 starts a new text) and `kept` how many of
    /// its positions stay fed; the rest are guesses that were wrong and have been taken back.
    fn fed(&mut self, _first: usize, _kept: usize, _states: &[V]) -> Result<(), ModelError> {
        Ok(())
    }
}

/// Never guesses.
pub struct NoDraft;

impl<V> Drafter<V> for NoDraft {
    fn draft(&mut self, _tokens: &[u32], _most: usize) -> Vec<u32> {
        Vec::new()
    }
}

/// Guesses that the text goes on as it did the last time it read like this: it looks for
/// the most recent earlier place where the last few tokens appeared, and proposes what
/// followed them there. Text that repeats itself (code, lists, quoted passages, an answer
/// that restates the question) is guessed well; text that does not is not guessed at all.
/// Each guess scans the text once per run length tried.
pub struct LookupDrafter {
    /// The longest run of last tokens to look for, tried first.
    pub longest: usize,
    /// The shortest run that still counts as "reads like this".
    pub shortest: usize,
}

impl LookupDrafter {
    /// Looks for the last `longest` tokens first, then for fewer, down to `shortest`.
    /// Both are at least 1.
    pub fn new(longest: usize, shortest: usize) -> Self {
        let shortest = shortest.max(1);
        LookupDrafter { longest: longest.max(shortest), shortest }
    }
}

impl<V> Drafter<V> for LookupDrafter {
    fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
        if most == 0 {
            return Vec::new();
        }
        for run in (self.shortest..=self.longest).rev() {
            if tokens.len() <= run {
                continue;
            }
            let wanted = &tokens[tokens.len() - run..];
            // The most recent earlier place, which must be followed by at least one token
            // that is not part of the run being matched against itself.
            let found = (0..tokens.len() - run).rev().find(|&start| &tokens[start..start + run] == wanted);
            if let Some(start) = found {
                let after = start + run;
                return tokens[after..tokens.len().min(after + most)].to_vec();
            }
        }
        Vec::new()
    }
}

/// What guessing did during one generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftStats {
    /// Sweeps through the model after the prompt.
    pub sweeps: usize,
    /// Guessed tokens that were fed.
    pub drafted: usize,
    /// Guessed tokens that were what the model chose.
    pub accepted: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A drafter used for the text alone: no model values.
    struct Text<'a, D: Drafter<()>>(&'a mut D);
    impl<D: Drafter<()>> Text<'_, D> {
        fn draft(&mut self, tokens: &[u32], most: usize) -> Vec<u32> {
            self.0.draft(tokens, most)
        }
    }

    #[test]
    fn lookup_proposes_what_followed_the_most_recent_longest_match() {
        let mut d = LookupDrafter::new(3, 1);
        // "1 2 3" appeared before, followed by 4 5.
        let mut d = Text(&mut d);
        assert_eq!(d.draft(&[1, 2, 3, 4, 5, 9, 1, 2, 3], 2), [4, 5]);
        // The proposal stops at the end of the text, and is cut to `most`.
        assert_eq!(d.draft(&[1, 2, 3, 4, 5, 9, 1, 2, 3], 10), [4, 5, 9, 1, 2, 3]);
        assert_eq!(d.draft(&[1, 2, 3, 4, 5, 9, 1, 2, 3], 1), [4]);
        // A longer match wins over a more recent shorter one: "7 8" then 1, but "8" then 2 later.
        assert_eq!(d.draft(&[7, 8, 1, 5, 8, 2, 7, 8], 1), [1]);
        // Among matches of the same length the most recent wins.
        assert_eq!(d.draft(&[6, 1, 6, 2, 6], 1), [2]);
        // A token that repeats matches itself one place back.
        assert_eq!(d.draft(&[4, 4, 4], 2), [4]);
    }

    #[test]
    fn lookup_proposes_nothing_without_a_match_or_room() {
        let mut d = LookupDrafter::new(3, 2);
        let mut d = Text(&mut d);
        assert!(d.draft(&[1, 2, 3, 4, 5], 4).is_empty());
        // One matching token is below the shortest run asked for.
        assert!(d.draft(&[5, 1, 2, 5], 4).is_empty());
        assert!(d.draft(&[1, 2, 9, 1, 2], 0).is_empty());
        assert!(d.draft(&[], 4).is_empty());
        assert!(d.draft(&[1], 4).is_empty());
        assert!(Text(&mut NoDraft).draft(&[1, 2, 1, 2], 4).is_empty());
    }
}
