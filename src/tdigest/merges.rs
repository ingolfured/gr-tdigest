use crate::tdigest::centroids::Centroid;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::iter::Peekable;

/// Merge stream that interleaves existing centroids with a run-length
/// encoding of sorted raw values (grouping identical values into one centroid).
pub(crate) struct MergeByMean<'a> {
    centroids: Peekable<std::slice::Iter<'a, Centroid>>,
    values: Peekable<std::slice::Iter<'a, f64>>,
    /// When pulling from values, group consecutive identical values into a single centroid.
    pending_value_run: Option<(f64, usize)>,
    singletons_on_values: bool,
}

impl<'a> MergeByMean<'a> {
    pub(crate) fn from_centroids_and_values(
        centroids: &'a [Centroid],
        values: &'a [f64],
        singletons_on_values: bool,
    ) -> Self {
        Self {
            centroids: centroids.iter().peekable(),
            values: values.iter().peekable(),
            pending_value_run: None,
            singletons_on_values,
        }
    }

    /// Drain a run of identical raw values into one centroid (pile).
    fn next_values_run(&mut self) -> Option<Centroid> {
        if let Some((val, len)) = self.pending_value_run.take() {
            return Some(if self.singletons_on_values {
                Centroid::new_singleton(val, len as f64)
            } else {
                Centroid::new_mixed(val, len as f64)
            });
        }
        let first = *self.values.next()?; // at least one value exists
        let mut len = 1usize;
        while let Some(&&v) = self.values.peek() {
            if v == first {
                self.values.next();
                len += 1;
            } else {
                break;
            }
        }
        Some(if self.singletons_on_values {
            Centroid::new_singleton(first, len as f64)
        } else {
            Centroid::new_mixed(first, len as f64)
        })
    }
}

impl<'a> Iterator for MergeByMean<'a> {
    type Item = Centroid;
    fn next(&mut self) -> Option<Self::Item> {
        // Always flush any stashed equal-mean value run before peeking again.
        if self.pending_value_run.is_some() {
            return self.next_values_run();
        }

        match (self.centroids.peek(), self.values.peek()) {
            (Some(c), Some(v)) => {
                if c.mean() < **v {
                    Some(*self.centroids.next().unwrap())
                } else if c.mean() > **v {
                    self.next_values_run()
                } else {
                    // Equal means: emit the existing centroid now, stash the *whole* values run.
                    let target = **v;
                    let mut len = 0usize;
                    while let Some(&&vv) = self.values.peek() {
                        if vv == target {
                            self.values.next();
                            len += 1;
                        } else {
                            break;
                        }
                    }
                    self.pending_value_run = Some((target, len));
                    Some(*self.centroids.next().unwrap())
                }
            }
            (Some(_), None) => Some(*self.centroids.next().unwrap()),
            (None, Some(_)) => self.next_values_run(),
            (None, None) => None,
        }
    }
}

/// k-way merge of centroid runs (by increasing mean).
pub(crate) struct KWayCentroidMerge<'a> {
    runs: Vec<&'a [Centroid]>,
    pos: Vec<usize>,
    heap: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)>,
}

impl<'a> KWayCentroidMerge<'a> {
    pub(crate) fn new(runs: Vec<&'a [Centroid]>) -> Self {
        let mut heap = BinaryHeap::with_capacity(runs.len());
        let pos = vec![0; runs.len()];
        for (i, r) in runs.iter().enumerate() {
            if let Some(c) = r.first() {
                heap.push((Reverse(OrderedFloat::from(c.mean())), i));
            }
        }
        Self { runs, pos, heap }
    }
}

impl<'a> Iterator for KWayCentroidMerge<'a> {
    type Item = Centroid;
    fn next(&mut self) -> Option<Self::Item> {
        let (Reverse(_m), run) = self.heap.pop()?;
        let i = self.pos[run];
        let slice = self.runs[run];
        let c = slice[i];
        self.pos[run] += 1;
        if self.pos[run] < slice.len() {
            let n = slice[self.pos[run]];
            self.heap
                .push((Reverse(OrderedFloat::from(n.mean())), run));
        }
        Some(c)
    }
}
