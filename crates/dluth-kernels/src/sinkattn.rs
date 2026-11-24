//! Attention for one query token, with two additions to the textbook form: a per-key bias
//! (used for relative position) and a "sink", one extra score per head that carries no
//! value and so only absorbs attention.

use crate::cpu::softmax_inplace;

/// One query token's view of attention.
///
/// `keys` and `values` hold, for each of `positions` cached tokens, `kv_heads` vectors of
/// `head_dim`: layout `[position][kv_head][head_dim]`. `bias` holds one number per query
/// head and position, `[head][position]`. `sinks` holds one score per query head.
pub struct Query<'a> {
    pub q: &'a [f32],
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub bias: &'a [f32],
    pub sinks: &'a [f32],
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub positions: usize,
}

/// Grouped-query attention with a bias and a sink.
///
/// For each query head: the score of a position is the dot product of query and key
/// divided by `sqrt(head_dim)`, plus that position's bias. The softmax runs over all
/// positions and the sink's score together. The result is the weighted sum of the values;
/// the sink has no value, so whatever weight it takes is simply missing from the sum.
/// Query head `h` reads key-value head `h / (heads / kv_heads)`.
pub fn attend(a: Query, out: &mut [f32]) {
    let stride = a.kv_heads * a.head_dim;
    assert!(
        a.positions >= 1
            && a.head_dim >= 1
            && a.kv_heads >= 1
            && a.heads.is_multiple_of(a.kv_heads)
            && a.q.len() == a.heads * a.head_dim
            && out.len() == a.q.len()
            && a.keys.len() == a.positions * stride
            && a.values.len() == a.keys.len()
            && a.bias.len() == a.heads * a.positions
            && a.sinks.len() == a.heads,
        "sinkattn::attend: q {}, out {}, keys {}, values {}, bias {}, sinks {} for {} heads, {} kv heads of {}, {} positions",
        a.q.len(),
        out.len(),
        a.keys.len(),
        a.values.len(),
        a.bias.len(),
        a.sinks.len(),
        a.heads,
        a.kv_heads,
        a.head_dim,
        a.positions
    );
    let group = a.heads / a.kv_heads;
    let scale = 1.0 / (a.head_dim as f32).sqrt();
    // One slot per position, then the sink.
    let mut weights = vec![0.0f32; a.positions + 1];
    for h in 0..a.heads {
        let kv_head = h / group;
        let query = &a.q[h * a.head_dim..(h + 1) * a.head_dim];
        let bias = &a.bias[h * a.positions..(h + 1) * a.positions];
        for (s, weight) in weights[..a.positions].iter_mut().enumerate() {
            let at = s * stride + kv_head * a.head_dim;
            let key = &a.keys[at..at + a.head_dim];
            *weight = query.iter().zip(key).map(|(x, y)| x * y).sum::<f32>() * scale + bias[s];
        }
        weights[a.positions] = a.sinks[h];
        softmax_inplace(&mut weights);
        let result = &mut out[h * a.head_dim..(h + 1) * a.head_dim];
        result.fill(0.0);
        for (s, &weight) in weights[..a.positions].iter().enumerate() {
            let at = s * stride + kv_head * a.head_dim;
            for (o, &value) in result.iter_mut().zip(&a.values[at..at + a.head_dim]) {
                *o += weight * value;
            }
        }
    }
}

/// One head's relative-position profile: `out[d] = sum_k v[k] * table[k][d]`.
///
/// `table` has one row per element of `v` and `reach` columns, one per distance. Only the
/// first `out.len()` distances are computed.
pub fn distance_profile(v: &[f32], table: &[f32], reach: usize, out: &mut [f32]) {
    assert!(
        table.len() == v.len() * reach && out.len() <= reach,
        "sinkattn::distance_profile: {} table values for {} rows of {reach}, {} outputs",
        table.len(),
        v.len(),
        out.len()
    );
    out.fill(0.0);
    // A reach of 0 means an empty table (checked above); a chunk size of 0 is not allowed.
    for (row, &x) in table.chunks_exact(reach.max(1)).zip(v) {
        for (o, &t) in out.iter_mut().zip(row) {
            *o += x * t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // Two query heads sharing one key-value head of size 2, three cached positions.
        let q = vec![1.0, 0.0, 0.0, 2.0];
        let keys = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let values = vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0];
        (q, keys, values)
    }

    fn shape<'a>(q: &'a [f32], keys: &'a [f32], values: &'a [f32], bias: &'a [f32], sinks: &'a [f32]) -> Query<'a> {
        Query { q, keys, values, bias, sinks, heads: 2, kv_heads: 1, head_dim: 2, positions: 3 }
    }

    #[test]
    fn a_hopeless_sink_and_no_bias_is_ordinary_attention() {
        let (q, keys, values) = sample();
        let mut out = [0.0f32; 4];
        attend(shape(&q, &keys, &values, &[0.0; 6], &[-1e30, -1e30]), &mut out);
        // Head 0 scores: [1, 0, 1] / sqrt(2).
        let r = 2.0f32.sqrt();
        let e = [(1.0 / r).exp(), 1.0, (1.0 / r).exp()];
        let z: f32 = e.iter().sum();
        let want0 = (e[0] * 1.0 + e[1] * 2.0 + e[2] * 3.0) / z;
        assert!((out[0] - want0).abs() < 1e-5 && (out[1] - 10.0 * want0).abs() < 1e-4, "{out:?}");
        // Head 1 scores: [0, 2, 2] / sqrt(2).
        let e = [1.0, (2.0 / r).exp(), (2.0 / r).exp()];
        let z: f32 = e.iter().sum();
        let want1 = (e[0] * 1.0 + e[1] * 2.0 + e[2] * 3.0) / z;
        assert!((out[2] - want1).abs() < 1e-5 && (out[3] - 10.0 * want1).abs() < 1e-4, "{out:?}");
    }

    #[test]
    fn the_sink_takes_weight_and_gives_nothing_back() {
        let (q, keys, values) = sample();
        // All real scores equal (zero queries). A sink score of 0 is then a fourth equal share.
        let q0 = vec![0.0; 4];
        let mut out = [0.0f32; 4];
        attend(shape(&q0, &keys, &values, &[0.0; 6], &[0.0, 0.0]), &mut out);
        assert!((out[0] - (1.0 + 2.0 + 3.0) / 4.0).abs() < 1e-6, "{out:?}");
        // A dominant sink leaves almost nothing.
        attend(shape(&q, &keys, &values, &[0.0; 6], &[50.0, 50.0]), &mut out);
        assert!(out.iter().all(|v| v.abs() < 1e-15), "{out:?}");
    }

    #[test]
    fn the_bias_is_added_per_head_and_per_key() {
        let (_, keys, values) = sample();
        let q0 = vec![0.0; 4];
        // Head 0 is pushed onto position 2, head 1 onto position 0.
        let bias = [-1e30, -1e30, 0.0, 0.0, -1e30, -1e30];
        let mut out = [0.0f32; 4];
        attend(shape(&q0, &keys, &values, &bias, &[-1e30, -1e30]), &mut out);
        assert_eq!(out, [3.0, 30.0, 1.0, 10.0]);
    }

    #[test]
    fn query_heads_read_their_group_of_key_value_heads() {
        // Four query heads, two key-value heads of size 1, one position.
        let q = [1.0, 1.0, 1.0, 1.0];
        let (keys, values) = ([0.0, 0.0], [5.0, 7.0]);
        let mut out = [0.0f32; 4];
        let query = Query {
            q: &q,
            keys: &keys,
            values: &values,
            bias: &[0.0; 4],
            sinks: &[-1e30; 4],
            heads: 4,
            kv_heads: 2,
            head_dim: 1,
            positions: 1,
        };
        attend(query, &mut out);
        assert_eq!(out, [5.0, 5.0, 7.0, 7.0]);
    }

    #[test]
    fn distance_profile_is_the_vector_times_the_table_columns() {
        // Two rows of a table over three distances.
        let table = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32; 3];
        distance_profile(&[2.0, 0.5], &table, 3, &mut out);
        assert_eq!(out, [7.0, 14.0, 21.0]);
        // Fewer distances than the table holds: only those are computed.
        let mut out = [0.0f32; 2];
        distance_profile(&[1.0, 1.0], &table, 3, &mut out);
        assert_eq!(out, [11.0, 22.0]);
    }

    #[test]
    #[should_panic(expected = "sinkattn::attend")]
    fn attend_checks_its_lengths() {
        let (q, keys, values) = sample();
        let mut out = [0.0f32; 4];
        attend(shape(&q, &keys, &values, &[0.0; 5], &[0.0, 0.0]), &mut out);
    }
}
