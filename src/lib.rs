use std::collections::HashMap;
use std::ops::{Add, Mul};

// ── Semiring ──────────────────────────────────────────────────────────────────
//
// Implementors must define:
//   Add, Mul (with Output = Self)
//   zero() — additive identity
//   one()  — multiplicative identity
//   is_zero() — true iff additive identity
//
// Clone is required because values are shared across dict branches during
// outer product and dot.

pub trait Semiring:
    Add<Output = Self> + Mul<Output = Self> + Clone + PartialEq + Sized
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
}

// ── Tensor ────────────────────────────────────────────────────────────────────
//
// A graded algebraic structure over a semiring S with keys of type K.
//   scalar — grade-0 (scalar) component
//   dict   — grade > 0 components; keys map to child Tensors
//
// Invariant: no child in dict satisfies is_zero(). Maintained by add and dot.
// The dict values are Boxed because the struct is recursive.

#[derive(Clone)]
pub struct Tensor<K, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: Semiring,
{
    pub scalar: S,
    pub dict: HashMap<K, Box<Tensor<K, S>>>,
}

impl<K, S> Tensor<K, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: Semiring,
{
    pub fn new(scalar: S, dict: HashMap<K, Box<Tensor<K, S>>>) -> Self {
        Self { scalar, dict }
    }

    pub fn zero() -> Self {
        Self::new(S::zero(), HashMap::new())
    }

    pub fn one() -> Self {
        Self::new(S::one(), HashMap::new())
    }

    /// A rank-0 leaf wrapping a semiring scalar.
    pub fn leaf(s: S) -> Self {
        Self::new(s, HashMap::new())
    }

    pub fn is_zero(&self) -> bool {
        self.scalar.is_zero() && self.dict.is_empty()
    }

    /// Scale every component of a tensor by a semiring scalar (scalar ⊗ tensor).
    pub fn scale_left(a: S, t: &Tensor<K, S>) -> Tensor<K, S> {
        let dict = t
            .dict
            .iter()
            .map(|(k, v)| (k.clone(), Box::new(Self::scale_left(a.clone(), v))))
            .collect();
        Tensor::new(a.clone() * t.scalar.clone(), dict)
    }

    /// Scale every component of a tensor by a semiring scalar (tensor ⊗ scalar).
    pub fn scale_right(t: &Tensor<K, S>, a: S) -> Tensor<K, S> {
        let dict = t
            .dict
            .iter()
            .map(|(k, v)| (k.clone(), Box::new(Self::scale_right(v, a.clone()))))
            .collect();
        Tensor::new(t.scalar.clone() * a.clone(), dict)
    }

    /// Contracted product (dot). Contracts matching leading key levels,
    /// accumulating with ⊕. Mirrors Julia's `dot` exactly.
    pub fn dot(a: &Tensor<K, S>, b: &Tensor<K, S>) -> Tensor<K, S> {
        // scalar × scalar
        let ss = a.scalar.clone() * b.scalar.clone();

        // scalar × b.dict  (a.scalar scales each child of b)
        let st: HashMap<K, Box<Tensor<K, S>>> = b
            .dict
            .iter()
            .map(|(k, v)| (k.clone(), Box::new(Self::scale_left(a.scalar.clone(), v))))
            .collect();

        // a.dict × scalar  (each child of a scaled by b.scalar)
        let ts: HashMap<K, Box<Tensor<K, S>>> = a
            .dict
            .iter()
            .map(|(k, v)| (k.clone(), Box::new(Self::scale_right(v, b.scalar.clone()))))
            .collect();

        // contracted dict × dict: recurse on shared keys, sum with ⊕
        let tt: Tensor<K, S> = {
            let shared_keys: Vec<K> = a
                .dict
                .keys()
                .filter(|k| b.dict.contains_key(*k))
                .cloned()
                .collect();

            shared_keys
                .iter()
                .map(|k| Self::dot(&a.dict[k], &b.dict[k]))
                .fold(Tensor::zero(), |acc, t| acc + t)
        };

        // merge st and ts with ⊕ on shared keys, then add tt
        let merged_st_ts = merge_add(st, ts);
        tt + Tensor::new(ss, merged_st_ts)
    }
}

// ── Addition ──────────────────────────────────────────────────────────────────
//
// Componentwise ⊕ at every grade. Union of keys; shared keys recurse.
// Prunes keys whose children sum to zero.

impl<K, S> Add for Tensor<K, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: Semiring,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut merged = self.dict;
        for (k, v) in rhs.dict {
            let entry = merged.remove(&k);
            let combined = match entry {
                Some(existing) => *existing + *v,
                None => *v,
            };
            if !combined.is_zero() {
                merged.insert(k, Box::new(combined));
            }
        }
        Tensor::new(self.scalar + rhs.scalar, merged)
    }
}

// ── Outer product ─────────────────────────────────────────────────────────────
//
// Tensor product with no contraction. Grade(a * b) = Grade(a) + Grade(b).
// Distributes over addition; hangs b beneath every leaf of a.

impl<K, S> Mul for Tensor<K, S>
where
    K: Eq + std::hash::Hash + Clone,
    S: Semiring,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // scalar × scalar
        let ss = self.scalar.clone() * rhs.scalar.clone();

        // scalar × rhs.dict
        let st: HashMap<K, Box<Tensor<K, S>>> = rhs
            .dict
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Box::new(Tensor::scale_left(self.scalar.clone(), v)),
                )
            })
            .collect();

        // self.dict × scalar
        let ts: HashMap<K, Box<Tensor<K, S>>> = self
            .dict
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Box::new(Tensor::scale_right(v, rhs.scalar.clone())),
                )
            })
            .collect();

        // self.dict × rhs (full tensor product; hangs rhs beneath each child of self)
        let tt: HashMap<K, Box<Tensor<K, S>>> = self
            .dict
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Box::new((**v).clone() * rhs.clone()),
                )
            })
            .collect();

        // merge st, ts, tt with ⊕ on shared keys
        let merged = merge_add(merge_add(st, ts), tt);
        Tensor::new(ss, merged)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Merge two dicts, summing (⊕) values on shared keys. Equivalent to
/// Julia's `merge(+, a, b)`.
fn merge_add<K, S>(
    mut a: HashMap<K, Box<Tensor<K, S>>>,
    b: HashMap<K, Box<Tensor<K, S>>>,
) -> HashMap<K, Box<Tensor<K, S>>>
where
    K: Eq + std::hash::Hash + Clone,
    S: Semiring,
{
    for (k, v) in b {
        let entry = a.remove(&k);
        let combined = match entry {
            Some(existing) => *existing + *v,
            None => *v,
        };
        if !combined.is_zero() {
            a.insert(k, Box::new(combined));
        }
    }
    a
}

// ── Display ───────────────────────────────────────────────────────────────────

impl<K, S> std::fmt::Display for Tensor<K, S>
where
    K: Eq + std::hash::Hash + Clone + std::fmt::Display,
    S: Semiring + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_scalar = !self.scalar.is_zero();
        let has_dict = !self.dict.is_empty();
        if has_scalar {
            write!(f, "{}", self.scalar)?;
            if has_dict {
                write!(f, " + ")?;
            }
        }
        if has_dict {
            write!(f, "{{")?;
            let mut first = true;
            for (k, v) in &self.dict {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", k, v)?;
                first = false;
            }
            write!(f, "}}")?;
        }
        if !has_scalar && !has_dict {
            write!(f, "{}", self.scalar)?;
        }
        Ok(())
    }
}

// ── Example semirings ─────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Debug)]
pub struct PlusTimes<T>(pub T);

impl<T> Add for PlusTimes<T>
where
    T: Add<Output = T> + Clone + PartialEq,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        PlusTimes(self.0 + rhs.0)
    }
}

impl<T> Mul for PlusTimes<T>
where
    T: Mul<Output = T> + Clone + PartialEq,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        PlusTimes(self.0 * rhs.0)
    }
}

impl Semiring for PlusTimes<f64> {
    fn zero() -> Self { PlusTimes(0.0) }
    fn one() -> Self  { PlusTimes(1.0) }
    fn is_zero(&self) -> bool { self.0 == 0.0 }
}

impl std::fmt::Display for PlusTimes<f64> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Boolean(pub bool);

impl Add for Boolean {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Boolean(self.0 || rhs.0) }
}

impl Mul for Boolean {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Boolean(self.0 && rhs.0) }
}

impl Semiring for Boolean {
    fn zero() -> Self    { Boolean(false) }
    fn one() -> Self     { Boolean(true) }
    fn is_zero(&self) -> bool { !self.0 }
}

impl std::fmt::Display for Boolean {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── Tests (mirrors examples.jl) ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf_pt(v: f64) -> Box<Tensor<&'static str, PlusTimes<f64>>> {
        Box::new(Tensor::leaf(PlusTimes(v)))
    }

    fn leaf_bool(v: bool) -> Box<Tensor<&'static str, Boolean>> {
        Box::new(Tensor::leaf(Boolean(v)))
    }

    #[test]
    fn test_addition_plus_times() {
        let t1 = Tensor::new(
            PlusTimes(0.0),
            HashMap::from([("a", leaf_pt(1.0)), ("b", leaf_pt(2.0))]),
        );
        let t2 = Tensor::new(
            PlusTimes(0.0),
            HashMap::from([("b", leaf_pt(3.0)), ("c", leaf_pt(4.0))]),
        );
        let sum = t1 + t2;
        // :a => 1, :b => 5, :c => 4
        assert_eq!(sum.dict["a"].scalar, PlusTimes(1.0));
        assert_eq!(sum.dict["b"].scalar, PlusTimes(5.0));
        assert_eq!(sum.dict["c"].scalar, PlusTimes(4.0));
    }

    #[test]
    fn test_dot_rank1_plus_times() {
        let t1 = Tensor::new(
            PlusTimes(0.0),
            HashMap::from([("a", leaf_pt(1.0)), ("b", leaf_pt(2.0))]),
        );
        let t2 = Tensor::new(
            PlusTimes(0.0),
            HashMap::from([("b", leaf_pt(3.0)), ("c", leaf_pt(4.0))]),
        );
        // only :b shared; 2 * 3 = 6
        let result = Tensor::dot(&t1, &t2);
        assert_eq!(result.scalar, PlusTimes(6.0));
    }

    #[test]
    fn test_dot_boolean() {
        let b1 = Tensor::new(
            Boolean(false),
            HashMap::from([("a", leaf_bool(true)), ("b", leaf_bool(true))]),
        );
        let b2 = Tensor::new(
            Boolean(false),
            HashMap::from([("b", leaf_bool(true)), ("c", leaf_bool(true))]),
        );
        // :b shared → scalar true
        let result = Tensor::dot(&b1, &b2);
        assert_eq!(result.scalar, Boolean(true));
    }

    #[test]
    fn test_zero_pruning() {
        let t1 = Tensor::new(
            PlusTimes(0.0),
            HashMap::from([("a", leaf_pt(1.0))]),
        );
        let t2 = Tensor::new(
            PlusTimes(0.0),
            HashMap::from([("a", leaf_pt(-1.0))]),
        );
        let sum = t1 + t2;
        // :a cancels, dict should be empty
        assert!(sum.dict.is_empty());
    }
}
