# tensor_algebra

Sparse tensors over arbitrary semirings with generic symbolic keys. A Rust port of [TensorAlgebra.jl](https://github.com/YOUR_USERNAME/TensorAlgebra.jl).

## Concepts

### Semiring

A semiring is a set with two operations ⊕ (addition) and ⊗ (multiplication), an additive identity `zero`, and a multiplicative identity `one`. Multiplication need not be commutative. Implement the `Semiring` trait along with `Add` and `Mul`:

```rust
use tensor_algebra::Semiring;
use std::ops::{Add, Mul};

#[derive(Clone, PartialEq, Debug)]
struct PlusTimes(f64);

impl Add for PlusTimes {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { PlusTimes(self.0 + rhs.0) }
}

impl Mul for PlusTimes {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { PlusTimes(self.0 * rhs.0) }
}

impl Semiring for PlusTimes {
    fn zero() -> Self    { PlusTimes(0.0) }
    fn one() -> Self     { PlusTimes(1.0) }
    fn is_zero(&self) -> bool { self.0 == 0.0 }
}
```

### Tensor

A `Tensor<K, S>` is a graded algebraic structure — a trie of semiring values keyed by `K`. Each node holds:

- `scalar: S` — the grade-0 component
- `dict: HashMap<K, Box<Tensor<K, S>>>` — grade > 0 components

Keys can be anything `Eq + Hash + Clone`: `&str`, `String`, a URI type, a CURIE, etc.

### Operations

| Operation | Description |
|-----------|-------------|
| `a + b` | Componentwise ⊕ at every grade; union of keys |
| `a * b` | Outer (tensor) product; grade adds |
| `Tensor::dot(&a, &b)` | Contracted product; contracts matching leading keys, accumulates with ⊕ |
| `Tensor::scale_left(s, &t)` | Scalar ⊗ tensor |
| `Tensor::scale_right(&t, s)` | Tensor ⊗ scalar |

## Usage

```rust
use tensor_algebra::{Semiring, Tensor};
use std::collections::HashMap;

// Build rank-1 tensors (sparse vectors)
let t1 = Tensor::new(
    PlusTimes(0.0),
    HashMap::from([
        ("a", Box::new(Tensor::leaf(PlusTimes(1.0)))),
        ("b", Box::new(Tensor::leaf(PlusTimes(2.0)))),
    ]),
);
let t2 = Tensor::new(
    PlusTimes(0.0),
    HashMap::from([
        ("b", Box::new(Tensor::leaf(PlusTimes(3.0)))),
        ("c", Box::new(Tensor::leaf(PlusTimes(4.0)))),
    ]),
);

// Addition: union of keys, ⊕ on shared keys
let sum = t1.clone() + t2.clone();
// "a" => 1.0, "b" => 5.0, "c" => 4.0

// Outer product: rank-1 × rank-1 → rank-2
let product = t1.clone() * t2.clone();

// Dot: contracts shared keys → scalar result
// Only "b" is shared: 2.0 * 3.0 = 6.0
let result = Tensor::dot(&t1, &t2);
assert_eq!(result.scalar, PlusTimes(6.0));
```

### Boolean reachability

```rust
use tensor_algebra::{Boolean, Tensor};
use std::collections::HashMap;

let b1 = Tensor::new(
    Boolean(false),
    HashMap::from([
        ("a", Box::new(Tensor::leaf(Boolean(true)))),
        ("b", Box::new(Tensor::leaf(Boolean(true)))),
    ]),
);
let b2 = Tensor::new(
    Boolean(false),
    HashMap::from([
        ("b", Box::new(Tensor::leaf(Boolean(true)))),
        ("c", Box::new(Tensor::leaf(Boolean(true)))),
    ]),
);

// "b" is shared → logical AND then OR → true
let result = Tensor::dot(&b1, &b2);
assert_eq!(result.scalar, Boolean(true));
```

## Included semirings

| Type | ⊕ | ⊗ | Use case |
|------|---|---|----------|
| `PlusTimes<f64>` | `+` | `×` | Standard arithmetic |
| `Boolean` | `\|\|` | `&&` | Reachability |

## License

MIT
