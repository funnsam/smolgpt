use std::ops::*;
use std::fmt;

#[macro_export]
macro_rules! matrix {
    ($w: tt x $h: tt $([$($v: expr),* $(,)?])*) => {{
        let mut m = Matrix::<$w, $h>::new_zeroed();

        matrix_fill!(m, 0, $([$($v,)*])*);

        m
    }};
}

#[allow(unused_macros)] // stoobid rust compiler
macro_rules! matrix_fill {
    ($m: expr, $y: expr, [$($v: expr,)+] $($rest: tt)*) => {{
        matrix_fill!($m, $y, 0, $($v,)*);
        matrix_fill!($m, $y + 1, $($rest)*);
    }};
    ($m: expr, $y: expr, $x: expr, $v: expr, $($rest: tt)*) => {{
        $m[($x, $y)] = $v;
        matrix_fill!($m, $y, $x + 1, $($rest)*);
    }};

    ($m: expr, $y: expr,) => {};
    ($m: expr, $y: expr, $x: expr,) => {};
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Matrix<const W: usize, const H: usize> {
    inner: [[f32; W]; H]
}

impl<const W: usize, const H: usize> Matrix<W, H> {
    pub const fn new_zeroed() -> Self {
        Self {
            inner: [[0.0; W]; H],
        }
    }

    pub fn softmax(&mut self, t: f32) {
        let mut sum = 0.0;
        for y in self.inner.iter_mut() {
            for x in y.iter_mut() {
                *x = (*x / t).exp();
                sum += *x;
            }
        }

        for y in self.inner.iter_mut() {
            for x in y.iter_mut() {
                *x /= sum;
            }
        }
    }

    pub fn softmaxed(&self, t: f32) -> Self {
        let mut c = self.clone();
        c.softmax(t);
        c
    }
}

impl<const W: usize, const H: usize> Index<(usize, usize)> for Matrix<W, H> {
    type Output = f32;

    fn index(&self, i: (usize, usize)) -> &f32 {
        &self.inner[i.1][i.0]
    }
}

impl<const W: usize, const H: usize> IndexMut<(usize, usize)> for Matrix<W, H> {
    fn index_mut(&mut self, i: (usize, usize)) -> &mut f32 {
        &mut self.inner[i.1][i.0]
    }
}

impl<const W: usize, const H: usize> fmt::Display for Matrix<W, H> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut align = [0; W];
        for y in self.inner.iter() {
            for (xi, x) in y.iter().enumerate() {
                align[xi] = align[xi].max(x.to_string().len());
            }
        }

        let inner_space = align.iter().sum::<usize>() + W + 1;
        writeln!(f, "┌{:<inner_space$}┐", "")?;

        for y in self.inner.iter() {
            write!(f, "│ ")?;

            for (xi, x) in y.iter().enumerate() {
                let x = x.to_string();
                write!(f, "{x:^0$} ", align[xi])?;
            }

            writeln!(f, "│")?;
        }

        writeln!(f, "└{:<inner_space$}┘", "")
    }
}

impl<const WAHB: usize, const HA: usize, const WB: usize> Mul<&Matrix<WB, WAHB>> for &Matrix<WAHB, HA> {
    type Output = Matrix<WB, HA>;

    fn mul(self, b: &Matrix<WB, WAHB>) -> Matrix<WB, HA> {
        let mut c = Matrix::new_zeroed();

        for y in 0..HA {
            for x in 0..WB {
                let mut dot = 0.0;

                for i in 0..WAHB {
                    dot += self[(i, y)] * b[(x, i)];
                }

                c[(x, y)] = dot;
            }
        }

        c
    }
}

#[cfg(test)]
#[test]
fn matrix_mul() {
    let a = matrix!(3 x 2
        [1.0, 2.0, 3.0]
        [4.0, 5.0, 6.0]
    );
    let b = matrix!(2 x 3
        [10.0, 11.0]
        [20.0, 21.0]
        [30.0, 31.0]
    );

    let c = &a * &b;

    println!("{a}"); // cargo test -- --nocapture
    println!("{b}");
    println!("{c}");

    assert_eq!(c[(0, 0)], 140.0);
    assert_eq!(c[(1, 0)], 146.0);
    assert_eq!(c[(0, 1)], 320.0);
    assert_eq!(c[(1, 1)], 335.0);
}
