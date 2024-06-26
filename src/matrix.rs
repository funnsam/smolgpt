use std::fmt;
use std::ops::*;

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

#[macro_export]
macro_rules! vector {
    ($h: tt [$($val: expr),* $(,)?]) => {
        matrix!(1 x $h $([$val])*)
    };
}

pub type Vector<const H: usize> = Matrix<1, H>;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Matrix<const W: usize, const H: usize> {
    pub inner: [[f32; W]; H],
}

impl<const W: usize, const H: usize> Matrix<W, H> {
    pub const fn new_zeroed() -> Self {
        Self {
            inner: [[0.0; W]; H],
        }
    }

    pub fn softmax(mut self, t: f32) -> Self {
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

        self
    }

    pub fn softmaxed(&self, t: f32) -> Self {
        self.clone().softmax(t)
    }

    pub fn softmax_by_column(&mut self, t: f32) {
        for y in self.inner.iter_mut() {
            let mut sum = 0.0;

            for x in y.iter_mut() {
                *x = (*x / t).exp();
                sum += *x;
            }

            for x in y.iter_mut() {
                *x /= sum;
            }
        }
    }

    pub fn softmaxed_by_column(&self, t: f32) -> Self {
        let mut c = self.clone();
        c.softmax_by_column(t);
        c
    }

    pub fn sigmoid(mut self) -> Self {
        for y in self.inner.iter_mut() {
            for x in y.iter_mut() {
                // LIGHT:
                //             1
                // σ(x) = ------------
                //         1 + e^(-x)
                *x = 1.0 / (1.0 + (-*x).exp());
            }
        }

        self
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
                write!(f, "{x:^0$} ", align[xi])?;
            }

            writeln!(f, "│")?;
        }

        writeln!(f, "└{:<inner_space$}┘", "")
    }
}

impl<const WAHB: usize, const HA: usize, const WB: usize> Mul<&Matrix<WB, WAHB>>
    for &Matrix<WAHB, HA>
{
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

impl<const W: usize, const H: usize> Add<&Self> for Matrix<W, H> {
    type Output = Matrix<W, H>;

    fn add(mut self, b: &Matrix<W, H>) -> Matrix<W, H> {
        for (yi, y) in self.inner.iter_mut().enumerate() {
            for (xi, x) in y.iter_mut().enumerate() {
                *x += b[(xi, yi)];
            }
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_mul() {
        let a = matrix!(
            3 x 2
            [1.0, 2.0, 3.0]
            [4.0, 5.0, 6.0]
        );
        let b = matrix!(
            2 x 3
            [10.0, 11.0]
            [20.0, 21.0]
            [30.0, 31.0]
        );

        let c = &a * &b;

        println!("{a}"); // cargo test -- --nocapture
        println!("{b}");
        println!("{c}");

        assert_eq!(c, matrix!(
            2 x 2
            [140.0, 146.0]
            [320.0, 335.0]
        ));
    }

    #[test]
    fn matrix_add() {
        let a = matrix!(
            3 x 2
            [1.0, -1.0, 2.0]
            [0.0, 3.0, 4.0]
        );
        let b = matrix!(
            3 x 2
            [2.0, -1.0, 5.0]
            [7.0, 1.0, 4.0]
        );

        println!("{a}"); // cargo test -- --nocapture

        let c = a + &b;

        println!("{b}");
        println!("{c}");

        assert_eq!(c, matrix!(
            3 x 2
            [3.0, -2.0, 7.0]
            [7.0, 4.0, 8.0]
        ));
    }

    #[test]
    fn vec_dot() {
        let a = matrix!(
            3 x 1
            [1.0, 2.0, 3.0]
        );
        let b = vector!(
            3
            [4.0, -5.0, 6.0]
        );

        assert_eq!(&a * &b, matrix!(1 x 1 [12.0]));
    }
}
