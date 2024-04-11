use crate::*;

pub struct Layer<const PREV_NODES: usize, const NUM_NODES: usize> {
    weights: Matrix<PREV_NODES, NUM_NODES>,
    biases: Vector<NUM_NODES>,
}

// LIGHT:
//    [ w00 w01 ... w0n ][ a0 ]   [ b0 ]
// σ( [ w10 w11 ... w1n ][ a1 ] + [ b1 ] )
//    [ ... ... ... ... ][ .. ]   [ .. ]
//    [ wk0 wk1 ... wkn ][ an ]   [ bn ]

impl<const PREV_NODES: usize, const NUM_NODES: usize> Layer<PREV_NODES, NUM_NODES> {
    pub fn evaluate(&self, values: Vector<PREV_NODES>) -> Vector<NUM_NODES> {
        (&self.weights * &values + &self.biases).sigmoid()
    }
}
