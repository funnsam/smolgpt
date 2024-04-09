mod matrix;
use matrix::*;

pub struct Gpt<const TOKENS: usize, const DIMENSION: usize, const CONTEXT_SIZE: usize> {
    embedding: Matrix<TOKENS, DIMENSION>,
    unembedding: Matrix<TOKENS, DIMENSION>,
}

pub struct GptTrainer<const TOKENS: usize, const DIMENSION: usize, const CONTEXT_SIZE: usize> {
    gpt: Gpt<TOKENS, DIMENSION, CONTEXT_SIZE>,
}
