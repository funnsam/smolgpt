use crate::matrix::*;

pub struct GptModel<
    const TOKENS: usize,
    const CONTEXT_SIZE: usize,
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
> {
    embedding: [Vector<EMBED_DIM>; TOKENS],
    unembedding: Matrix<TOKENS, EMBED_DIM>,

    layers: [Layer<EMBED_DIM, QUERY_DIM, NUM_HEADS>; NUM_LAYERS],
}

#[derive(Debug)]
struct Layer<
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
    const NUM_HEADS: usize,
> {
    heads: [Head<EMBED_DIM, QUERY_DIM>; NUM_HEADS],
}

#[derive(Debug)]
struct Head<
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
> {
    key: Matrix<QUERY_DIM, EMBED_DIM>,
    query: Matrix<EMBED_DIM, QUERY_DIM>,

    value_u: Matrix<QUERY_DIM, EMBED_DIM>,
    value_d: Matrix<EMBED_DIM, QUERY_DIM>,
}

pub struct GptTrainer<
    const TOKENS: usize,
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
    const CONTEXT_SIZE: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
> {
    pub model: GptModel<TOKENS, EMBED_DIM, QUERY_DIM, CONTEXT_SIZE, NUM_HEADS, NUM_LAYERS>,
}

impl<
    const TOKENS: usize,
    const CONTEXT_SIZE: usize,
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
> GptModel<TOKENS, CONTEXT_SIZE, EMBED_DIM, QUERY_DIM, NUM_HEADS, NUM_LAYERS> {
    pub const fn parameters() -> usize {
        // embedding
        TOKENS * EMBED_DIM +
        // attention
        (4 * EMBED_DIM * QUERY_DIM) * NUM_HEADS * NUM_LAYERS +
        // up projection
        // down projection
        // unembedding
        TOKENS * EMBED_DIM
    }

    pub fn predict_next(&self, ctx: &[usize; CONTEXT_SIZE]) -> usize {
        let mut embed = Vec::with_capacity(CONTEXT_SIZE);

        for i in ctx.iter() {
            embed.push(self.embedding[*i].clone());
            println!("{}", embed.last().unwrap());
        }

        255
    }
}

impl<
    const TOKENS: usize,
    const CONTEXT_SIZE: usize,
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
> GptTrainer<TOKENS, CONTEXT_SIZE, EMBED_DIM, QUERY_DIM, NUM_HEADS, NUM_LAYERS> {
    pub fn new_random() -> Self {
        let mut embedding = Vec::with_capacity(TOKENS);

        for _ in 0..TOKENS {
            let embed = Vector::new_zeroed();
            embedding.push(embed);
        }

        let unembedding = Matrix::new_zeroed();
        let mut layers = Vec::with_capacity(NUM_LAYERS);

        for _ in 0..NUM_LAYERS {
            let mut heads = Vec::with_capacity(NUM_HEADS);

            for _ in 0..NUM_HEADS {
                let key = Matrix::new_zeroed();
                let query = Matrix::new_zeroed();
                let value_u = Matrix::new_zeroed();
                let value_d = Matrix::new_zeroed();
                heads.push(Head {
                    key,
                    query,
                    value_u,
                    value_d,
                });
            }

            layers.push(Layer {
                heads: heads.try_into().unwrap(),
            });
        }

        Self {
            model: GptModel {
                embedding: embedding.try_into().unwrap(),
                unembedding,
                layers: layers.try_into().unwrap(),
            }
        }
    }
}
