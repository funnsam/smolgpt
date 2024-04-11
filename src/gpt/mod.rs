use crate::*;

mod mlp;

pub struct GptModel<
    const TOKENS: usize,
    const CONTEXT_SIZE: usize,
    const EMBED_DIM: usize,
    const QUERY_DIM: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
> {
    embedding: [Vector<EMBED_DIM>; TOKENS],
    unembedding: Matrix<EMBED_DIM, TOKENS>,

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

    pub fn predict_next(&self, ctx: &[usize; CONTEXT_SIZE], temperature: f32) -> Vector<TOKENS> {
        // embedding
        let mut embed = Vec::with_capacity(CONTEXT_SIZE);

        for i in ctx.iter() {
            embed.push(self.embedding[*i].clone());
        }

        let mut embed = TryInto::<[_; CONTEXT_SIZE]>::try_into(embed).unwrap();

        // TODO: inbetween layers

        // unembedding
        (&self.unembedding * embed.last().unwrap()).softmax(temperature)
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
        use rand::*;
        let mut rng = rngs::StdRng::from_entropy();

        fn fill_random<const W: usize, const H: usize>(mat: &mut Matrix<W, H>, rng: &mut rngs::StdRng) {
            for y in mat.inner.iter_mut() {
                for x in y.iter_mut() {
                    *x = rng.gen();
                }
            }
        }

        let mut embedding = Vec::with_capacity(TOKENS);

        for _ in 0..TOKENS {
            let mut embed = Vector::new_zeroed();
            fill_random(&mut embed, &mut rng);
            embedding.push(embed);
        }

        let mut unembedding = Matrix::new_zeroed();
        fill_random(&mut unembedding, &mut rng);

        let mut layers = Vec::with_capacity(NUM_LAYERS);

        for _ in 0..NUM_LAYERS {
            let mut heads = Vec::with_capacity(NUM_HEADS);

            for _ in 0..NUM_HEADS {
                let mut key = Matrix::new_zeroed();
                fill_random(&mut key, &mut rng);
                let mut query = Matrix::new_zeroed();
                fill_random(&mut query, &mut rng);
                let mut value_u = Matrix::new_zeroed();
                fill_random(&mut value_u, &mut rng);
                let mut value_d = Matrix::new_zeroed();
                fill_random(&mut value_d, &mut rng);

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
