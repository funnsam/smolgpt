use smolgpt::*;

mod tokens;

const CONTEXT_SIZE: usize = 4;
const EMBED_DIM: usize = 8;
const QUERY_DIM: usize = 4;
const NUM_HEADS: usize = 4;
const NUM_LAYERS: usize = 4;

type GptT = GptTrainer<{ tokens::TOKEN_TO_STR.len() }, CONTEXT_SIZE, EMBED_DIM, QUERY_DIM, NUM_HEADS, NUM_LAYERS>;
type GptM = GptModel<{ tokens::TOKEN_TO_STR.len() }, CONTEXT_SIZE, EMBED_DIM, QUERY_DIM, NUM_HEADS, NUM_LAYERS>;

fn main() {
    tokens::print_tokens(&[0, 239, 251, 32, 6, 8, 254, 255]);

    let gpt = GptT::new_random();
    let gpt = gpt.model;

    println!("Parameters: {}", GptM::parameters());

    let mut tokens = [255; CONTEXT_SIZE];
    tokens[0] = 1; // 'the'
    let mut position = 1;

    let mut token = select(gpt.predict_next(&tokens, 1.0));

    while token != 255 {
        print!("{} ", tokens::TOKEN_TO_STR[token]);
        tokens[position] = token;

        token = select(gpt.predict_next(&tokens, 1.0));

        if position > CONTEXT_SIZE - 1 {
            for i in 0..CONTEXT_SIZE {
                if i < CONTEXT_SIZE - 1 {
                    tokens[i] = tokens[i + 1];
                } else {
                    tokens[i] = 0;
                }
            }
        } else {
            position = (position + 1).min(3);
        }
    }
}

fn select<const S: usize>(v: matrix::Vector<S>) -> usize {
    v.inner.iter().map(|a| a[0]).enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0
}
