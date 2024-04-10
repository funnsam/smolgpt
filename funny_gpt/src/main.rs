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
    tokens::print_tokens(&[1, 241, 251, 255]);

    let gpt = GptT::new_random();
    let gpt = gpt.model;

    println!("{}", GptM::parameters());

    let mut tokens = [0; CONTEXT_SIZE];
    tokens[0] = 1; // 'the'
    let mut position = 1;

    let mut token = gpt.predict_next(&tokens);

    while token != 255 {
        print!("{} ", tokens::TOKEN_TO_STR[token]);
        tokens[position] = token;

        token = gpt.predict_next(&tokens);

        if position > CONTEXT_SIZE {
            for i in 0..CONTEXT_SIZE {
                if i < CONTEXT_SIZE - 1 {
                    tokens[i] = tokens[i + 1];
                } else {
                    tokens[i] = 0;
                }
            }
        } else {
            position += 1;
        }
    }
}
